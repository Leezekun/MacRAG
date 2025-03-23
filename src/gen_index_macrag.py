import json
import os
import time
from sentence_transformers import SentenceTransformer
import faiss
import argparse
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter

import copy
import jsonlines
import re
import numpy as np

from glob import glob

from dotenv import load_dotenv
load_dotenv()

## LLM Prompts
persona = '''
You are a senior reporter and have been working in the media industry for a long time.
Your primary objective is for a given text, summarize it to an appropriate length.
As an analytical and systematic thinker, please carefully summarize the documents.
'''
 
instruction = '''
Use the following steps and respond to user inputs.
Please think step by step and do not restate each step before proceeding.
 
Step 1:
The user will provide document information.
 
Step 2:
Please proceed with the summary in the following order based on the given document
    1) Please write the Title and Key words of documents.
    2) Organize subheadings with considering given page_content, but make sure to mention what you're covering. If you have tables or graph then please include the table titles or column information and explain the contents in the subheadings.
    3) Summarize the given page_content while maintaining contexts with specified information in details from the given texts and the detailed information of numbers as much as possible.
    4) Output Length of Summary should be less then length 500. ex) "keypoint" -> length is 8
    5) Please think by Document so if input is two Document information then output also should be two.
 
 
Step 3:
Provide the final output in JSON format as follows:
[ 
    {"Title":"...", "Keywords":"...", "Subheadings":"...", "Summary":"..."}
]
Please be careful about number of output, it depends on the number of input.
Please double check that the keys in the outputs are unique, that is, there is only one of each "Title,"  "Keywords", "Subheadings", and "Summary" as the keys in the output.
 
Do not display the output from Step 1, Step 2 and provide only the outputs of Step 3.
'''

assistant = '''
[   
    {
    "Title": "Anarchism: Philosophy, History, and Modern Resurgence",
    "Keywords": "Anarchism, Anti-Authority, Stateless Societies, Revolutionary Strategies, Modern Resurgence",
    "Subheadings": "From Enlightenment Roots to Modern Movements: The Historical and Ideological Development of Anarchism",
    "Summary": "Anarchism is a political philosophy opposing all forms of authority, aiming to abolish institutions like the state and capitalism. It advocates for stateless societies and voluntary associations. Modern anarchism emerged from the Enlightenment and was influential in late 19th and early 20th-century workers' struggles, participating in revolutions like the Paris Commune and Spanish Civil War. The movement resurged in the late 20th and early 21st centuries within anti-capitalist and anti-globalization movements, using both revolutionary and evolutionary strategies."
    }
]
'''

### Utility Functions.
def store_json(json_object, store_path):
    with open(store_path, "w", encoding='utf-8') as fout:
        json.dump(json_object, fout, ensure_ascii=False)

def store_jsonl(list_of_objects, store_path):
    with jsonlines.open(store_path, "w") as writer:
        writer.write_all(list_of_objects)

def read_json(json_file_path):
    '''
    read json file
    '''
    try:
        with open(json_file_path, "r") as file:
            data = json.load(file)
    except:
        try:
            with open(json_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except:
            pass
    return data

def file_exist(file_path):
    return os.path.exists(file_path)

### Main Functions

def split_into_chunks(content, chunk_size, overlap_chunk_size):
    '''
    split content corresponding with chunk_size, chunk_overlap_size
    '''
    stop_list = ['!', '。', '，', '！', '?', '？', ',', '.', ';', "\n"]
    text_splitter = RecursiveCharacterTextSplitter(
                                                chunk_size = chunk_size,   
                                                chunk_overlap = overlap_chunk_size, # default
                                                length_function = len,
                                                separators=stop_list,
                                                )
    sentences = text_splitter.split_text(content)
    return sentences

def split_sentences(content, chunk_size, min_sentence, overlap):
    '''
    split strategy for LongRAG paper
    '''
    stop_list = ['!', '。', '，', '！', '?', '？', ',', '.', ';']
    split_pattern = f"({'|'.join(map(re.escape, stop_list))})"
    sentences = re.split(split_pattern, content)
    
    if len(sentences) == 1:
        return sentences
    
    sentences = [sentences[i] + sentences[i+1] for i in range(0, len(sentences) - 1, 2)]
    chunks = []
    temp_text = ''
    sentence_overlap_len = 0
    start_index = 0

    for i, sentence in enumerate(sentences):
        temp_text += sentence
        if get_word_count(temp_text) >= chunk_size - sentence_overlap_len or i == len(sentences) - 1:
            if i + 1 > overlap:
                sentence_overlap_len = sum([get_word_count(sentences[j]) for j in range(i+1-overlap, i+1)])
            if chunks:
                if start_index > overlap:
                    start_index -= overlap
            chunk_text = ''.join(sentences[start_index:i+1])
            if not chunks:
                chunks.append(chunk_text)
            elif i == len(sentences) - 1 and (i - start_index + 1) < min_sentence:
                chunks[-1] += chunk_text
            else:
                chunks.append(chunk_text)
            temp_text = ''
            start_index = i + 1
    
    return chunks

def generate_batch_prompt_json(text):
    prompt = f"{persona}\n{instruction}\n\nInput text: {text}"
    return {"request":{"contents": [{"role": "user", "parts": [{"text": prompt}]}]}}

def split_documents_into_chunks(documents, chunk_size, overlap_chunk_size):
    document_chunks = []
    chunk_id = 0
    for idx, item in tqdm(enumerate(documents), total=len(documents), desc="Processing data"):
        content = item.get("paragraph_text") or item.get("ch_content") or item.get("ch_contenn")
        chunks = split_into_chunks(content, chunk_size, overlap_chunk_size)
        for chunk in chunks:
            document_chunks.append({ 'chunk_id': chunk_id, 'document_id': idx, 'chunk': chunk })
            chunk_id = chunk_id + 1
    return document_chunks

def extract_title(text):
    '''
    # Use regular expressions to extract the Title part that follows "Passage number:"
    '''    
    pattern = r'Passage \d+:\n(.*?)(?=\n)'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0]

def extract_text_between_markers(text):
    '''
    extract contents between 'page_content:' and 'metadata'
    '''    
    match = re.search(r'page_content:(.*?)metadata', text, re.DOTALL)
    return match.group(1).strip() if match else None

def divide_into_100_parts(number):
    '''
    split docs 100 for check progress bar
    '''
    step = number // 100
    ranges = [(i * step, (i + 1) * step) for i in range(100)]
    return ranges

def get_llm_response(args, input_text):
    '''
    run llm
    '''
    if "gpt" in args.summary_model:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            result = client.chat.completions.create(
                                model=args.summary_model,
                                messages=[
                                        {"role": "system", "content": f"{persona} \n {instruction}"},
                                        {"role": "assistant", "content" : f"{assistant}"},
                                        {"role": "user", "content": f"Input: {input_text}"}
                                        ],
                                temperature= 0,

                            )
            output = result.choices[0].message.content
        except Exception as e:
            print(e)
            output = '{}'.format(e)
    else:
        raise ValueError(f"We only use gpt model for now.")
    return output


### main function
def make_raw_chunk(args):
    '''
    input: document (paragraph)
    
    1. document -> raw_chunk
    2. make raw_txt file
    3. make vector.index (=VectorDB), chunks.json, id_to_rawid.json
    
    output: vector.index (=VectorDB), chunks.json, id_to_rawid.json
    '''
    emb_dict = {"intfloat/multilingual-e5-large":"e5", "BAAI/bge-m3":"bge"}
    ## 0. Read Document file
    print("0. Read Document file ...")
    document = read_json(args.document_chunk_path) # ../data/corpus/raw/hotpotqa.json
    # document = document[:100] # in gen_index, test
    raw_txt_path = "../data/raw_data/{}_{}/{}/raw_txt/".format(args.chunk_size, args.overlap_chunk_size, args.dataset) 
    os.makedirs(raw_txt_path, exist_ok=True)            
    print("    Read Document file Done !!!")
    ## 1&2. make chunk and write chunk_*.txt
    id_to_rawid = {}
    raw_chunks = []
    chunk_id = 0
    for idx, item in tqdm(enumerate(document), total=len(document), desc="1&2. make raw_chunk and raw_txt file"):        
        content = item.get("paragraph_text") or item.get("ch_content") or item.get("ch_contenn")
        ##################### Long-RAG paper code ###############################################################
        # chunks = split_sentences(content, chunk_size, min_sentence, overlap)      
        # for i, chunk in enumerate(chunks):
        #     id_to_rawid[len(processed_chunks) + i] = idx
        # processed_chunks.extend(chunks)
        #########################################################################################################
        ## 1. document -> raw_chunk (split content with considering chunk_size, overlap size)
        chunks = split_into_chunks(content, args.chunk_size, args.overlap_chunk_size)
        for chunk in chunks:
            metadata_info = {'source': extract_title(content), 'chunk_id':chunk_id, 'document_id':idx}
            id_to_rawid[chunk_id] = idx
            raw_chunk = "page_content: " + chunk + "\n\n" + "metadata: " + "{}".format(metadata_info)
            raw_chunks.append(raw_chunk)
            
            ## 2. make raw_txt file
            with open(raw_txt_path + "chunk_{}.txt".format(chunk_id), "w", encoding="utf-8") as f:
                f.write(raw_chunk)
            
            chunk_id = chunk_id + 1
    
    # 3.make vector.index (=VectorDB), chunks.json, id_to_rawid.json
    print("3. make vector.index (=VectorDB), chunks.json, id_to_rawid.json ...")
    save_path = "../data/corpus/processed/raw_{}_{}_{}/{}/".format(args.chunk_size, args.overlap_chunk_size, emb_dict[args.emb_model] ,args.dataset)
    os.makedirs(save_path, exist_ok=True)
    
    ## 3.1. chunks.json
    with open(f"{save_path}/chunks.json", "w", encoding='utf-8') as fout:
        json.dump(raw_chunks, fout, ensure_ascii=False)
    ## 3.2. id_to_rawid.json
    with open(f"{save_path}/id_to_rawid.json", "w", encoding='utf-8') as fout:
        json.dump(id_to_rawid, fout, ensure_ascii=False)
    
    ## 3.3. vector.index (=VectorDB)
    print("Embedding Model: {}".format(args.emb_model))
    import time
    model = SentenceTransformer(args.emb_model)
    start_emb_time = time.time()
    steps = divide_into_100_parts(len(raw_chunks))
        
    embeddings_temp = []
    for step in tqdm(steps[:], desc="make embedding vector..."):
        embedding = model.encode(raw_chunks[step[0]:step[1]])
        embeddings_temp += [embedding]        
    embeddings = np.vstack(embeddings_temp)        
    end_emb_time = time.time()
    # print("embedding time: {}".format(end_emb_time - start_emb_time))
    
    start_faiss_time = time.time()
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, save_path + "vector.index")
    end_faiss_time = time.time()
    # print("faiss write time: {}".format(end_faiss_time - start_faiss_time))    
    print("    make vector.index (=VectorDB), chunks.json, id_to_rawid.json Done !!!")
    
    
def make_summary(args):
    '''
    make summary and summary would be 1:1 mapping with raw_chunk
    1. read raw chunk (txt format)
    2. Summary by LLM
    '''
    max_try = 3    
    
    ## 1. Read raw_txt files
    print("1. Read raw_txt files ...")
    file_list = sorted([x for x in glob(f"{args.chunk_dir}"+"/raw_txt/*") if ".txt" in x] , key=lambda x: int(x.split('chunk_')[1].split('.')[0]))
    print("    Read raw_txt files Done !!!")
    summary_txt_dir = args.chunk_dir + "/summary_txt/"
    os.makedirs(summary_txt_dir, exist_ok=True)
    print("2. Summmary Start...")
    for num, filepath in tqdm(enumerate(file_list[:]), total = len(file_list[:]), desc="    summary each chunk ..."):
        try:
            #          
            with open(filepath, 'r', encoding='utf-8') as f:
                org_chunk = f.read()
            
            try_= 0 
            answer_status = False
            standard = True
            
            # max try 3
            while try_<max_try and answer_status != standard: 
                print("    try - {}".format(try_+1))
                try:                    
                    start_time = time.time()
                    result = get_llm_response(args, org_chunk)                    
                    print("    summary duration: {}".format(time.time() - start_time))                    
                    
                    ## post-process
                    #### - check title, keywords, subheadings, summary is in summary result
                    json_format={}
                    matches = re.finditer(r'"([^"]+)"\s?:\s?"(.+?)"', result)
                    for match in matches:
                        try:
                            key = match.group(1)
                            value = match.group(2)
                            json_format[key] = value
                        except:
                            pass
                    
                    answer_list = ['title', 'keywords', 'subheadings', 'summary']
                    check_json_format={}
                    check_json_format = {key.lower() : value for key, value in json_format.items() if key.lower() in answer_list}

                    if sorted([x for x in check_json_format.keys()]) == sorted(answer_list):
                        # break loop
                        answer_status=True
                    else:
                        pass
                except:
                    pass
                try_ +=1
                
            ### until max_try or loop closed
            if sorted([x for x in check_json_format.keys()]) == sorted(answer_list):
                pass
            else:
                print("    summary fail. check summary_fail_list folder")
                os.makedirs(f"{summary_txt_dir}"+"summary_fail_list/", exist_ok=True)
                with open(f"{summary_txt_dir}"+"summary_fail_list/" + os.path.basename(filepath), 'w', encoding='utf-8') as f:
                    f.write('')
            
            json_format={}        
            with open(filepath, 'r', encoding='utf-8') as f:
                raw_txt_docs = f.readlines()
                raw_metadata = []
                for i, elem in enumerate(raw_txt_docs):
                    if elem.startswith('metadata:'):
                        if "\n" in elem:
                            elem = elem.replace("\n","")
                        elem = elem.replace('metadata:','')
                        raw_meta = eval(elem)
                        raw_metadata.append(raw_meta)

            json_format['Source'] = raw_metadata[0]['source'] # raw_txt
            json_format['chunk_id'] = raw_metadata[0]['chunk_id'] # raw_txt
            json_format['Title'] = check_json_format['title'] # LLM
            json_format['Keywords'] = check_json_format['keywords'] # LLM
            json_format['Subheadings'] = check_json_format['subheadings'] # LLM
            json_format['Summary'] = check_json_format['summary'] # LLM
            with open(f'{summary_txt_dir}' + os.path.basename(filepath), 'w', encoding="utf-8") as f:
                f.write('{}'.format(json_format))
                
            print("    Summary and Save Done !!!")
            
        except:
            print("    summary fail. check summary_fail_list folder")
            os.makedirs(f"{summary_txt_dir}"+"summary_fail_list/", exist_ok=True)
            
            with open(f"{summary_txt_dir}"+"summary_fail_list/" + os.path.basename(filepath), 'w', encoding='utf-8') as f:
                f.write('')

def summary_chunk_slicing(args):
    '''
    1. read summary_chunk (txt file)
    2. split only summary part (source, title, keywords, summary ... are in summary_chunk )
    3. make vector.index (=VectorDB), chunks.json, id_to_rawid.json
    '''
    
    emb_dict = {"intfloat/multilingual-e5-large":"e5", "BAAI/bge-m3":"bge"}
    #save_path = "../data/corpus/processed/sum_{}_{}_raw_{}_{}/".format(args.chunk_size, args.overlap_chunk_size, args.chunk_dir.split("/")[-2] ,emb_dict[args.emb_model])
    save_path = "../data/corpus/processed/sum_{}_{}_raw_{}_{}/{}/".format(args.chunk_size, args.overlap_chunk_size, args.chunk_dir.split("/")[-2] ,emb_dict[args.emb_model], args.dataset)
    os.makedirs(save_path, exist_ok=True)

    ## 0. Read summary chunk files
    print("Read summary chunk files ...")
    summary_file_list = sorted([x for x in glob(f"{args.chunk_dir}"+"/summary_txt/*") if ".txt" in x], key=lambda x: int(x.split('chunk_')[1].split('.')[0]))
    print("Read summary chunk files Done !!!")
    
    
    processed_chunks = []; metadatas =[]
    text_splitter = RecursiveCharacterTextSplitter(
                                                    chunk_size = args.chunk_size,   
                                                    chunk_overlap = args.overlap_chunk_size, # default
                                                    length_function = len,
                                                    separators=["\n\n", "\n", " ", ""],
                                                    )
    
        
    id_to_rawid = {}
    for file in tqdm(summary_file_list, desc="make vector.index (=VectorDB), chunks.json, id_to_rawid.json"):
        processed_chunk = []
        try:
            with open(file, "r") as f:
                df = f.read()
        except:
            # with open(file, "r", encoding="utf-8") as f:
            with open(file, "r", encoding="cp949") as f:
                df = f.read()
        
        try:
            with open(file.replace("summary_txt", "raw_txt"), "r") as f:
                raw_df = f.read()
        except:
            with open(file.replace("summary_txt", "raw_txt"), "r", encoding="utf-8") as f:
                raw_df = f.read()
        
        raw_chunks = extract_text_between_markers(df)
        metadata = eval(raw_df.split("metadata: ")[-1])
        
        try:
            json_format = eval(df)
            ## temp_code
            json_format = {'chunk_id' if key == 'chunk_num' else key: value for key, value in json_format.items()}

            documents = text_splitter.split_text(json_format['Summary'])
            for doc in documents:
                json_format['Summary'] = doc
                processed_chunk += ['{}'.format(json_format)]
                metadatas += [{'source': json_format['Source'], 'chunk_id': json_format['chunk_id'], 'document_id': metadata['document_id']}]
        except:
            pass
        
        # 추가로 하나 더! (Summary 외의 정보들 넣기)
        processed_chunk += ['{}'.format({k: v for k, v in json_format.items() if k != 'Summary'})]
        metadatas += [{'source': json_format['Source'], 'chunk_id': json_format['chunk_id'], 'document_id': metadata['document_id']}]

        # id_to_rawid 생성
        for i in range(len(processed_chunk)):
            id_to_rawid[len(processed_chunks) + i] = metadata['document_id']
        
        processed_chunks += processed_chunk
    
    
    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/chunks.json", "w", encoding='utf-8') as fout:
        json.dump(processed_chunks, fout, ensure_ascii=False)
    with open(f"{save_path}/id_to_rawid.json", "w", encoding='utf-8') as fout:
        json.dump(id_to_rawid, fout, ensure_ascii=False)
    
    
    print("Embedding Model: {}".format(args.emb_model))
    import time
    model = SentenceTransformer(args.emb_model)
    start_emb_time = time.time()
    steps = divide_into_100_parts(len(processed_chunks))
        
    embeddings_temp = []
    for step in tqdm(steps[:], desc="make embedding vector..."):
        embedding = model.encode(processed_chunks[step[0]:step[1]])
        embeddings_temp += [embedding]        
    embeddings = np.vstack(embeddings_temp)        
    end_emb_time = time.time()
    # print("embedding time: {}".format(end_emb_time - start_emb_time))
    
    start_faiss_time = time.time()
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, save_path + "vector.index")
    end_faiss_time = time.time()
    
    

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    ## used for make_raw_chunk
    parser.add_argument("--dataset", type=str, choices=["hotpotqa", "2wikimultihopqa", "musique"], default="hotpotqa", help="Name of the dataset")
    parser.add_argument('--document_chunk_path', type=str, 
                        choices=["../data/corpus/raw/hotpotqa.json", "../data/corpus/raw/2wikimultihopqa.json", "../data/corpus/raw/musique.json"], 
                        default="../data/corpus/raw/hotpotqa.json", 
                        help="document - json format file path")    
    parser.add_argument('--chunk_size', type=int, default=1500)
    parser.add_argument('--overlap_chunk_size', type=int, default=500)
    parser.add_argument('--emb_model', type=str, default="intfloat/multilingual-e5-large",choices=['intfloat/multilingual-e5-large', 'BAAI/bge-m3'], help="emb model name")
    
    ## used for make_summary
    parser.add_argument("--summary_model", type=str, default="gpt-4o", help="LLM model for summarization")
    parser.add_argument("--chunk_dir", type=str, default="../data/raw_data/1500_500/hotpotqa", help="raw chunk path (txt file)")
    
    # New arguments for pipeline control
    parser.add_argument("--step", type=int, default=0, help="Which step to run: 0=all, 1=make_raw_chunk, 2=make_summary, 3=summary_chunk_slicing")
    
    return parser.parse_args()

def get_word_count(text):
    """Count the number of words in text"""
    return len(text.split())

model2path = {
    "emb_model_main": "intfloat/multilingual-e5-large",
    "emb_model_sub": "BAAI/bge-m3"
}

if __name__ == '__main__':
    args = parse_arguments()
    
    # Make sure the dataset and all relevant paths are consistent
    if args.dataset not in args.document_chunk_path:
        raise ValueError(f"Dataset {args.dataset} is not consistent with the provided document_chunk_path {args.document_chunk_path}")
    if args.dataset not in args.chunk_dir:
        raise ValueError(f"Dataset {args.dataset} is not consistent with the provided chunk_dir {args.chunk_dir}")
    
    # Set dynamic output paths based on arguments
    if not args.chunk_dir:
        args.chunk_dir = f"../data/raw_data/{args.chunk_size}_{args.overlap_chunk_size}/{args.dataset}"
    
    # Determine which steps to run
    run_all = args.step == 0
    
    # Step 1: Process raw documents into chunks
    if run_all or args.step == 1:
        print("\n" + "="*50)
        print("STEP 1: CREATING RAW CHUNKS")
        print("="*50)
        make_raw_chunk(args)
        print("Raw chunk processing complete!")
    
    # Step 2: Generate summaries for each chunk
    if run_all or args.step == 2:
        print("\n" + "="*50)
        print("STEP 2: GENERATING SUMMARIES")
        print("="*50)
        make_summary(args)
        print("Summary generation complete!")
    
    # Step 3: Process and index summaries
    if run_all or args.step == 3:
        print("\n" + "="*50)
        print("STEP 3: PROCESSING SUMMARIES AND CREATING INDICES")
        print("="*50)
        summary_chunk_slicing(args)
        print("Summary processing and indexing complete!")
    
    if run_all:
        print("\n" + "="*50)
        print("ALL STEPS COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Data for {args.dataset} has been processed and indexed.")
        print(f"You can find the results in:")
        emb_dict = {"intfloat/multilingual-e5-large":"e5", "BAAI/bge-m3":"bge"}
        save_path = f"../data/corpus/processed/sum_{args.chunk_size}_{args.overlap_chunk_size}_raw_{args.chunk_dir.split('/')[-2]}_{emb_dict[args.emb_model]}/{args.dataset}/"
        print(f"- {save_path}")
    
