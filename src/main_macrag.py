import sys

# sys.path.append('')
# sys.path.append('/usr/miniconda3/envs/paper/lib/python311.zip')
# sys.path.append('/usr/miniconda3/envs/paper/lib/python3.11')
# sys.path.append('/usr/miniconda3/envs/paper/lib/python3.11/lib-dynload')
# sys.path.append('/home/PH_sy_ji/.local/lib/python3.11/site-packages')
# sys.path.append('/usr/miniconda3/envs/paper/lib/python3.11/site-packages')

import re
import json
import faiss
from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, \
    AutoModelForSequenceClassification
from transformers.generation.utils import GenerationConfig
import numpy as np
import pandas as pd
import torch
import os
import random
from sentence_transformers import SentenceTransformer
from datetime import datetime
import backoff
import logging
import argparse
import yaml
from metric import F1_scorer
from api import call_api
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from collections import defaultdict

logger = logging.getLogger()

choices = [
    "glm-4", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-0125", "chatGLM3-6b-8k","LongAlign-7B-64k", "Llama3-70b-8k", "Llama2-13b-chat-longlora", "LongRAG-chatglm3-32k", "LongRAG-qwen1.5-32k","LongRAG-vicuna-v1.5-16k", "LongRAG-llama3-8k",  "LongRAG-llama2-4k",
    "vicuna-7b-v1.5-16k","llama3-8b-instruct-8k", "chatglm3-6b-32k",    
    "gpt-4o",  "gemini-1.5-pro", "gemini-1.5-flash",
    "mistral-7b-instruct-v0.1",
    "mistral-7b-instruct-v0.3",
    "llama3.1-8b-instruct",
    "qwen1.5-7b-chat-32k", 
    "qwen2.5-7b-instruct",
]
 

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["hotpotqa", "2wikimultihopqa", "musique"], default="hotpotqa",
                    help="Name of the dataset")
parser.add_argument('--top_k1', type=int, default=100, help="Number of candidates after initial retrieval")
parser.add_argument('--top_k2', type=int, default=7, help="Number of candidates after reranking")
parser.add_argument('--model', type=str, choices=choices, default="chatglm3-6b-32k", help="Model for generation")
parser.add_argument('--lrag_model', type=str, choices=choices, default="", help="Model for LongRAG")
parser.add_argument('--rb', action="store_true", default=False, help="Vanilla RAG")
parser.add_argument('--raw_pred', action="store_true", default=False, help="LLM direct answer without retrieval")
parser.add_argument('--rl', action="store_true", default=False, help="RAG-Long")
parser.add_argument('--ext', action="store_true", default=False, help="Only using Extractor")
parser.add_argument('--fil', action="store_true", default=False, help="Only using Extractor")
parser.add_argument('--ext_fil', action="store_true", default=False, help="Using Extractor and Filter")
parser.add_argument('--ext_rb', action="store_true", default=False, help="Only using Extractor version with R&B")
parser.add_argument('--rb_ext_fil', action="store_true", default=False,
                    help="Using Extractor and Filter with R&B instead of involving the entire documents")
parser.add_argument('--MaxClients', type=int, default=1)
parser.add_argument('--log_path', type=str, default="")
parser.add_argument('--r_path', type=str, default="processed/sum_600_400_raw_e5", help="Path to the vector database")
parser.add_argument('--emb_model_name', type=str, choices=["BAAI/bge-m3", "intfloat/multilingual-e5-large"],
                    default="intfloat/multilingual-e5-large", help="data name")
parser.add_argument('--version', type=str, default="v1", help="exp version")
parser.add_argument('--rerun', action="store_true", default=False, help="Rerun in case of 'None' from API Pred Answer ")

parser.add_argument('--chunk_size', type=int, default=1500)
parser.add_argument('--overlap_chunk_size', type=int, default=500)

parser.add_argument('--temperature', type=float, default=0.0001)
parser.add_argument('--look_up_initial_chunk', action="store_true", default=True,
                    help="True: Look up initial chunk from the slice of the initial chunk's summary chunk, False: Use the chunk retrieval-level itself")
parser.add_argument('--upscaling', type=int, default=1,
                    help="1: Scaling Inference until top_k2 by consuming remained candidates within upsample_top_k2, 0: Basic")
parser.add_argument('--upsample_factor', type=int, default=4,
                    help="4: Since there will be deduplication of overlapped chunks from the slices, for 450/300 we need at least this")
parser.add_argument('--chunk_ext', type=int, default=1,
                    help="Chunk extention window, 0: without context extension, 1: See neighbors to extend chunk's context window for the propagation, 2: Includes neighbors' nightbors too for the propagation")
parser.add_argument('--merge_version', type=int, default=1,
                    help="1: Merge connected extended chunks and concatenate all the chunks in the same document or 2: Merge only connected extended chunks in the same document")
parser.add_argument('--with_reranking', type=int, default=1,
                    help="1: Retireval Performance with reranking or 0: without reranking")
parser.add_argument('--reranking_model', type=str, choices=["bge_m3", "marco_MiniLM"], default="marco_MiniLM",
                    help="Reranking Model Name")

parser.add_argument('--prompt_version', type=int, default=1,
                    help="0. Answer based on only LLM's Knowledge, 1. Basic Prompt in LongRAG, 2. Answer based on only Retrieved Info, 3. Modified Prompt of LongRAG")

args = parser.parse_args()


def file_exists(chunk_id, args):
    file_path = "../data/raw_data/{}_{}/{}/raw_txt/chunk_{}.txt".format(
        args.chunk_size, args.overlap_chunk_size, args.dataset, chunk_id)
    return os.path.exists(file_path)


def count_keywords(predictions, keywords):
    counts = {keyword: 0 for keyword in keywords}
    for pred in predictions:
        for keyword in keywords:
            counts[keyword] += pred.lower().count(keyword)
    return counts


def save_results_to_csv(eval_result, log_path, dataset_name):
    f1_scores = eval_result["F1"]
    doc_lengths = eval_result["doc_len"]
    count_none = eval_result["Count_None"]
    count_provide = eval_result["Count_Provide"]

    # Dataframe for metrics
    df_f1 = pd.DataFrame(f1_scores, index=[0])
    df_doc_len = pd.DataFrame(doc_lengths, index=[0])
    df_count_none = pd.DataFrame(count_none, index=[0])
    df_count_provide = pd.DataFrame(count_provide, index=[0])

    # Save CSV
    df_f1.to_csv(f"{log_path}/f1_scores.csv", index=False)
    df_doc_len.to_csv(f"{log_path}/doc_lengths.csv", index=False)
    df_count_none.to_csv(f"{log_path}/count_none.csv", index=False)
    df_count_provide.to_csv(f"{log_path}/count_provide.csv", index=False)

    # Calculation of Average of F1
    f1_values = [f1_scores["R&B"], f1_scores["R&L"], f1_scores["Ext"], f1_scores["Fil"], f1_scores["E&F"]]
    f1_average = sum(f1_values) / len(f1_values)

    # Save results into one CSV
    data = {
        "": [dataset_name, "Performance(F1)", "Count_None", "Count_provide"],
        "RAG-Base": ["", f1_scores["R&B"], count_none["R&B"], count_provide["R&B"]],
        "RAG-Long": ["", f1_scores["R&L"], count_none["R&L"], count_provide["R&L"]],
        "Extractor": ["", f1_scores["Ext"], count_none["Ext"], count_provide["Ext"]],
        "Filter": ["", f1_scores["Fil"], count_none["Fil"], count_provide["Fil"]],
        "E&F": ["", f1_scores["E&F"], count_none["E&F"], count_provide["E&F"]],
        "Average": ["", f1_average, "", ""]
    }
    df_all = pd.DataFrame(data)
    df_all.to_csv(f"{log_path}/eval_result.csv", index=False)


## Extract Function
def get_word_len(input):
    tokenized_prompt = \
    set_prompt_tokenizer(input, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return len(tokenized_prompt)


def set_prompt(input, maxlen):
    tokenized_prompt = \
    set_prompt_tokenizer(input, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if len(tokenized_prompt) > maxlen:
        half = int(maxlen * 0.5)
        input = set_prompt_tokenizer.decode(tokenized_prompt[:half],
                                            skip_special_tokens=True) + set_prompt_tokenizer.decode(
            tokenized_prompt[-half:], skip_special_tokens=True)
    return input, len(tokenized_prompt)


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model2path, model_name):
    if "gpt" in model_name or "glm-4" in model_name or "glm3-turbo-128k" in model_name or "gemini" in model_name:
        return model_name, model_name
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name or "longalign-6b" in model_name or "qwen" in model_name or "llama3" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model2path[model_name], trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model2path[model_name], trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16, device_map='auto')
    if "vicuna" in model_name:
        from fastchat.model import load_model
        model, _ = load_model(model2path[model_name], device="cpu", num_gpus=0, load_8bit=False, cpu_offloading=False,
                              debug=False)
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(model2path[model_name], trust_remote_code=True, use_fast=False)
    elif "llama2" in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(model2path[model_name])
        model = LlamaForCausalLM.from_pretrained(model2path[model_name], torch_dtype=torch.bfloat16, device_map='auto')
    model = model.eval()
    return model, tokenizer


@backoff.on_exception(backoff.expo, (Exception), max_time=200)
def pred(model_name, model, tokenizer, prompt, maxlen, max_new_tokens=32, temperature=1):
    '''
    use opensource LLM or call API
    '''
    try:
        if "longalign" in model_name.lower() and max_new_tokens == 32:
            max_new_tokens = 128
        prompt, prompt_len = set_prompt(prompt, maxlen)
        history = []
        if "internlm" in model_name or "chatglm" in model_name or "longalign-6b" in model_name:
            response, history = model.chat(tokenizer, prompt, history=history, max_new_tokens=max_new_tokens,
                                           temperature=temperature, num_beams=1, do_sample=False)
            return response, prompt_len
        elif "baichuan" in model_name:
            messages = [{"content": prompt, "role": "user"}]
            model.generation_config = GenerationConfig.from_pretrained(model2path["baichuan2-7b-4k"],
                                                                       temperature=temperature,
                                                                       max_new_tokens=max_new_tokens, num_beams=1,
                                                                       do_sample=False)
            response = model.chat(tokenizer, messages)
            return response, prompt_len
        elif "llama3" in model_name:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
                model.device)
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            outputs = model.generate(input_ids, eos_token_id=terminators, max_new_tokens=max_new_tokens,
                                     temperature=temperature, num_beams=1, do_sample=False)
            response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            return response, prompt_len
        elif "glm-4" in model_name or "glm3-turbo-128k" in model_name or "gpt" in model_name or "gemini" in model_name:
            response = call_api(prompt, model_name, max_new_tokens, temperature=temperature)
            return response, prompt_len
        elif "qwen" in model_name:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=max_new_tokens, num_beams=1,
                                           do_sample=False, temperature=temperature)
            response = tokenizer.batch_decode(
                [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)],
                skip_special_tokens=True)[0]
            return response, prompt_len
        elif "llama" in model_name:
            input = tokenizer(f"[INST]{prompt}[/INST]", truncation=False, return_tensors="pt").to(model.device)
        elif "vicuna" in model_name:
            from fastchat.model import get_conversation_template
            conv = get_conversation_template("vicuna")
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            input = tokenizer(conv.get_prompt(), truncation=False, return_tensors="pt").to(model.device)
        context_length = input.input_ids.shape[-1]
        output = model.generate(**input, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False,
                                temperature=temperature)
        response = tokenizer.decode(output[0][context_length:], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(1)
        return None, None
    return response, prompt_len


def setup_logger(logger, filename='log'):
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter(fmt="[%(asctime)s][%(levelname)s] - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(os.path.join(log_path, filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)


def print_args(args):
    logger.info(f"{'*' * 30} CONFIGURATION {'*' * 30}")
    for key, val in sorted(vars(args).items()):
        keystr = f"{key}{' ' * (30 - len(key))}"
        logger.info(f"{keystr} --> {val}")
    logger.info(f"LongRAG model used: {args.lrag_model}")
    logger.info(f"{'*' * 30} CONFIGURATION {'*' * 30}")


def search_q(args, question):
    '''
    input: question

    1. vector_search
    2. sort_section
    3. generation

    output: model predict result
    '''

    # 1. Retriever part
    ## raw_pred: predict without retriever Information (only use LLM background knowledge)
    doc_len = {}
    raw_pred = ""
    if args.raw_pred:
        raw_pred, doc_len = search_cache_and_predict(raw_pred, f'{log_path}/raw_pred.json', 'raw_pred', question,
                                                     model_name, model, tokenizer, lambda: create_prompt(question),
                                                     doc_len, maxlen)

    ## retriever process
    ### vector_search: retrieve top 100
    ### sort_section: reranking top7
    retriever, match_id = vector_search(question)
    rerank, match_id = sort_section(args, question, retriever, match_id)

    # 2. Generation Part
    filter_output = []
    extractor_output = []
    fil_pred = ext_pred = ext_fil_pred = rb_pred = rl_pred = ext_rb_pred = rb_ext_fil_pred = ''

    ## 2.1. Filter

    if args.fil:
        fil_pred = load_cache(f'{log_path}/fil_pred.json', 'fil_pred', question, doc_len, 'Fil')
        if (not fil_pred):
            filter_output = filter(question, rerank, args.temperature)
            fil_pred, doc_len = search_cache_and_predict(fil_pred, f'{log_path}/fil_pred.json', 'fil_pred', question,
                                                         model_name, model, tokenizer,
                                                         lambda: create_prompt(''.join(filter_output), question),
                                                         maxlen, args.temperature, doc_len, 'Fil')
        if (fil_pred == "None-api") and args.rerun:  # This part is just for re-filling empty prediction from LLMs
            filter_output = filter(question, rerank, args.temperature)
            fil_pred, doc_len = search_cache_and_predict(fil_pred, f'{log_path_none_rerun}/fil_pred.json', 'fil_pred',
                                                         question, model_name, model, tokenizer,
                                                         lambda: create_prompt(''.join(filter_output), question),
                                                         maxlen, args.temperature, doc_len, 'Fil')

    ## 2.2. Extractor
    if args.ext:
        ext_pred = load_cache(f'{log_path}/ext_pred.json', 'ext_pred', question, doc_len, 'Ext')
        if (not ext_pred):
            extractor_output = extractor(question, rerank, match_id, args.temperature)
            ext_pred, doc_len = search_cache_and_predict(ext_pred, f'{log_path}/ext_pred.json', 'ext_pred', question,
                                                         model_name, model, tokenizer, lambda: create_prompt(
                    ''.join(rerank[:args.top_k2] + extractor_output), question), maxlen, args.temperature, doc_len,
                                                         'Ext')
        if (ext_pred == "None-api") and args.rerun:  # This part is just for re-filling empty prediction from LLMs
            extractor_output = extractor(question, rerank, match_id, args.temperature)
            ext_pred, doc_len = search_cache_and_predict(ext_pred, f'{log_path_none_rerun}/ext_pred.json', 'ext_pred',
                                                         question, model_name, model, tokenizer, lambda: create_prompt(
                    ''.join(rerank[:args.top_k2] + extractor_output), question), maxlen, args.temperature, doc_len,
                                                         'Ext')

    ## 2.3. Extractor & Filter
    if args.ext_fil:
        ext_fil_pred = load_cache(f'{log_path}/ext_fil_pred.json', 'ext_fil_pred', question, doc_len, 'E&F')
        if (not ext_fil_pred):
            if not filter_output:
                filter_output = filter(question, rerank, args.temperature)
            if not extractor_output:
                extractor_output = extractor(question, rerank, match_id, args.temperature)
            ext_fil_pred, doc_len = search_cache_and_predict(ext_fil_pred, f'{log_path}/ext_fil_pred.json',
                                                             'ext_fil_pred', question, model_name, model, tokenizer,
                                                             lambda: create_prompt(
                                                                 ''.join(filter_output + extractor_output), question),
                                                             maxlen, args.temperature, doc_len, 'E&F')
        if (ext_fil_pred == "None-api") and args.rerun:  # This part is just for re-filling empty prediction from LLMs
            if not filter_output:
                filter_output = filter(question, rerank, args.temperature)
            if not extractor_output:
                extractor_output = extractor(question, rerank, match_id, args.temperature)
            ext_fil_pred, doc_len = search_cache_and_predict(ext_fil_pred, f'{log_path_none_rerun}/ext_fil_pred.json',
                                                             'ext_fil_pred', question, model_name, model, tokenizer,
                                                             lambda: create_prompt(
                                                                 ''.join(filter_output + extractor_output), question),
                                                             maxlen, args.temperature, doc_len, 'E&F')

    ## 2.4. RAG-Base
    if args.rb:
        rb_pred = load_cache(f'{log_path}/rb_pred.json', 'rb_pred', question, doc_len, 'R&B')
        if (not rb_pred):
            rb_pred, doc_len = search_cache_and_predict(rb_pred, f'{log_path}/rb_pred.json', 'rb_pred', question,
                                                        model_name, model, tokenizer,
                                                        lambda: create_prompt(''.join(rerank[:args.top_k2]), question),
                                                        maxlen, args.temperature, doc_len, 'R&B')
        if (rb_pred == "None-api") and args.rerun:  # This part is just for re-filling empty prediction from LLMs
            rb_pred, doc_len = search_cache_and_predict(rb_pred, f'{log_path_none_rerun}/rb_pred.json', 'rb_pred',
                                                        question, model_name, model, tokenizer,
                                                        lambda: create_prompt(''.join(rerank[:args.top_k2]), question),
                                                        maxlen, args.temperature, doc_len, 'R&B')

    ## 2.5. RAG-Long
    if args.rl:
        rl_pred = load_cache(f'{log_path}/rl_pred.json', 'rl_pred', question, doc_len, 'R&L')
        if not rl_pred:
            if args.upscaling == 1:
                rl_pred, doc_len = search_cache_and_predict(rl_pred, f'{log_path}/rl_pred.json', 'rl_pred', question,
                                                            model_name, model, tokenizer, lambda: create_prompt(
                        ''.join(s2l_doc_look_up(rerank, match_id, lrag_maxlen, args.top_k2)[0]), question), maxlen,
                                                            args.temperature, doc_len, 'R&L')
            else:
                rl_pred, doc_len = search_cache_and_predict(rl_pred, f'{log_path}/rl_pred.json', 'rl_pred', question,
                                                            model_name, model, tokenizer, lambda: create_prompt(
                        ''.join(s2l_doc(rerank, match_id, lrag_maxlen)[0]), question), maxlen, args.temperature,
                                                            doc_len, 'R&L')
        if (rl_pred == "None-api") and args.rerun:
            if args.upscaling == 1:
                rl_pred, doc_len = search_cache_and_predict(rl_pred, f'{log_path_none_rerun}/rl_pred.json', 'rl_pred',
                                                            question, model_name, model, tokenizer,
                                                            lambda: create_prompt(''.join(
                                                                s2l_doc_look_up(rerank, match_id, lrag_maxlen,
                                                                                args.top_k2)[0]), question), maxlen,
                                                            args.temperature, doc_len, 'R&L')
            else:
                rl_pred, doc_len = search_cache_and_predict(rl_pred, f'{log_path_none_rerun}/rl_pred.json', 'rl_pred',
                                                            question, model_name, model, tokenizer,
                                                            lambda: create_prompt(
                                                                ''.join(s2l_doc(rerank, match_id, lrag_maxlen)[0]),
                                                                question), maxlen, args.temperature, doc_len, 'R&L')

    ## 2.6. Extractor_R&B
    if args.ext_rb:
        ext_rb_pred = load_cache(f'{log_path}/ext_rb_pred.json', 'ext_rb_pred', question, doc_len, 'Ext_RB')
        if (not ext_rb_pred):
            extractor_rb_output = extractor_rb(question, rerank[:args.top_k2], match_id, args.temperature)
            ext_rb_pred, doc_len = search_cache_and_predict(ext_rb_pred, f'{log_path}/ext_rb_pred.json', 'ext_rb_pred',
                                                            question, model_name, model, tokenizer,
                                                            lambda: create_prompt(
                                                                ''.join(rerank[:args.top_k2] + extractor_rb_output),
                                                                question), maxlen, args.temperature, doc_len, 'Ext_RB')
        if (ext_rb_pred == "None-api") and args.rerun:  # This part is just for re-filling empty prediction from LLMs
            extractor_rb_output = extractor_rb(question, rerank, match_id, args.temperature)
            ext_rb_pred, doc_len = search_cache_and_predict(ext_rb_pred, f'{log_path_none_rerun}/ext_rb_pred.json',
                                                            'ext_rb_pred', question, model_name, model, tokenizer,
                                                            lambda: create_prompt(
                                                                ''.join(rerank[:args.top_k2] + extractor_rb_output),
                                                                question), maxlen, args.temperature, doc_len, 'Ext_RB')

    ## 2.7. Extractor_R&B_Ext_Fil
    if args.rb_ext_fil:
        rb_ext_fil_pred = load_cache(f'{log_path}/rb_ext_fil_pred.json', 'rb_ext_fil_pred', question, doc_len,
                                     'RB_Ext_Fil')
        if (not rb_ext_fil_pred):
            if not filter_output:
                filter_output = filter(question, rerank, args.temperature)
            if not extractor_rb_output:
                extractor_rb_output = extractor_rb(question, rerank[:args.top_k2], match_id, args.temperature)
            rb_ext_fil_pred, doc_len = search_cache_and_predict(rb_ext_fil_pred, f'{log_path}/rb_ext_fil_pred.json',
                                                                'rb_ext_fil_pred', question, model_name, model,
                                                                tokenizer, lambda: create_prompt(
                    ''.join(filter_output + extractor_rb_output), question), maxlen, args.temperature, doc_len,
                                                                'RB_Ext_Fil')
        if (
                rb_ext_fil_pred == "None-api") and args.rerun:  # This part is just for re-filling empty prediction from LLMs
            extractor_rb_output = extractor_rb(question, rerank, match_id, args.temperature)
            rb_ext_fil_pred, doc_len = search_cache_and_predict(rb_ext_fil_pred,
                                                                f'{log_path_none_rerun}/rb_ext_fil_pred.json',
                                                                'rb_ext_fil_pred', question, model_name, model,
                                                                tokenizer, lambda: create_prompt(
                    ''.join(filter_output + extractor_rb_output), question), maxlen, args.temperature, doc_len,
                                                                'RB_Ext_Fil')

    return question, retriever, rerank, raw_pred, rb_pred, ext_pred, fil_pred, rl_pred, ext_fil_pred, doc_len, ext_rb_pred, rb_ext_fil_pred


def load_cache(cache_path, pred_key, question, doc_len=None, doc_key=None):
    '''
    check if predict result exist or not
    '''
    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                data = json.loads(line)
                if data['question'] == question:
                    pred_result = data[pred_key]
                    if doc_len is not None and doc_key is not None:
                        doc_len[doc_key] = data["input_len"]
                    return pred_result
    return ''


def search_cache_and_predict(pred_result, cache_path, pred_key, question, model_name, model, tokenizer,
                             create_prompt_func, maxlen, temperature, doc_len=None, doc_key=None):
    '''
    pred and write result
    '''
    doc_len = {}
    if (not pred_result):
        query = create_prompt_func()

        pred_result, input_len = pred(model_name, model, tokenizer, query, maxlen, temperature=temperature)
        with open(cache_path, 'a', encoding='utf-8') as f:
            json.dump({'question': question, pred_key: pred_result, "input_len": input_len}, f, ensure_ascii=False)
            f.write('\n')
        if doc_len is not None and doc_key is not None:
            doc_len[doc_key] = input_len
    if (
            pred_result == "None-api") and args.rerun:  # This part is just for the re-run to fill the empty prediction from LLMs
        query = create_prompt_func()

        pred_result, input_len = pred(model_name, model, tokenizer, query, maxlen, temperature=temperature)

        # Read the existing cache
        with open(cache_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Write the updated cache
        with open(cache_path, 'w', encoding='utf-8') as f:
            replaced = False
            for line in lines:
                data = json.loads(line)
                if data['question'] == question and data[pred_key] == "None-api":
                    print(f"Working on replacing 'None-api' with a fill-in... :{cache_path}")
                    data[pred_key] = pred_result
                    data["input_len"] = input_len
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        if doc_len is not None and doc_key is not None:
            doc_len[doc_key] = input_len
    return pred_result, doc_len


def combine_without_overlap(text1, text2):
    ''' # Deduplication from the two strings
    2가지 string을 overlap 중복은 제거하고 하나로 합치는 함수

    Input: A = '안녕 나의 이름은 제미나이야', B = '이름은 제미나이야 내 소개를 할게'
    Output: '안녕 나의 이름은 제미나이야 내 소개를 할게'

    '''
    text1 = text1.strip()
    text2 = text2.strip()
    # Length of text1 & text2
    len_text1 = len(text1)
    len_text2 = len(text2)

    # Find continuous overlapped parts between text1 & text2
    overlap = ''
    for i in range(len_text1):
        if text1[i:].strip() == text2[:len(text1[i:].strip())]:
            # Maximum continuous overlapped parts
            overlap = text1[i:].strip()
            # print(overlap)
            break

            # 겹치는 부분을 제외한 나머지 부분을 합침
    result = text1 + text2[len(overlap):]

    return result


def s2l_doc(rerank, match_id, maxlen):
    '''
    chunk - paragraph mapping
    '''
    unique_raw_id = []
    contents = []
    s2l_index = {}
    section_index = [id_to_rawid[str(i)] for i in match_id]
    for index, id in enumerate(section_index):
        data = raw_data[id]
        text = data["paragraph_text"]
        if id in unique_raw_id and get_word_len(text) < maxlen:
            continue
        if get_word_len(text) >= maxlen:
            content = rerank[index]
        else:
            unique_raw_id.append(id)
            content = text
        s2l_index[len(contents)] = [i for i, v in enumerate(section_index) if v == section_index[index]]
        contents.append(content)
    return contents, s2l_index


def s2l_doc_look_up(rerank, match_id, maxlen, top_k2):
    '''
    chunk - paragraph mapping
    '''
    unique_raw_id = []
    contents = []
    s2l_index = {}
    section_index = [id_to_rawid[str(i)] for i in match_id]
    count_paragraph = 0

    for index, id in enumerate(section_index):
        if count_paragraph < top_k2:
            data = raw_data[id]
            text = data["paragraph_text"]
            if id in unique_raw_id and get_word_len(text) < maxlen:
                continue
            if get_word_len(text) >= maxlen:
                content = rerank[index]
            else:
                unique_raw_id.append(id)
                content = text
            s2l_index[len(contents)] = [i for i, v in enumerate(section_index) if v == section_index[index]]
            contents.append(content)
            count_paragraph += 1
        else:
            print("Length of Contents reached top_k2", count_paragraph == top_k2)
            print("Length of Contents: ", count_paragraph, "Reached the target k", len(contents))
            break
    return contents, s2l_index


def filter(question, rank_docs, temperature):
    '''
    input: question, chunk

    1. Organize the think process (call LLM)​
    2. determine if the chunk based on the think process is related to question​ (call LLM 7 times  (call llm for each document) )​

    output: question, chunk - filtered
    '''

    content = "\n".join(rank_docs)
    query = f"{content}\n\nPlease combine the above information and give your thinking process for the following question:{question}."
    think_pro, _ = pred(lrag_model_name, lrag_model, lrag_tokenizer, query, lrag_maxlen, 1000, temperature=temperature)
    selected = []

    prompts = [f"""Given an article:{d}\nQuestion: {question}.\n
             Thought process:{think_pro}.\n
             Your task is to use the thought process provided to decide whether you need to cite the article to answer this question.
             If you need to cite the article, set the status value to True. If not, set the status value to False.
             Please output the response in the following json format: {{"status": "{{the value of status}}"}}""" for d
               in rank_docs]
    pool = ThreadPool(processes=args.MaxClients)
    all_responses = pool.starmap(pred,
                                 [(lrag_model_name, lrag_model, lrag_tokenizer, prompt, lrag_maxlen, 32) for prompt in
                                  prompts])

    count_selected = 0
    if args.upscaling == 1:
        for i, r in enumerate(all_responses):
            if count_selected < args.top_k2:
                try:
                    result = json.loads(r[0])
                    res = result["status"]
                    if len(all_responses) != len(rank_docs):
                        break
                    if res.lower() == "true":
                        selected.append(rank_docs[i])
                        count_selected += 1
                except:
                    match = re.search("True|true", r[0])
                    if match:
                        selected.append(rank_docs[i])
                        count_selected += 1
            else:
                print("Length of Selected reached top_k2", count_selected == args.top_k2)
                print("Length of Selected Contents: ", count_selected, "Reached the target k", len(selected))
                break
    else:
        for i, r in enumerate(all_responses):
            try:
                result = json.loads(r[0])
                res = result["status"]
                if len(all_responses) != len(rank_docs):
                    break
                if res.lower() == "true":
                    selected.append(rank_docs[i])
            except:
                match = re.search("True|true", r[0])
                if match:
                    selected.append(rank_docs[i])

    if len(selected) == 0:
        selected = rank_docs
    return selected


def r2long_unique(rerank, match_id):
    unique_raw_id = list(set(id_to_rawid[str(i)] for i in match_id))
    section_index = [id_to_rawid[str(i)] for i in match_id]
    contents = [''.join(rerank[i] for i in range(len(section_index)) if section_index[i] == uid) for uid in
                unique_raw_id]
    return contents, unique_raw_id


def extractor_rb(question, rerank, match_id, temperature):
    '''
    input: question, chunk, match_id

    1. Mapping parent chunk(=paragraph) by using s2l_doc function and if parent chunks overlap, use only one​
    2. Generate information (Global Information) to answer the question (call LLM)​

    output: summary
    '''
    content = ''.join(rerank)
    query = f"{content}.\n\nBased on the above background, please output the information you need to cite to answer the question below.\n{question}"
    response = pred(lrag_model_name, lrag_model, lrag_tokenizer, query, lrag_maxlen, 1000, temperature=temperature)[0]
    # logger.info(f"cite_passage responses: {all_responses}")

    return [response]


def extractor(question, docs, match_id, temperature):
    '''
    input: question, chunk, match_id

    1. Mapping parent chunk(=paragraph) by using s2l_doc function and if parent chunks overlap, use only one​
    2. Generate information (Global Information) to answer the question (call LLM)​

    output: summary
    '''
    if args.upscaling == 1:
        long_docs = s2l_doc_look_up(docs, match_id, lrag_maxlen, args.top_k2)[0]
    else:
        long_docs = s2l_doc(docs, match_id, lrag_maxlen)[0]

    ### Original Script
    # long_docs = s2l_doc(docs, match_id, lrag_maxlen)[0]
    ### Original Script

    content = ''.join(long_docs)
    query = f"{content}.\n\nBased on the above background, please output the information you need to cite to answer the question below.\n{question}"
    response = pred(lrag_model_name, lrag_model, lrag_tokenizer, query, lrag_maxlen, 1000, temperature=temperature)[0]
    # logger.info(f"cite_passage responses: {all_responses}")
    return [response]


def vector_search(question):
    '''
    retrieve top_k1 (default is 100) chunk and id
    input: question
    output: content, match_id
    '''
    feature = emb_model.encode([question])
    distance, match_id = vector.search(feature, args.top_k1)
    content = [chunk_data[int(i)] for i in match_id[0]]
    return content, list(match_id[0])


def extract_document_id(text):
    # Regular expression to find the document_id in the metadata
    pattern = r"'document_id': (\d+)"
    match = re.search(pattern, text)

    if match:
        return int(match.group(1))
    else:
        return None


def is_valid_subset(match_id, extended_context_output_match_id):
    # Check if every element in extended_context_output_match_id is in match_id
    return set(extended_context_output_match_id).issubset(set(match_id))


def sort_lists_by_match_id(match_id, extended_context_output_result, extended_context_output_match_id):
    if not is_valid_subset(match_id, extended_context_output_match_id):
        raise ValueError("extended_context_output_match_id is not a valid subset of match_id.")

    # Create a mapping from match_id to its index
    match_id_index = {id_val: index for index, id_val in enumerate(match_id)}

    # Create a list of tuples (match_id, result) from the extended lists
    extended_pairs = list(zip(extended_context_output_match_id, extended_context_output_result))

    # Sort the extended pairs based on the order of match_id using the mapping
    sorted_extended_pairs = sorted(extended_pairs, key=lambda x: match_id_index[x[0]])

    # Unzip the sorted pairs back into two lists
    sorted_extended_context_output_match_id, sorted_extended_context_output_result = zip(*sorted_extended_pairs)

    return list(sorted_extended_context_output_result), list(sorted_extended_context_output_match_id)


def sort_section(args, question, section, match_id):
    '''
    filter top_k2 chunk is related with question or not
    '''
    q = [question] * len(section)

    ####################################################################################
    ####################################################################################
    if args.look_up_initial_chunk:
        # mapping and delete duplicates
        section_fixed = []
        chunk_id_list = []
        for sect in section:
            chunk_id = eval(sect)['chunk_id']
            try:
                with open("../data/raw_data/{}_{}/{}/raw_txt/chunk_{}.txt".format(args.chunk_size,
                                                                                  args.overlap_chunk_size, args.dataset,
                                                                                  chunk_id), "r",
                          encoding="utf-8") as f:
                    raw_chunk = f.read()
            except:
                with open("../data/raw_data/{}_{}/{}/raw_txt/chunk_{}.txt".format(args.chunk_size,
                                                                                  args.overlap_chunk_size, args.dataset,
                                                                                  chunk_id), "r") as f:
                    raw_chunk = f.read()
            section_fixed += [raw_chunk]
            chunk_id_list.append(chunk_id)
        temp = pd.DataFrame()
        temp['q'] = q
        temp['section'] = section_fixed
        temp['match_id'] = match_id
        temp["chunk_id"] = chunk_id_list
        temp = temp.drop_duplicates(subset='section', keep='first')
        section = temp['section'].tolist()
        match_id = temp['match_id'].tolist()
        chunk_ids = temp['chunk_id'].tolist()
        q = temp['q'].tolist()
    ####################################################################################
    ####################################################################################
    features = cross_tokenizer(q, section, padding=True, truncation=True, return_tensors="pt").to(device)
    cross_model.eval()
    with torch.no_grad():
        scores = cross_model(**features).logits.squeeze(dim=1)
    sort_scores = torch.argsort(scores, dim=0, descending=True).cpu()

    with open(f'../data/corpus/{args.r_path}/{args.dataset}/id_to_rawid.json', encoding='utf-8') as f:
        id_to_rawid = json.load(f)

    # Create doc_ids list regardless of upscaling setting
    doc_ids = []
    for m_id in match_id:
        doc_ids.append(id_to_rawid[str(m_id)])

    if args.with_reranking == 1:
        ### With Rerank
        ### Check whether this experiment is for our method or other methods
        if args.upscaling == 1:
            upsample_top_k2 = min(len(sort_scores), args.upsample_factor * args.top_k2)
            result = [section[sort_scores[i].item()] for i in range(upsample_top_k2)]
            match_id = [match_id[sort_scores[i].item()] for i in range(upsample_top_k2)]
            if args.upscaling == 1:
                chunk_ids = [chunk_ids[sort_scores[i].item()] for i in range(upsample_top_k2)]
            # Update doc_ids based on new match_id order
            doc_ids = [doc_ids[sort_scores[i].item()] for i in range(upsample_top_k2)]
        else:
            result = [section[sort_scores[i].item()] for i in range(args.top_k2)]
            match_id = [match_id[sort_scores[i].item()] for i in range(args.top_k2)]
            if args.upscaling:
                chunk_ids = [chunk_ids[sort_scores[i].item()] for i in range(args.top_k2)]
            # Update doc_ids based on new match_id order
            doc_ids = [doc_ids[sort_scores[i].item()] for i in range(args.top_k2)]
    elif args.with_reranking == 0:
        ### Without Rerank
        ### Check whether this experiment is for our method or other methods
        if args.upscaling == 1:
            upsample_top_k2 = min(len(sort_scores), args.upsample_factor * args.top_k2)
            result = section[:upsample_top_k2]
            match_id = match_id[:upsample_top_k2]
            if args.upscaling == 1:
                chunk_ids = chunk_ids[:upsample_top_k2]
            # Update doc_ids
            doc_ids = doc_ids[:upsample_top_k2]
        else:
            result = section[:args.top_k2]
            match_id = match_id[:args.top_k2]
            if args.upscaling == 1:
                chunk_ids = chunk_ids[:args.top_k2]
            # Update doc_ids
            doc_ids = doc_ids[:args.top_k2]

    if args.look_up_initial_chunk and args.chunk_ext >= 0:  # >= implied alway merging whatever chunk_ext values are, if > then, h=0 did not make merging too.
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print('Triggered: args.look_up_initial_chunk & args.chunk_ext > 0:')
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        add_chunk_id = set()
        # add_match_id = []
        # add_result = []
        doc_id_with_chunk_and_chunk_id = defaultdict(list)
        # Iterate over the top-k results and their corresponding match_ids
        for idx, (chunk, chunk_id, m_id, d_id) in enumerate(zip(result, chunk_ids, match_id, doc_ids)):
            # tmp_doc_id = id_to_rawid[str(m_id)]
            # print('$$$$$ tmp_doc_id == d_id', tmp_doc_id == d_id)
            # Determine the range of chunk_ids to check based on args.chunk_ext
            chunk_id_range = range(chunk_id - args.chunk_ext, chunk_id + args.chunk_ext + 1)
            # Check each chunk_id in the determined range
            for cid in chunk_id_range:
                if cid not in add_chunk_id:
                    add_chunk_id.add(cid)
                    # If the file exists, read the raw chunk data
                    if file_exists(cid, args):
                        try:
                            with open("../data/raw_data/{}_{}/{}/raw_txt/chunk_{}.txt".format(args.chunk_size,
                                                                                              args.overlap_chunk_size,
                                                                                              args.dataset, cid), "r",
                                      encoding="utf-8") as f:
                                raw_chunk = f.read()
                        except:
                            with open("../data/raw_data/{}_{}/{}/raw_txt/chunk_{}.txt".format(args.chunk_size,
                                                                                              args.overlap_chunk_size,
                                                                                              args.dataset, cid),
                                      "r") as f:
                                raw_chunk = f.read()

                        if extract_document_id(raw_chunk) == d_id:
                            doc_id_with_chunk_and_chunk_id[d_id].append((cid, raw_chunk, m_id))

                    else:
                        pass

        for key in doc_id_with_chunk_and_chunk_id.keys():
            print('Before Sorting: ', doc_id_with_chunk_and_chunk_id[key])
            doc_id_with_chunk_and_chunk_id[key] = sorted(doc_id_with_chunk_and_chunk_id[key], key=lambda x: x[0])
            print('After Sorting: ', doc_id_with_chunk_and_chunk_id[key])

        extended_context_output_result = []
        extended_context_output_match_id = []

        ordered_unique_docs = []
        doc_seen = set()
        for d_id in doc_ids:
            if d_id not in doc_seen:
                doc_seen.add(d_id)
                ordered_unique_docs.append(d_id)
                chunk_num_seen = set()  # Check chunk_num = cid
                raw_txt_all = ""
                for i in range(len(doc_id_with_chunk_and_chunk_id[d_id])):
                    chunk_num, org_chunk, m_id = doc_id_with_chunk_and_chunk_id[d_id][i][0], \
                                                 doc_id_with_chunk_and_chunk_id[d_id][i][1], \
                                                 doc_id_with_chunk_and_chunk_id[d_id][i][2]
                    org_chunk_content_start = org_chunk.find("page_content:") + len("page_content:")
                    org_chunk_content_end = org_chunk.find("\n\nmetadata:") + len("\n\n")
                    curr_doc_id = id_to_rawid[str(m_id)]
                    if chunk_num not in chunk_num_seen:
                        if i == 0:
                            raw_txt_all = combine_without_overlap(raw_txt_all, org_chunk[
                                                                               org_chunk_content_start:org_chunk_content_end].strip())
                            prev_chunk_num = chunk_num
                            prev_doc_id = curr_doc_id
                            prev_meta_data = org_chunk[org_chunk_content_end:].strip()
                        elif chunk_num == prev_chunk_num + 1 and curr_doc_id == prev_doc_id:
                            raw_txt_all = combine_without_overlap(raw_txt_all, org_chunk[
                                                                               org_chunk_content_start:org_chunk_content_end].strip())
                            prev_chunk_num = chunk_num
                            prev_doc_id = curr_doc_id
                            prev_meta_data = org_chunk[org_chunk_content_end:].strip()
                        else:
                            if args.merge_version == 1:
                                raw_txt_all += " " + prev_meta_data + "\n"
                                # extended_context_output_result.append(raw_txt_all)
                                # extended_context_output_match_id.append(m_id)
                                prev_meta_data = ""

                                # print("@"*50)
                                # print(m_id)
                                # print(raw_txt_all)
                                # print("@"*50)
                                # raw_txt_all = ""
                                raw_txt_all += org_chunk[org_chunk_content_start:org_chunk_content_end].strip()
                                prev_chunk_num = chunk_num
                                prev_doc_id = curr_doc_id
                                prev_meta_data = org_chunk[org_chunk_content_end:].strip()
                            elif args.merge_version == 2:
                                raw_txt_all += " " + prev_meta_data + "\n"
                                extended_context_output_result.append(raw_txt_all)
                                extended_context_output_match_id.append(m_id)
                                prev_meta_data = ""

                                # print("@"*50)
                                # print(m_id)
                                # print(raw_txt_all)
                                # print("@"*50)
                                raw_txt_all = ""
                                raw_txt_all += org_chunk[org_chunk_content_start:org_chunk_content_end].strip()
                                prev_chunk_num = chunk_num
                                prev_doc_id = curr_doc_id
                                prev_meta_data = org_chunk[org_chunk_content_end:].strip()
                            elif args.merge_version == 3:
                                raw_txt_all += " " + prev_meta_data
                                extended_context_output_result.append(raw_txt_all)
                                extended_context_output_match_id.append(m_id)
                                prev_meta_data = ""

                                # print("@"*50)
                                # print(m_id)
                                # print(raw_txt_all)
                                # print("@"*50)
                                raw_txt_all = ""
                                raw_txt_all += org_chunk[org_chunk_content_start:org_chunk_content_end].strip()
                                prev_chunk_num = chunk_num
                                prev_doc_id = curr_doc_id
                                prev_meta_data = org_chunk[org_chunk_content_end:].strip()
                        if curr_doc_id != prev_doc_id:
                            print('$$$$$$$$$$$$$$$$$$ ERROR $$$$$$$$$$$$$$$$$$')
                    chunk_num_seen.add(chunk_num)
                raw_txt_all += " " + prev_meta_data
                extended_context_output_result.append(raw_txt_all)
                extended_context_output_match_id.append(m_id)
        del features
        del scores
        torch.cuda.empty_cache()

        # print('extended_context_output_result: ', extended_context_output_result[1])
        print('Original ordered_unique_docs id: ', ordered_unique_docs)
        print('Original docs id: ', doc_ids)
        print('Original match id: ', match_id)
        print('extended_context_output_match_id: ', extended_context_output_match_id)
        print('extended_context_output_result: ', len(extended_context_output_result))

        # import pdb; pdb.set_trace()
        return extended_context_output_result, extended_context_output_match_id

    del features
    del scores
    torch.cuda.empty_cache()

    return result, match_id


def create_prompt(input, question):
    if args.prompt_version == 0:
        user_prompt = f"Answer the question based on only your knowledge. Only give me the answer and do not output any other words.\n\nAnswer the question based on only your knowledge. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:"
    elif args.prompt_version == 1:
        user_prompt = f"Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{input}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:"
    elif args.prompt_version == 2:
        user_prompt = f"Answer the question based on the only given passages, and please only use facts contained in the given passages in providing the answer and if you think ambiguous then please don't answer. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{input}\n\nAnswer the question based on the only given passages, and please double-check that the answer is generated only from the given passages in providing the answer and if you think ambiguous then please don't answer. Only give me the answer and do not output any other words.\n\nQuestion: {question}\nAnswer:"
    elif args.prompt_version == 3:
        user_prompt = f"Answer the question based on the given passages. Only give me the answer in as fewer number of words as possible.\n\nThe following are given passages. <Start of Passage>: \n{input}\n<End of Passage>\nAnswer the question based on the given passages.  Only give me the answer in as fewer number of words as possible.\n\nQuestion: {question}\nAnswer:"
    return user_prompt


if __name__ == '__main__':
    seed_everything(42)
    index_path = f'../data/corpus/{args.r_path}/{args.dataset}/vector.index'  # Vector index path

    ## 1. Load Vector, Data, Model...

    #### 1.1. Load Vector
    print("Vector Load ... {}".format(index_path))
    vector = faiss.read_index(index_path)
    print("Vector Load Done!!!")
    #### 1.2. Load Data
    ###### Paragraph
    with open(f'../data/corpus/raw/{args.dataset}.json', encoding='utf-8') as f:
        raw_data = json.load(f)
    ###### chunk <-> Paragraph
    ###### - summary_chunk_slicing
    ###### - for retriever
    with open(f'../data/corpus/{args.r_path}/{args.dataset}/id_to_rawid.json', encoding='utf-8') as f:
        id_to_rawid = json.load(f)
    ###### chunk
    ###### - chunk contents
    ###### - for retriever
    with open(f"../data/corpus/{args.r_path}/{args.dataset}/chunks.json", "r", encoding='utf-8') as fin:
        chunk_data = json.load(fin)

    now = datetime.now()
    now_time = now.strftime("%Y-%m-%d-%H:%M:%S")
    # log_path = args.log_path or f'./log/{args.r_path.split("/")[-1]}/{args.dataset}/{args.model}/{args.lrag_model or "base"}/{now_time}'
    # log_path = args.log_path or f'./log/{args.r_path.split("/")[-1]}/{args.dataset}/{args.model}/{args.lrag_model}/{args.version}'

    if args.chunk_ext > 0:
        if args.with_reranking == 0:
            if args.upscaling == 1:
                log_path = args.log_path or f'./log/{args.r_path.split("/")[-1]}/{args.dataset}/{args.model}/prompt_v{args.prompt_version}_{args.version}_upscaling_chunk_ext{args.chunk_ext}_without_reranking_top_k1_{args.top_k1}_top_k2_{args.top_k2}_merge_v{args.merge_version}'
                print('log_path: ', log_path)
            else:
                log_path = args.log_path or f'./log/{args.r_path.split("/")[-1]}/{args.dataset}/{args.model}/prompt_v{args.prompt_version}_{args.version}_no_upscaling_chunk_ext{args.chunk_ext}_without_reranking_top_k1_{args.top_k1}_top_k2_{args.top_k2}_merge_v{args.merge_version}'
                print('log_path: ', log_path)
        else:
            if args.upscaling == 1:
                log_path = args.log_path or f'./log/{args.r_path.split("/")[-1]}/{args.dataset}/{args.model}/prompt_v{args.prompt_version}_{args.version}_upscaling_chunk_ext{args.chunk_ext}_with_reranking_top_k1_{args.top_k1}_top_k2_{args.top_k2}_merge_v{args.merge_version}_{args.reranking_model}_upsampling_{args.upsample_factor}'
                print('log_path: ', log_path)
            else:
                log_path = args.log_path or f'./log/{args.r_path.split("/")[-1]}/{args.dataset}/{args.model}/prompt_v{args.prompt_version}_{args.version}_no_upscaling_chunk_ext{args.chunk_ext}_with_reranking_top_k1_{args.top_k1}_top_k2_{args.top_k2}_merge_v{args.merge_version}_{args.reranking_model}'
                print('log_path: ', log_path)
    else:
        if args.with_reranking == 0:
            if args.upscaling == 1:
                log_path = args.log_path or f'./log/{args.r_path.split("/")[-1]}/{args.dataset}/{args.model}/prompt_v{args.prompt_version}_{args.version}_upscaling_chunk_ext{args.chunk_ext}_without_reranking_top_k1_{args.top_k1}_top_k2_{args.top_k2}_merge_v{args.merge_version}'
                print('log_path: ', log_path)
            else:
                log_path = args.log_path or f'./log/{args.r_path.split("/")[-1]}/{args.dataset}/{args.model}/prompt_v{args.prompt_version}_{args.version}_no_upscaling_chunk_ext{args.chunk_ext}_without_reranking_top_k1_{args.top_k1}_top_k2_{args.top_k2}_merge_v{args.merge_version}'
                print('log_path: ', log_path)
        else:
            if args.upscaling == 1:
                log_path = args.log_path or f'./log/{args.r_path.split("/")[-1]}/{args.dataset}/{args.model}/prompt_v{args.prompt_version}_{args.version}_upscaling_chunk_ext{args.chunk_ext}_with_reranking_top_k1_{args.top_k1}_top_k2_{args.top_k2}_merge_v{args.merge_version}_{args.reranking_model}_upsampling_{args.upsample_factor}'
                print('log_path: ', log_path)
            else:
                log_path = args.log_path or f'./log/{args.r_path.split("/")[-1]}/{args.dataset}/{args.model}/prompt_v{args.prompt_version}_{args.version}_no_upscaling_chunk_ext{args.chunk_ext}_with_reranking_top_k1_{args.top_k1}_top_k2_{args.top_k2}_merge_v{args.merge_version}_{args.reranking_model}'
                print('log_path: ', log_path)

    os.makedirs(log_path, exist_ok=True)

    ###### config load -> model_path, model_name, max_len
    with open("../config/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    #### 1.3. Load Model (Generation, Reranking...)
    model_name = args.model.lower()
    model2path = config["model_path"]
    maxlen = config["model_maxlen"][model_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## emb_model ["BAAI/bge-m3", "intfloat/multilingual-e5-large"]
    emb_model = SentenceTransformer("{}".format(args.emb_model_name)).to(device)
    if args.reranking_model == "marco_MiniLM":
        print("############# marco_MiniLM model reranking #############")
        cross_tokenizer = AutoTokenizer.from_pretrained(model2path["rerank_model"])  # "ms-marco-MiniLM-L-12-v2"
        cross_model = AutoModelForSequenceClassification.from_pretrained(model2path["rerank_model"]).to(
            device)  # "ms-marco-MiniLM-L-12-v2"
    elif args.reranking_model == "bge_m3":
        print("############# bge_m3 model reranking #############")
        cross_tokenizer = AutoTokenizer.from_pretrained(model2path["rerank_model_bge"])  # "bge-reranker-v2-m3"
        cross_model = AutoModelForSequenceClassification.from_pretrained(model2path["rerank_model_bge"]).to(
            device)  # "bge-reranker-v2-m3"
    model, tokenizer = load_model_and_tokenizer(model2path, model_name)

    ###### args.lrag_model is none
    if args.lrag_model:
        lrag_model_name = args.lrag_model.lower()
        lrag_maxlen = config["model_maxlen"][lrag_model_name]
        lrag_model, lrag_tokenizer = (model, tokenizer) if model_name == lrag_model_name else load_model_and_tokenizer(
            model2path, lrag_model_name)
    else:
        lrag_model_name, lrag_model, lrag_tokenizer, lrag_maxlen = (model_name, model, tokenizer, maxlen)
    set_prompt_tokenizer = AutoTokenizer.from_pretrained(model2path["chatglm3-6b-32k"], trust_remote_code=True)
    setup_logger(logger)
    print_args(args)

    #### 1.4. Load eval data
    questions, answer, raw_preds, rank_preds, ext_preds, fil_preds, longdoc_preds, ext_fil_preds, docs_len, ext_rb_preds, rb_ext_fil_preds = [], [], [], [], [], [], [], [], [], [], []
    with open(f'../data/eval/{args.dataset}.json', encoding='utf-8') as f:
        qs_data = json.load(f)

    for d in qs_data:
        questions.append(d["question"])
        answer.append(d["answers"])

    ## 2. Run evaluation by each eval dataset
    # Examine files' existence
    files_to_check = [
        "ext_fil_pred.json",
        "ext_pred.json",
        "fil_pred.json",
        "rb_pred.json",
        "rl_pred.json",
        "ext_rb_pred.json"]
    # Examine file lengths too
    import shutil


    def check_files(log_path, files):
        for file in files:
            file_path = os.path.join(log_path, file)
            if not os.path.exists(file_path):
                return False
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) != 200:
                    return False
        return True
        # If it is true, then duplicate the folder


    if check_files(log_path, files_to_check) and args.rerun:
        log_path_none_rerun = f"{log_path}_none_rerun"
        shutil.copytree(log_path, log_path_none_rerun)
        print(f"Folder copied to {log_path_none_rerun}")
    else:
        print("One or more files do not exist or do not have enough lines.")
    # questions = questions[:5]
    for index, query in tqdm(enumerate(questions), total=len(questions), desc="evaluation start...!!"):
        logger.info(f"Question: {query}")
        question, retriever, rerank, raw_pred, rb_pred, ext_pred, fil_pred, rl_pred, ext_fil_pred, doc_len, ext_rb_pred, rb_ext_fil_pred = search_q(
            args, query)

        raw_preds.append(raw_pred)
        rank_preds.append(rb_pred)
        ext_preds.append(ext_pred)
        fil_preds.append(fil_pred)
        longdoc_preds.append(rl_pred)
        ext_fil_preds.append(ext_fil_pred)
        docs_len.append(doc_len)
        ext_rb_preds.append(ext_rb_pred)
        rb_ext_fil_preds.append(rb_ext_fil_pred)

    ## 3. performance check
    all_len1 = all_len2 = all_len3 = all_len4 = all_len5 = all_len6 = all_len7 = 0
    for dl in docs_len:
        all_len1 += dl.get('Ext', 0)
        all_len2 += dl.get('Fil', 0)
        all_len3 += dl.get('R&B', 0)
        all_len4 += dl.get('R&L', 0)
        all_len5 += dl.get('E&F', 0)
        all_len6 += dl.get('Ext_RB', 0)
        all_len7 += dl.get('RB_Ext_Fil', 0)

    doc_len_eval = {
        "Ext": all_len1 / len(docs_len),
        "Fil": all_len2 / len(docs_len),
        "R&B": all_len3 / len(docs_len),
        "R&L": all_len4 / len(docs_len),
        "E&F": all_len5 / len(docs_len),
        "Ext_RB": all_len6 / len(docs_len),
        "RB_Ext_Fil": all_len7 / len(docs_len)
    }

    F1 = {
        "raw_pre": F1_scorer(raw_preds, answer),
        "R&B": F1_scorer(rank_preds, answer),
        "Ext": F1_scorer(ext_preds, answer),
        "Fil": F1_scorer(fil_preds, answer),
        "R&L": F1_scorer(longdoc_preds, answer),
        "E&F": F1_scorer(ext_fil_preds, answer),
        "Ext_RB": F1_scorer(ext_rb_preds, answer),
        "RB_Ext_Fil": F1_scorer(rb_ext_fil_preds, answer)
    }
    keywords = ["none-api", "provide"]
    count_none = {
        "Ext": count_keywords(ext_preds, keywords)["none-api"],
        "Fil": count_keywords(fil_preds, keywords)["none-api"],
        "R&B": count_keywords(rank_preds, keywords)["none-api"],
        "R&L": count_keywords(longdoc_preds, keywords)["none-api"],
        "E&F": count_keywords(ext_fil_preds, keywords)["none-api"],
        "Ext_RB": count_keywords(ext_rb_preds, keywords)["none-api"],
        "RB_Ext_Fil": count_keywords(rb_ext_fil_preds, keywords)["none-api"]
    }
    count_provide = {
        "Ext": count_keywords(ext_preds, keywords)["provide"],
        "Fil": count_keywords(fil_preds, keywords)["provide"],
        "R&B": count_keywords(rank_preds, keywords)["provide"],
        "R&L": count_keywords(longdoc_preds, keywords)["provide"],
        "E&F": count_keywords(ext_fil_preds, keywords)["provide"],
        "Ext_RB": count_keywords(ext_rb_preds, keywords)["provide"],
        "RB_Ext_Fil": count_keywords(rb_ext_fil_preds, keywords)["provide"]
    }

    eval_result = {"F1": F1, "doc_len": doc_len_eval, "Count_None": count_none, "Count_Provide": count_provide}
    if args.rerun:
        with open(f"{log_path_none_rerun}/eval_result_rerun.json", "w") as fout:
            json.dump(eval_result, fout, ensure_ascii=False, indent=4)
        # Save results into CSV file
        save_results_to_csv(eval_result, log_path_none_rerun, args.dataset)

    else:
        with open(f"{log_path}/eval_result.json", "w") as fout:
            json.dump(eval_result, fout, ensure_ascii=False, indent=4)
        # Save results into CSV file
        save_results_to_csv(eval_result, log_path, args.dataset)