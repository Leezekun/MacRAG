      
import json
import requests
import re
# from zhipuai import ZhipuAI
from openai import OpenAI
import backoff
import time
from openai import OpenAI
import httpx
import yaml
with open("../config/config.yaml", "r") as file:
    config = yaml.safe_load(file)["api"]

def glm(prompt,model,max_tokens):
    try:
        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key=config["zp_key"]) 
        prompt=[{"role": "user", "content":prompt}]
        response = client.chat.completions.create(
        model=model,
        temperature = 0.99,
        messages=prompt,
        max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
            print(f"An error occurred: {e}")  
            time.sleep(0.5)
            return None

def gemini(prompt, model, max_tokens, temperature):
    try:
        gemini_prompt_setting = {
            "model": f"{model}",
            "apikey": config["gemini_key"],
            "input": { },
            "generation_config": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
                # "topP": 1,
                # "topK": 32
                                }
                     }
        gemini_input_messages = {
                    "role": "user",
                    "parts": {
                                "text": f"{prompt}"
                            }
                    }
        gemini_safety_settings = [
                                {
                                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                "threshold": "BLOCK_NONE"
                                },
                                {
                                "category": "HARM_CATEGORY_HATE_SPEECH",
                                "threshold": "BLOCK_NONE"
                                },
                                {
                                "category": "HARM_CATEGORY_HARASSMENT",
                                "threshold": "BLOCK_NONE"
                                },
                                {
                                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                                "threshold": "BLOCK_NONE"
                                }
                            ]
        gemini_prompt_setting['input']['contents'] = gemini_input_messages
        gemini_prompt_setting['input']['safety_settings']=gemini_safety_settings

        payload = json.dumps(gemini_prompt_setting)
        url = config["gemini_url"]

        headers = {'Content-Type': 'application/json','User-Agent': 'Mozilla/5.0'}
        response = requests.request("POST", url, headers=headers, data=payload)
            
        output = json.loads(response.text)['candidates'][0]['content']['parts'][0]['text']
        output = output.replace("```json\n","")
        response = output.replace("\n```","")
        return response
    
    except Exception as e:
        print(f"An error occurred: {e}")
        time.sleep(5)
        try:
            response = requests.request("POST", url, headers=headers, data=payload)

            output = json.loads(response.text)['candidates'][0]['content']['parts'][0]['text']
            output = output.replace("```json\n","")
            response = output.replace("\n```","")
            return response
        except Exception as e:
            print(f"An error occurred: {e}")
    #    time.sleep(5)


    return "None"
    


def gpt(prompt,model,max_tokens,temperature): 
    try:
        client = OpenAI(
            api_key=config["openai_key"],
        )
        message = [{"role": "user", "content": prompt}]
        completion = client.chat.completions.create(model= model, messages= message, max_tokens=max_tokens, temperature=temperature)      
        response = completion.choices[0].message.content
        return response

    except Exception as e:
        print(f"An error occurred: {e}")  
        time.sleep(1)
    return "None"

# Remove duplicate sentences to prevent GPT API from refusing to respond due to repeated input

def remove_consecutive_repeated_sentences(text, threshold=5):
    sentences = re.split(r'([。！？,，])', text)
    
    cleaned_sentences = []
    current_sentence = None
    count = 0

    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip()
        if i + 1 < len(sentences):
            delimiter = sentences[i + 1]
        else:
            delimiter = ''

        if sentence == current_sentence:
            count += 1
        else:
            if count >= threshold:
                cleaned_sentences.append(current_sentence + delimiter)
            elif current_sentence:
                cleaned_sentences.extend([current_sentence + delimiter] * count)
            current_sentence = sentence
            count = 1

    if count >= threshold:
        cleaned_sentences.append(current_sentence + delimiter)
    else:
        cleaned_sentences.extend([current_sentence + delimiter] * count)
    
    cleaned_text = ''.join(cleaned_sentences)
    return cleaned_text


@backoff.on_exception(backoff.expo, (Exception), max_time=500)
def call_api(prompt,model,max_new_tokens, temperature = 1):
    if "glm" in model:
        res=glm(prompt,model, max_new_tokens)
    elif "gpt" in model:
        res=gpt(prompt,model,max_new_tokens, temperature)
        if not res:
            prompt=remove_consecutive_repeated_sentences(prompt)
            res=gpt(prompt,model,max_new_tokens)
    elif "gemini" in model:
        res = gemini(prompt, model, max_new_tokens, temperature)
    assert res != None
    return res

if __name__ == "__main__":

    print(call_api("Hello", "gemini", 100))
    print(call_api("Hello", "gpt-4o", 100))
    print(call_api("Hello","glm-4",100))
    print(call_api("Hello","chatglm_turbo",100))
    



    
