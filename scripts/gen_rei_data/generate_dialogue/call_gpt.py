import requests
from openai import OpenAI


def call_chatgpt_api(prompt_file_path, task=None, reference=None, dialogue=None, confuse_name=None, api_file_path='generate_dataset/api', 
                     model="gpt-4o-mini", max_tokens=500, temperature=0.7):
    with open(api_file_path, 'r', encoding='utf-8') as file:
        api_key = file.read().strip()
    
    with open(prompt_file_path, 'r', encoding='utf-8') as file:
        prompt_template = file.read()
    
    prompt = prompt_template

    if task is not None:
        prompt = prompt.replace("{THE TASK}", task)
    if reference is not None:
        prompt = prompt.replace("{THE REFERENCE}", reference)
    if dialogue is not None:
        prompt = prompt.replace("{THE DIALOGUE}", dialogue)
    if confuse_name is not None:
        prompt = prompt.replace("{CONFUSE NAME}", confuse_name)
    
    # print('-------------------prompt-------------------')
    # print(prompt)
    # print('-------------------prompt-------------------')

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        # "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        print(f"the model use is {response.json()['model']}")
        print(f"the input token number is {response.json()['usage']['prompt_tokens']}\
              the output token number is {response.json()['usage']['completion_tokens']}")
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code}, {response.text}"

if __name__ == "__main__": 
    api_file_path = 'generate_dataset/api'
    prompt_file_path = 'generate_dataset/prompts/prompt1-1'  # 设置 prompt 文件路径
    task = "Place the vase on the coffee table"  # 需要替换的任务内容
    response = call_chatgpt_api(prompt_file_path, task)
    print(response)
