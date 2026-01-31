import json 
import os
from call_gpt import call_chatgpt_api

split_file_path = 'alfred/data/splits/pre2.json'
base_path = 'alfred/data/json_2.1.0'
prompt_file_path = 'generate_dataset/prompts/prompt2-1' 

with open(split_file_path, 'r') as f:
    split_data = json.load(f)

dic_list = split_data["valid_seen"]
file_num = 0
error_num = 0
max_num = 100

for item in dic_list:
    task_path = item["task"]
    dialogue = item["memory1-1"]
    reference = item["reference"]
    json_file_path = os.path.join(base_path, task_path, 'pp', 'ann_0.json')
    
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            task = json.load(f)['turk_annotations']['anns'][int(item["repeat_idx"])]["task_desc"]
            file_num += 1

            import random
            with open('generate_dataset/confuse name', 'r') as file:
                names = file.read().splitlines()
            confuse_name = random.choice(names)

            response = call_chatgpt_api(prompt_file_path, refernce=reference, confuse_name=confuse_name, dialogue=dialogue)
            print('------------response------------')
            print(response)
            print('------------response------------')

            item["memory2-1"] = response
            item["confuse_name"] = confuse_name

    else:
        print(f"{json_file_path} doesn't exist！")
        error_num += 1
    
    if file_num >= max_num:
        break

with open(split_file_path, 'w') as f:
    json.dump(split_data, f, indent=4)

print(f"{file_num} tasks have been solved!")
if error_num:
    print(f"error: {error_num} tasks have not been read!")
