import json 
import os
from call_gpt import call_chatgpt_api
import random

split_file_path = 'alfred/data/splits/pre_experiment_tasks3.json'
base_path = 'alfred/data/json_2.1.0'

generate_goals = ['1-1', '1-2', '1-3', '2-1', '2-2', '2-3', '3-1', '3-2', '3-3']
# generate_goals = ['1-2', '1-3', '2-1', '2-2', '2-3', '3-1']
generate_goals = ['1-3', '2-3', '3-3']
generate_goals = ['3-3']
sources = ['', '1-1', '1-1', '1-1', '2-1', '2-1', '2-1', '3-1', '3-1']
# sources = ['1-1', '1-1', '1-1', '2-1', '2-1', '2-1']
sources = ['3-1']
for i, generate_goal in enumerate(generate_goals):
    source = sources[i]
    print("-------------", generate_goal, source)
    prompt_file_path = f'generate_dataset/prompts/prompt{generate_goal}' 
    if generate_goal in ["2-2", "3-2"]:
        prompt_file_path = f'generate_dataset/prompts/prompt1-2' 
    if generate_goal in ["2-3", "3-3"]:
        prompt_file_path = f'generate_dataset/prompts/prompt1-3' 

    with open(split_file_path, 'r') as f:
        split_data = json.load(f)

    dic_list = split_data["valid_seen"]
    file_num = 0
    error_num = 0
    max_num = 1000

    task = None
    reference = None
    confuse_name = None
    dialogue = None
    for i, item in enumerate(dic_list):
        print(f"----number {i} of {len(dic_list)}----")
        # if f"memory{generate_goal}" in item:
        #     print(f"memory{generate_goal} already exists for task {i}, skipping...")
        #     continue
        print(f"memory{generate_goal} is going to generage for task {i}...")
        task_path = item["task"]
        if generate_goal in ["1-2", "1-3", "2-1", "2-2", "2-3", "3-1", "3-2", "3-3"]:
            dialogue = item[f"memory{source}"]
            reference = item["reference"]
        if generate_goal in ["2-2", "2-3", "3-1", "3-2", "3-3"]:
            confuse_name = item['confuse_name']
        json_file_path = os.path.join(base_path, task_path, 'pp', 'ann_0.json')
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                task = json.load(f)['turk_annotations']['anns'][int(item["repeat_idx"])]["task_desc"]
                file_num += 1
                
                if generate_goal in ["1-1"]:
                    prompt_reference = 'generate_dataset/recoganize_reference_prompt'  
                    reference = call_chatgpt_api(prompt_reference, task)
                    item["reference"] = reference
                    print(reference)

                if generate_goal in ["2-1"]:
                    with open('generate_dataset/confuse name', 'r') as file:
                        names = file.read().splitlines()
                    confuse_name = random.choice(names)
                response = call_chatgpt_api(prompt_file_path, task=task, reference=reference, confuse_name=confuse_name, dialogue=dialogue)
                # print('------------response------------')
                # print(response)
                # print('------------response------------')
                if generate_goal in ["2-1"]:
                    item["confuse_name"] = confuse_name
                if generate_goal in ["1-1"]:
                    item["task_dec"] = task
                item[f"memory{generate_goal}"] = response
        else:
            print(f"{json_file_path} doesn't exist！")
            error_num += 1
        with open(split_file_path, 'w') as f:
            json.dump(split_data, f, indent=4)
        
        if file_num >= max_num:
            break


    print(f"{file_num} tasks have been solved!")
    if error_num:
        print(f"error: {error_num} tasks have not been read!")
