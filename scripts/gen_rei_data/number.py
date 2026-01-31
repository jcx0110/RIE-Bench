# read the log file
# generate the .json file
# output the repeat_idx, task_dict, task_name in a text file

# import
import numpy as np
import json
import os

# set the log file
log_file_path = 'outputs/alfred/LLaMa3.1-8B_train4/evaluate.log'
with open(log_file_path, 'r') as file:
    lines = file.readlines()

successful_tasks_number = {}
successful_tasks_dic = []
successful_tasks = []
success_repeat_idx = []
current_task_number = None
current_task_dic = None
current_task = None
current_repeat_idx = None

for line in lines:
    if "repeat_idx" in line:
        current_repeat_idx = line.split(': ')[1].split(',')[0]

    if "Evaluating (" in line:
        current_task_number = line.split('Evaluating ')[1].split(')')[0].strip()
        current_task_dic = line.split('json_2.1.0/')[1]
        current_task_dic = current_task_dic.replace('\n', '')
        # print(current_task_dic)

    if "Task" in line:
        print(line)
        current_task = line.split('Task: ')[1]
        current_task = current_task.replace('\n', '')
        # print(current_task)

    if "success:" in line and current_task_number is not None:
        success_status = line.split("success: ")[1].strip()
        if success_status == "True":
            successful_tasks_number[current_task_number] = True 
            successful_tasks_dic.append(current_task_dic)
            successful_tasks.append(current_task)
            success_repeat_idx.append(current_repeat_idx)
        current_task_number = None
        current_task_dic = None
        current_task = None
        current_repeat_idx = None

# generate the .json file
output_data = {
    "valid_seen": []
}

for idx, task in zip(success_repeat_idx, successful_tasks_dic):
    output_data["valid_seen"].append({
        "repeat_idx": int(idx),
        "task": task
    })

print(output_data )

file_name = os.path.basename(os.path.dirname(log_file_path))
output_directory = 'alfred/data/splits'
output_file_path = os.path.join(output_directory, f"{file_name}.json")
os.makedirs(output_directory, exist_ok=True)
with open(output_file_path, "w") as outfile:
    json.dump(output_data, outfile, indent=4, ensure_ascii=False)

print(f"data has been written into {output_file_path} successfully")

# generate the .txt file
output_directory = 'alfred/data/splits'
output_txt_path = os.path.join(output_directory,  f"{file_name}.txt")

os.makedirs(output_directory, exist_ok=True)
with open(output_txt_path, "w") as outfile:
    for i in range(len(successful_tasks_dic)):
        task_dict = successful_tasks_dic[i]
        task_name = successful_tasks[i]
        repeat_idx = success_repeat_idx[i]

        outfile.write(f"{repeat_idx}, {task_dict}, {task_name}\n")

print(f"data has been written into {output_txt_path} successfully")