import json
import os
import random

# Generate goals and corresponding reference counts
task_confusing_levels = ['1-1', '1-2', '1-3', '2-1', '2-2', '2-3', '3-1', '3-2', '3-3']

# File paths
split_file_path = 'alfred/data/splits/choosed_task.json'
# split_file_path = 'alfred/data/splits/pre_experiment_tasks.json'
base_path = 'alfred/data/json_2.1.0 copy'

# Load data from the split file
with open(split_file_path, 'r') as f:
    split_data = json.load(f)

# Filter valid tasks and randomly select 100
valid_tasks_look = []
valid_tasks_clean = []
valid_tasks_cool = []
valid_tasks_heat = []
valid_tasks_place = []
valid_tasks_stack = []

task_counter = 0



for item in split_data["valid_seen"]:
    task_path = item["task"]
    item["idx"] = task_counter  
    task_counter += 1 
    task_path = item["task"]
    if 'look_at_obj' in task_path:
        valid_tasks_look.append(item)
    elif 'pick_clean' in task_path:
        valid_tasks_clean.append(item)
    elif 'pick_cool' in task_path:
        valid_tasks_cool.append(item)
    elif 'pick_heat' in task_path:
        valid_tasks_heat.append(item)
    elif 'pick_and_place_simple' in task_path:
        valid_tasks_place.append(item)
    elif 'pick_and_place_with_movable' in task_path:
        valid_tasks_stack.append(item)
    else:
        print('error')

print(f"Total valid tasks available: {len(valid_tasks_look)}")
print(f"Total valid tasks available: {len(valid_tasks_clean)}")
print(f"Total valid tasks available: {len(valid_tasks_cool)}")
print(f"Total valid tasks available: {len(valid_tasks_heat)}")
print(f"Total valid tasks available: {len(valid_tasks_place)}")
print(f"Total valid tasks available: {len(valid_tasks_stack)}")

# Randomly choose 1000 tasks from valid_tasks
chosen_tasks_look = random.sample(valid_tasks_look, min(15, len(valid_tasks_look))) 
chosen_tasks_clean = random.sample(valid_tasks_clean, min(17, len(valid_tasks_clean))) 
chosen_tasks_cool = random.sample(valid_tasks_cool, min(17, len(valid_tasks_cool)))
chosen_tasks_heat = random.sample(valid_tasks_heat, min(16, len(valid_tasks_heat)))
chosen_tasks_place = random.sample(valid_tasks_place, min(19, len(valid_tasks_place)))
chosen_tasks_stack = random.sample(valid_tasks_stack, min(16, len(valid_tasks_stack))) 
chosen_tasks = chosen_tasks_look + chosen_tasks_clean + chosen_tasks_cool + chosen_tasks_heat + chosen_tasks_place + chosen_tasks_stack

# Save the chosen tasks to a new JSON file
output_data = {"valid_seen": chosen_tasks}
with open("alfred/data/splits/prompt_experiment100.json", "w") as output_file:
    json.dump(output_data, output_file, indent=4)

# Print summary of task counts
print(f"Valid tasks choosed: {len(chosen_tasks)}")
print(f"Randomly chosen 1000 tasks saved to 'prompt_experiment100.json'")


chosen_task_indices = [task['idx'] for task in chosen_tasks]


output_data = {"valid_seen_indices": chosen_task_indices}
with open("generate_dataset/result_analyze_output/prompt_experiment100_indices.json", "w") as output_file:
    json.dump(output_data, output_file, indent=4)

print(f"任务 idx 数量: {len(chosen_task_indices)}")
print(f"任务 idx 列表已保存到 'prompt_experiment100_indices.json'")