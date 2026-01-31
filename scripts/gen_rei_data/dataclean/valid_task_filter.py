import json
import os
import random

# Generate goals and corresponding reference counts
task_confusing_levels = ['1-1', '1-2', '1-3', '2-1', '2-2', '2-3', '3-1', '3-2', '3-3']
human_reference_numbers = [2, 2, 0, 2, 2, 0, 0, 0, 0]
last_reference_numbers = [1, 0, 0, 1, 0, 0, 1, 0, 0,]

# Initialize counters
total_tasks = 0
invalid_tasks = 0

# File paths
split_file_path = 'alfred/data/splits/LLaMa3.1-8B_merged2.json'
# split_file_path = 'alfred/data/splits/pre_experiment_tasks.json'
base_path = 'alfred/data/json_2.1.0 copy'

# Load data from the split file
with open(split_file_path, 'r') as f:
    split_data = json.load(f)

# Filter valid tasks and randomly select 100
unvalid_tasks_look = []
unvalid_tasks_clean = []
unvalid_tasks_cool = []
unvalid_tasks_heat = []
unvalid_tasks_place = []
unvalid_tasks_stack = []

for item in split_data["valid_seen"]:
    task_path = item["task"]
    reference = item["reference"].split(',')[0].strip()  # Process reference
    json_file_path = os.path.join(base_path, task_path, 'pp', f'ann_{int(item["repeat_idx"])}.json')
    valid_number = 0
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            json_content = json.load(f)
            total_tasks += 1
            for i, task_confusing_level in enumerate(task_confusing_levels):
                new_object = f"memory{task_confusing_level}"
                memory = item.get(new_object, "")
                memory_lines = memory.splitlines()

                # Process memory lines
                human_lines = [line.replace("Alice:", "Human:") for line in memory_lines if "Alice:" in line]
                if human_lines:
                    last_sentence = human_lines[-1].replace("Human:", "").strip()
                    human_lines = human_lines[:-1]

                human_lines = [line.replace("Human:", "") for line in human_lines]

                # Count occurrences of the reference
                human_references = sum(1 for line in human_lines if reference in line or f'{reference}s' in line or f'{reference}es' in line)
                last_sentence_references = 1 if reference in last_sentence else 0
                total_references = human_references + last_sentence_references
                # print(f'Goal is {task_confusing_level}. The number of human_references and last sentence is {human_references}, {last_sentence_references}')
                # Only keep valid tasks that meet reference count criteria
                if human_references >= human_reference_numbers[i] and last_sentence_references == last_reference_numbers[i]:
                    if task_confusing_level in ['3-1', '3-2', '3-3']:
                        if human_references <= 1:
                            valid_number = valid_number + 1
                        else: 
                            invalid_tasks += 1
                            print("!!!", reference)
                            print(f'Goal is {task_confusing_level}. The number of human_references and last sentence is {human_references}, {last_sentence_references}')
                            break
                    else: 
                        valid_number = valid_number + 1
                else:
                    # print(f"Task {task_confusing_level}: Found {total_references} references, expected >= {reference_number[i]}")
                    invalid_tasks += 1
                    print("???", reference)
                    print(f'Goal is {task_confusing_level}. The number of human_references and last sentence is {human_references}, {last_sentence_references}')
                    break
    else:
        print(f"File {json_file_path} does not exist!")
    if valid_number == 9:
        pass
    else:
        if 'look_at_obj' in task_path:
            unvalid_tasks_look.append(item)
        elif 'pick_clean' in task_path:
            unvalid_tasks_clean.append(item)
        elif 'pick_cool' in task_path:
            unvalid_tasks_cool.append(item)
        elif 'pick_heat' in task_path:
            unvalid_tasks_heat.append(item)
        elif 'pick_and_place_simple' in task_path:
            unvalid_tasks_place.append(item)
        elif 'pick_and_place_with_movable' in task_path:
            unvalid_tasks_stack.append(item)
        else:
            print('error')

unvalid_task = unvalid_tasks_look + unvalid_tasks_clean + unvalid_tasks_cool + unvalid_tasks_heat + unvalid_tasks_place + unvalid_tasks_stack

# Save the chosen tasks to a new JSON file
output_data = {"valid_seen": unvalid_task}
with open("alfred/data/splits/unvalid_tasks2.json", "w") as output_file:
    json.dump(output_data, output_file, indent=4)

# Print summary of task counts
print(f"Total tasks checked: {total_tasks}")
print(f"Valid tasks choosed: {len(unvalid_task)}")
print(f"Total invalid tasks: {invalid_tasks}")
print(f"Randomly chosen 1000 tasks saved to 'unvalid_tasks2.json'")