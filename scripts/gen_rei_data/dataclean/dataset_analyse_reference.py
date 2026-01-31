import json
import os
import random
from collections import defaultdict

split_file_path = 'alfred/data/splits/choosed_task.json'
base_path = 'alfred/data/json_2.1.0 copy'
output_file_path = "pre_experiment_tasks.json"

generate_goals = ['1-1', '1-2', '1-3', '2-1', '2-2', '2-3', '3-1', '3-2', '3-3']
human_reference_numbers = [2, 2, 0, 2, 2, 0, 0, 0, 0]
last_reference_numbers = [1, 0, 0, 1, 0, 0, 1, 0, 0]

def load_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        print(f"File {file_path} Doesn't exist!")
        return None

def count_references(memory, reference):
    memory_lines = memory.splitlines()
    human_lines = [line.replace("Alice:", "Human:") for line in memory_lines if "Alice:" in line]

    if human_lines:
        last_sentence = human_lines[-1].replace("Human:", "").strip()
        human_lines = human_lines[:-1]
    else:
        last_sentence = ""

    human_lines = [line.replace("Human:", "") for line in human_lines]
    human_references = sum(1 for line in human_lines if reference in line or f'{reference}s' in line or f'{reference}es' in line)
    last_sentence_references = 1 if reference in last_sentence else 0
    return human_references, last_sentence_references

split_data = load_json(split_file_path)
if not split_data:
    raise ValueError("Unable to load segmentation data file.")

valid_tasks = []
reference_distributions = defaultdict(list)   # distribution
total_tasks = 0
invalid_tasks = 0

for item in split_data.get("valid_seen", []):
    task_path = item["task"]
    reference = item["reference"].split(',')[0].strip()
    json_file_path = os.path.join(base_path, task_path, 'pp', f'ann_{int(item["repeat_idx"])}.json')

    json_content = load_json(json_file_path)
    if not json_content:
        continue

    total_tasks += 1
    valid_number = 0

    for i, generate_goal in enumerate(generate_goals):
        memory_key = f"memory{generate_goal}"
        memory = item.get(memory_key, "")

        human_references, last_sentence_references = count_references(memory, reference)
        reference_distributions[generate_goal].append(human_references)

        if (
            human_references >= human_reference_numbers[i] and
            last_sentence_references == last_reference_numbers[i]
        ):
            if generate_goal in ['3-1', '3-2', '3-3'] and human_references > 2:
                invalid_tasks += 1
                break
            valid_number += 1
        else:
            invalid_tasks += 1
            break

    if valid_number == len(generate_goals):
        valid_tasks.append(item)

chosen_tasks = random.sample(valid_tasks, min(1000, len(valid_tasks)))
output_data = {"valid_seen": chosen_tasks}

with open(output_file_path, "w") as output_file:
    json.dump(output_data, output_file, indent=4)

print(f"Total task：{total_tasks}")
print(f"Valid task number：{len(valid_tasks)}")
print(f"Invalid task number：{invalid_tasks}")

print("\nHuman Reference distribution：")
for goal, references in reference_distributions.items():
    print(f"Goal {goal}, len{len(references)}")
    print(f"Goal {goal}: {sorted(references)}")

