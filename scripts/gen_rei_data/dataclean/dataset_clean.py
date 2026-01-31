import json
import os
import random
import sys

# Generate goals and corresponding reference counts
task_confusing_level = ['1-1', '1-2', '1-3', '2-1', '2-2', '2-3', '3-1', '3-2', '3-3']
human_reference_numbers = [3, 3, 0, 3, 3, 0, 1, 1, 0]
last_reference_numbers = [1, 0, 0, 1, 0, 0, 1, 0, 0]

# Initialize counters
total_tasks = 0
invalid_tasks = 0

# File paths
split_file_path = 'alfred/data/splits/LLaMa3.1-8B_merged_copy.json'
base_path = 'alfred/data/json_2.1.0 copy'

# Load data from the split file
with open(split_file_path, 'r') as f:
    split_data = json.load(f)

for item_number, item in enumerate(split_data["valid_seen"][:]):
    task_path = item["task"]
    reference = item["reference"].split(',')[0].strip()  # Process reference
    json_file_path = os.path.join(base_path, task_path, 'pp', f'ann_{int(item["repeat_idx"])}.json')
    no_reference_last1 = None
    no_reference_last2 = None
    normal_reference_human_part1 = None
    normal_reference_human_part2 = None
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            json_content = json.load(f)
            for i, generate_goal in enumerate(task_confusing_level):
                new_object = f"memory{generate_goal}"
                memory = item.get(new_object, "")
                memory_lines = memory.splitlines()
                # Process memory lines
                human_lines = [line.replace("Alice:", "Human:") for line in memory_lines if "Alice:" in line]
                if human_lines:
                    last_sentence = human_lines[-1].replace("Human:", "").strip()
                # Count occurrences of the reference
                human_references = sum(1 for line in human_lines if reference in line)
                last_sentence_references = 1 if reference in last_sentence else 0
                # for some item '1-2' hace correct last sentence but '1-3' doesn't ...
                if last_sentence_references == 0 and generate_goal in ['1-2', '1-3']:
                    no_reference_last1 = last_sentence
                if last_sentence_references == 0 and generate_goal in ['2-2', '2-3', '3-2', '3-3']:
                    no_reference_last2 = last_sentence
                # for some item '1-1' hace correct human part but '1-3' doesn't ...
                if human_references >= human_reference_numbers[i] and generate_goal == '1-1':
                    normal_reference_human_part1 = memory_lines[:-1]
                if human_references >= human_reference_numbers[i] and generate_goal == '2-1':
                    normal_reference_human_part2 = memory_lines[:-1]
            for i, generate_goal in enumerate(task_confusing_level):
                new_object = f"memory{generate_goal}"
                memory = item.get(new_object, "")
                memory_lines = memory.splitlines()

                # Process memory lines
                human_lines = [line.replace("Alice:", "Human:") for line in memory_lines if "Alice:" in line]
                if human_lines:
                    last_sentence = human_lines[-1].replace("Human:", "").strip()
                    human_lines = human_lines[:-1]

                human_lines = [line.replace("Human:", "") for line in human_lines]

                # Count occurrences of the reference
                human_references = sum(1 for line in human_lines if reference in line)
                if generate_goal == '1-2' and human_references <= human_reference_numbers[i] and normal_reference_human_part1 != None:
                    item[new_object] = '\n'.join(normal_reference_human_part1) + '\n' + memory_lines[-1]
                    split_data["valid_seen"][item_number] = item
                if generate_goal == '2-2' and human_references <= human_reference_numbers[i] and normal_reference_human_part2 != None:
                    item[new_object] = '\n'.join(normal_reference_human_part2) + '\n' + memory_lines[-1]
                    split_data["valid_seen"][item_number] = item
                
                last_sentence_references = 1 if reference in last_sentence else 0
                total_references = human_references + last_sentence_references
                # Only keep valid tasks that meet reference count criteria
                
                if last_sentence_references != last_reference_numbers[i] and last_reference_numbers[i] == 0:
                    if generate_goal in ['1-2', '1-3']:
                        if no_reference_last1 == None:
                            no_reference_last1 = last_sentence.replace(reference, 'it')
                            item[new_object] = item.get(new_object, "").replace(last_sentence, no_reference_last1)
                        else:
                            print(f'create new, last sentence is {last_sentence_references}')
                            item[new_object] = item.get(new_object, "").replace(last_sentence, no_reference_last1)  # Overwrite last sentence
                        split_data["valid_seen"][item_number] = item
                    if generate_goal in ['2-2', '2-3', '3-2', '3-3'] and no_reference_last2 != None:
                        print(f'create new, last sentence is {last_sentence_references}')
                        item[new_object] = item.get(new_object, "").replace(last_sentence, no_reference_last2)  # Overwrite last sentence
                        split_data["valid_seen"][item_number] = item
                if human_references >= human_reference_numbers[i] and generate_goal in ['3-2', '3-3']:
                    if human_references > 1:
                        for line_number, line in enumerate(memory_lines[1: ]):
                            memory_lines[line_number+1] = memory_lines[line_number+1].replace(reference, 'it')
                        item[new_object] = '\n'.join(memory_lines)                        
                        split_data["valid_seen"][item_number] = item
                        print("create new reference 3-2 or 3-3")
                if human_references >= human_reference_numbers[i] and generate_goal in ['3-1']:
                    if human_references > 1:
                        for line_number, line in enumerate(memory_lines[1: -1]):
                            memory_lines[line_number+1] = memory_lines[line_number+1].replace(reference, 'it')
                        item[new_object] = '\n'.join(memory_lines)
                        split_data["valid_seen"][item_number] = item
                        print("create new reference 3-1")


save_path = 'alfred/data/splits/choosed_task.json'
# After processing, write back the modified data to the file
with open(save_path, 'w') as f:
    json.dump(split_data, f, indent=4)


print(f"Processed {total_tasks} tasks, with {invalid_tasks} invalid tasks.")
