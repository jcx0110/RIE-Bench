import json
import os
generate_datesets = ["reference"]
generate_datesets = ['1-1', '1-2', '1-3', '2-1', '2-2', '2-3', '3-1', '3-2', '3-3', "reference"]
for generate_dateset in generate_datesets:
    split_file_path = 'alfred/data/splits/pre_experiment_tasks3.json'
    base_path = 'alfred/data/json_2.1.0 copy'

    with open(split_file_path, 'r') as f:
        split_data = json.load(f)

    dic_list = split_data["valid_seen"]
    file_num = 0
    error_num = 0
    try: 
        if generate_dateset == "reference":
            new_object = generate_dateset
            for item in dic_list[:]:
                task_path = item["task"]
                json_file_path = os.path.join(base_path, task_path, 'pp', f'ann_{int(item["repeat_idx"])}.json')
                
                if os.path.exists(json_file_path):
                    with open(json_file_path, 'r') as f:
                        json_content = json.load(f)
                        
                        reference = item.get(new_object, "")
                        print(reference)
                        
                        json_content['turk_annotations']['anns'][int(item["repeat_idx"])][new_object] = reference
                                            
                        with open(json_file_path, 'w') as f:
                            json.dump(json_content, f, indent=4)

                    file_num += 1
                else:
                    print(f"file {json_file_path} not exist!")
                    error_num += 1
        else: 
            new_object = f"memory{generate_dateset}"
            for item in dic_list[:]:
                task_path = item["task"]
                json_file_path = os.path.join(base_path, task_path, 'pp', f'ann_{int(item["repeat_idx"])}.json')
                
                if os.path.exists(json_file_path):
                    with open(json_file_path, 'r') as f:
                        json_content = json.load(f)
                        
                        memory = item.get(new_object, "")
                        memory_lines = memory.splitlines()
                        
                        human_lines = [line.replace("Alice:", "Human:") for line in memory_lines]
                        
                        if "Robot:" in human_lines[-1]:
                            human_lines = human_lines[:-1]
                            
                        if human_lines:
                            # last_sentence = human_lines[-1].replace("Human:", "").strip()  
                            last_sentence = human_lines[-1].strip()  
                            human_lines = human_lines[:-1]  
                        
                        # human_lines = [line.replace("Human:", "") for line in human_lines]
                        
                        json_content['turk_annotations']['anns'][int(item["repeat_idx"])][f"robot_human_memory{generate_dateset}"] = human_lines
                        
                        json_content['turk_annotations']['anns'][int(item["repeat_idx"])][f"task_desc{generate_dateset}"] = last_sentence
                        
                        with open(json_file_path, 'w') as f:
                            json.dump(json_content, f, indent=4)
                    
                    with open(json_file_path, 'r') as f:
                        updated_content = json.load(f)
                        stored_memory = updated_content['turk_annotations']['anns'][int(item["repeat_idx"])][f"robot_human_memory{generate_dateset}"]
                        stored_task_desc = updated_content['turk_annotations']['anns'][int(item["repeat_idx"])][f"task_desc{generate_dateset}"]
                        print(f"saved memory for task {task_path}:")
                        for line in stored_memory:
                            print(line)
                        print(stored_task_desc)
                        print(f"saved task_desc for task {task_path}: {json_file_path}")
                        print("\n" + "-"*50 + "\n")
                    
                    file_num += 1
                else:
                    print(f"file {json_file_path} not exist!")
                    error_num += 1
    except:
        pass
        
    print(f"{file_num} memories have been migrated!")
    if error_num:
        print(f"error: {error_num} tasks have not been processed!")
