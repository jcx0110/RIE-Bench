import json

def merge_json_files(file_list, output_file):
    merged_data = []

    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            valid_seen_data = data.get("valid_seen", [])
            merged_data.extend(valid_seen_data)
            print(f"Merged {file_path} with {len(valid_seen_data)} items.")

    merged_dict = {"valid_seen": merged_data}

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(merged_dict, f_out, ensure_ascii=False, indent=4)
    print(f"Saved merged file {output_file} with {len(merged_data)} items.")

file_list = [
    "alfred/data/splits/LLaMa3.1-8B_train.json", 
    "alfred/data/splits/LLaMa3.1-8B_train4.json",
    "alfred/data/splits/LLama3.1-8B_valid_all.json",
    "alfred/data/splits/LLaMa3.1-8B_validunseen_all.json"
]
output_file = "alfred/data/splits/LLaMa3.1-8B_custom_merged.json"  
merge_json_files(file_list, output_file)
