import json
import math

def split_json_file(file_path, output_prefix):
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取valid_seen的数据
    valid_seen_data = data.get("valid_seen", [])
    num_items = len(valid_seen_data)
    split_size = math.ceil(num_items / 5)

    # 分割数据并保存到不同的文件中
    for i in range(8):
        start = i * split_size
        end = min(start + split_size, num_items)
        split_data = valid_seen_data[start:end]
        
        # 构建新的字典结构
        split_dict = {"valid_seen": split_data}
        
        # 输出分割后的文件
        output_file = f"{output_prefix}_part{i+1}.json"
        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(split_dict, f_out, ensure_ascii=False, indent=4)
        print(f"Saved {output_file} with {len(split_data)} items.")

# 使用示例
file_path = "alfred/data/splits/LLaMa3.1-8B_train1.json"  # 替换为你的JSON文件路径
output_prefix = "alfred/data/splits/LLaMa3.1-8B_train1"      # 输出文件名前缀
split_json_file(file_path, output_prefix)
