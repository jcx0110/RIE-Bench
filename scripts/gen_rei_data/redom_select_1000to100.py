import json

# 1. 读取原始 1000 个任务
in_path = "alfred/data/splits/pre_experiment_tasks3.json"
out_path = "alfred/data/splits/pre_experiment_tasks3_subsample_100.json"

with open(in_path, "r") as f:
    data = json.load(f)

# 假设任务都在 data["valid_seen"] 里
all_tasks = data["valid_seen"]
print("原任务数:", len(all_tasks))

# 2. 每 10 个取 1 个（从第 0 个开始：0, 10, 20, ..., 990）
selected_tasks = all_tasks[::10]

print("抽取后任务数:", len(selected_tasks))  # 预期为 100

# 3. 写回成同样结构的 JSON
new_data = {
    "valid_seen": selected_tasks
}

with open(out_path, "w") as f:
    json.dump(new_data, f, indent=4)

print("已保存到:", out_path)
