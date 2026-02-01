import re
from typing import Dict, List, Any


# --------------------------------------------------
# 0. Sanitize LLM output: keep only 1 DAG, strip comments
# --------------------------------------------------
def sanitize_dag_text(raw: str) -> str:
    """Make LLM DAG output more parse-friendly.

    - Keep only the LAST 'nodes:' block
    - Strip everything before that
    - Strip trailing comments starting with '#'
    - Drop empty or junk lines
    """
    lines = raw.splitlines()

    nodes_indices = [i for i, l in enumerate(lines) if l.strip().startswith("nodes:")]
    if not nodes_indices:
        return raw

    start_idx = nodes_indices[-1]
    lines = lines[start_idx:]

    cleaned = []
    for line in lines:
        if "#" in line:
            line = line.split("#", 1)[0]
        line = line.rstrip()
        if line.strip() == "":
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


# --------------------------------------------------
# 1. Parse DAG Text → Python dict (robust)
# --------------------------------------------------
def parse_dag_text_robust(dag_text: str) -> Dict[int, Dict[str, Any]]:
    dag_text = sanitize_dag_text(dag_text)
    lines = dag_text.strip().split("\n")

    nodes: Dict[int, Dict[str, Any]] = {}
    current_id = None

    node_re = re.compile(r"node_(\d+)\s*:")
    type_re = re.compile(r"type:\s*(.*)")
    name_re = re.compile(r"name:\s*(.*)")
    arm_re = re.compile(r"arm_num:\s*(\d+)")
    edge_re = re.compile(r"edge:\s*\[(.*?)\]")

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()

        # Detect node id
        node_match = node_re.match(line)
        if node_match:
            current_id = int(node_match.group(1))
            nodes[current_id] = {"edge": []}
            continue

        if current_id is None:
            continue

        t = type_re.match(line)
        if t:
            nodes[current_id]["type"] = t.group(1).strip()
            continue

        n = name_re.match(line)
        if n:
            name_val = n.group(1).strip()
            j = i + 1
            extra_parts = []
            while j < len(lines):
                next_line = lines[j]
                if (
                    next_line.startswith(" ")
                    or next_line.startswith("\t")
                ) and not any(
                    kw in next_line.strip().split()[0]
                    for kw in ["node_", "name:", "type:", "arm_num:", "edge:"]
                ):
                    extra_parts.append(next_line.strip())
                    j += 1
                else:
                    break
            if extra_parts:
                name_val = name_val + " " + " ".join(extra_parts)

            nodes[current_id]["name"] = name_val
            continue

        a = arm_re.match(line)
        if a:
            try:
                nodes[current_id]["arm_num"] = int(a.group(1))
            except ValueError:
                pass
            continue

        # edges
        e = edge_re.match(line)
        if e:
            edge_str = e.group(1).strip()
            if edge_str == "":
                nodes[current_id]["edge"] = []
            else:
                tokens = re.split(r"[,\s]+", edge_str)
                edges = []
                for tok in tokens:
                    tok = tok.strip()
                    if tok.isdigit():
                        edges.append(int(tok))
                nodes[current_id]["edge"] = edges
            continue

    valid_nodes = {
        nid: info for nid, info in nodes.items()
        if "name" in info and isinstance(info["name"], str)
    }

    for nid, info in valid_nodes.items():
        if "edge" not in info or not isinstance(info["edge"], list):
            info["edge"] = []

    existing_ids = set(valid_nodes.keys())
    for nid, info in valid_nodes.items():
        cleaned_edges = [e for e in info["edge"] if e in existing_ids and e != nid]
        info["edge"] = cleaned_edges

    return valid_nodes


# --------------------------------------------------
# 2. Convert DAG dict → Linear executable plan
# --------------------------------------------------
def dag_to_plan(nodes: Dict[int, Dict[str, Any]]) -> List[str]:
    if not nodes:
        raise ValueError("No valid nodes parsed from DAG text.")

    graph: Dict[int, List[int]] = {}
    indegree: Dict[int, int] = {}

    # Initialize
    for i in nodes:
        graph[i] = []
        indegree[i] = 0

    # Build edges
    for i, info in nodes.items():
        for pre in info.get("edge", []):
            graph[pre].append(i)
            indegree[i] += 1

    queue = [i for i in nodes if indegree[i] == 0]
    queue.sort()

    plan: List[str] = []

    while queue:
        current = queue.pop(0)
        plan.append(nodes[current]["name"])

        for nxt in graph[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
                queue.sort()

    if len(plan) != len(nodes):
        raise ValueError("Cycle detected or invalid DAG structure.")

    return plan


# --------------------------------------------------
# 3. Pipeline: DAG text → plan
# --------------------------------------------------
def dag_text_to_plan(dag_text: str) -> List[str]:
    nodes = parse_dag_text_robust(dag_text)
    return dag_to_plan(nodes)


# --------------------------------------------------
# 4. Simple manual test
# --------------------------------------------------
if __name__ == "__main__":
    dag_text = """
nodes:
    node_1: 
        type: occupy
        name: grasp "lemon"
        arm_num: 1
        edge: []
    node_2:
        type: release
        name: put "lemon" onto "plate"
        arm_num: 1
        edge: [1]
    node_3: 
        type: occupy
        name: grasp "apple"
        arm_num: 1
        edge: []
    node_4:
        type: release
        name: put "apple" onto "plate"
        arm_num: 1
        edge: [3]
    node_5: 
        type: occupy
        name: grasp "sponge"
        arm_num: 1
        edge: []
    node_6:
        type: tool use
        name: wipe "table" with "sponge"
        arm_num: 1
        edge: [5]
    node_7:
        type: operate
        name: open "drawer"
        arm_num: 1
        edge: []
    node_8:
        type: release
        name: put "sponge" into "drawer"
        arm_num: 1
        edge: [6, 7]
    node_9: 
        type: occupy
        name: grasp "mug"
        arm_num: 1
        edge: []
    node_10:
        type: release
        name: put "mug" into "drawer"
        arm_num: 1
        edge: [7, 9]
    node_11:
        type: operate
        name: close "drawer"
        arm_num: 1
        edge: [8, 10]
    node_12:
        type: complete
        name: task complete
        arm_num: 0
        edge: [2, 4, 11]  # final node
This DAG describes how to clean the table and store items.
nodes:
    node_1:
        name: THIS SECOND DAG SHOULD BE IGNORED
        edge: []
"""

    print("=== Parsed DAG → Executable Plan ===")
    plan = dag_text_to_plan(dag_text)
    for i, step in enumerate(plan, 1):
        print(f"{i}. {step}")
