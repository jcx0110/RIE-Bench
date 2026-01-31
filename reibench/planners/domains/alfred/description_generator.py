import glob
import os
import sys
from collections import namedtuple

def find_build(fd_path):
    for release in ['release', 'release64', 'release32']:  # TODO: list the directory
        path = os.path.join(fd_path, 'builds/{}/'.format(release))
        if os.path.exists(path):
            return path
    # TODO: could also just automatically compile
    raise RuntimeError('Please compile FastDownward first [.../pddlstream$ ./downward/build.py]')

directory = os.path.dirname(os.path.abspath(__file__))
FD_PATH = os.path.join(directory, '../../downward')
TRANSLATE_PATH = os.path.join(find_build(FD_PATH), 'bin/translate')
sys.path.append(TRANSLATE_PATH)

from pddl_parser import pddl_file

Domain = namedtuple('Domain', ['name', 'requirements', 'types', 'type_dict', 'constants',
                               'predicates', 'predicate_dict', 'functions', 'actions', 'axioms'])
Problem = namedtuple('Problem', ['task_name', 'task_domain_name', 'task_requirements',
                                 'objects', 'init', 'goal', 'use_metric'])

domain_file = "domain.pddl"
problem_path = os.path.join(directory, 'p*.pddl')
problem_files = glob.glob(problem_path)

for problem_file in problem_files:

    task = pddl_file.open(domain_file, problem_file)
    count = {}
    description = ""
    
    # 统计物品的数量
    for obj in task.objects:
        if obj.type_name not in count.keys():
            count[obj.type_name] = 0
        count[obj.type_name] += 1

    # 为物品生成描述
    description += f'You have {count["object"]} objects in total. \n'
    description += "The robot is ready to interact with the environment. \n"
    
    # 生成每个物品的状态
    for obj in task.objects:
        description += f"Object {obj.name} is a {obj.type_name}. "
        if any(atom.predicate == "found" and atom.args[0] == obj.name for atom in task.init):
            description += f"{obj.name} has been found. "
        if any(atom.predicate == "holding" and atom.args[0] == "robot" and atom.args[1] == obj.name for atom in task.init):
            description += f"The robot is holding {obj.name}. "
        description += "\n"

    # 检查是否有目标
    if len(task.goal.parts) > 0:
        goals = task.goal.parts
    else:
        goals = [task.goal]
    description += f"Your goal is to achieve the following conditions: \n"
    
    # 生成目标描述
    for goal in goals:
        description += f"Ensure {goal.args[0]} is {goal.args[1]}. "
        description += "\n"

    # 输出到文件
    nl_file = os.path.splitext(problem_file)[0] + ".nl"
    with open(nl_file, 'w') as f:
        f.write(description)
