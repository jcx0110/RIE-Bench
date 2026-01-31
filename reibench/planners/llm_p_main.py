import argparse
import glob
import json
import os
import random
import sys
import time
import backoff

import openai
import requests
import backoff
from openai import OpenAI  # 如果需要OpenAI模块，也可以继续使用。

import torch
import torch._dynamo
torch._dynamo.config.verbose = True
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModel
from transformers import StoppingCriteria, StoppingCriteriaList
from torch.nn import CrossEntropyLoss
import guidance
from guidance import models, select, gen, user, assistant
import logging
import re
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# FAST_DOWNWARD_ALIAS = "lama"
FAST_DOWNWARD_ALIAS = "seq-opt-fdss-1"

def postprocess(x):
    return x.strip()


def get_cost(x):
    splitted = x.split()
    counter = 0
    found = False
    cost = 1e5
    for i, xx in enumerate(splitted):
        if xx == "cost":
            counter = i
            found = True
            break
    if found:
        cost = float(splitted[counter+2])
    return cost


###############################################################################
#
# Define different problem domains
#
###############################################################################

DOMAINS = [
    "barman",
    "blocksworld",
    "floortile",
    "grippers",
    "storage",
    "termes",
    "tyreworld",
    "manipulation"
]


class Domain:
    def __init__(self):
        # every domain should contain the context as in "in-context learning" (ICL)
        # which are the example problem in natural language.
        # For instance, in our case, context is:
        # 1. p_example.nl  (a language description of the problem)
        # 2. p_example.pddl (the ground-truth problem pddl for the problem)
        # 3. p_example.sol  (the ground-truth solution in natural language to the problem)
        self.context = ("p_example.nl", "p_example.pddl", "p_example.sol")
        self.tasks = [] # should be list of tuples like (descritpion, ground_truth_pddl)

        self.grab_tasks()

    def grab_tasks(self):
        path = f"./llm-pddl/domains/{self.name}"
        nls = []
        for fn in glob.glob(f"{path}/*.nl"):
            fn_ = fn.split("/")[-1]
            if "domain" not in fn_ and "p_example" not in fn_:
                if os.path.exists(fn.replace("nl", "pddl")):
                    nls.append(fn_)
        sorted_nls = sorted(nls)
        self.tasks = [(nl, nl.replace("nl", "pddl")) for nl in sorted_nls]

    def __len__(self):
        return len(self.tasks)

    def get_task_suffix(self, i):
        nl, pddl = self.tasks[i]
        return f"{self.name}/{pddl}"

    def get_task_file(self, i):
        nl, pddl = self.tasks[i]
        return f"./llm-pddl/domains/{self.name}/{nl}", f"./llm-pddl/domains/{self.name}/{pddl}"

    def get_task(self, i):
        nl_f, pddl_f = self.get_task_file(i)
        with open(nl_f, 'r') as f:
            nl = f.read()
        with open(pddl_f, 'r') as f:
            pddl = f.read()
        return postprocess(nl), postprocess(pddl)

    def get_context(self):
        nl_f   = f"./llm-pddl/domains/{self.name}/{self.context[0]}"
        pddl_f = f"./llm-pddl/domains/{self.name}/{self.context[1]}"
        sol_f  = f"./llm-pddl/domains/{self.name}/{self.context[2]}"
        with open(nl_f, 'r') as f:
            nl   = f.read()
        with open(pddl_f, 'r') as f:
            pddl = f.read()
        with open(sol_f, 'r') as f:
            sol  = f.read()
        return postprocess(nl), postprocess(pddl), postprocess(sol)

    def get_domain_pddl(self):
        domain_pddl_f = self.get_domain_pddl_file()
        with open(domain_pddl_f, 'r') as f:
            domain_pddl = f.read()
        return postprocess(domain_pddl)

    def get_domain_pddl_file(self):
        domain_pddl_f = f"./llm-pddl/domains/{self.name}/domain.pddl"
        return domain_pddl_f

    def get_domain_nl(self):
        domain_nl_f = self.get_domain_nl_file()
        try:
            with open(domain_nl_f, 'r') as f:
                domain_nl = f.read()
        except:
            domain_nl = "Nothing"
        return postprocess(domain_nl)

    def get_domain_nl_file(self):
        domain_nl_f = f"./llm-pddl/domains/{self.name}/domain.nl"
        return domain_nl_f

class Alfred(Domain):
    name = "alfred" # this should match the directory name

class Barman(Domain):
    name = "barman" # this should match the directory name

class Floortile(Domain):
    name = "floortile" # this should match the directory name

class Termes(Domain):
    name = "termes" # this should match the directory name

class Tyreworld(Domain):
    name = "tyreworld" # this should match the directory name

class Grippers(Domain):
    name = "grippers" # this should match the directory name

class Storage(Domain):
    name = "storage" # this should match the directory name

class Blocksworld(Domain):
    name = "blocksworld" # this should match the directory name

class Manipulation(Domain):
    name = "manipulation" # this should match the directory name

###############################################################################
#
# The agent that leverages classical planner to help LLMs to plan
#
###############################################################################


class Planner:
    def __init__(self):
        self.openai_api_keys = self.load_openai_keys()
        self.use_chatgpt = True
        self.modelname = "gpt"
        # meta-llama/Llama-3.1-8B-Instruct, mistralai/Ministral-8B-Instruct-2410, deepseek-ai/deepseek-math-7b-instruct
        self.device = torch.device("cuda:0")

    def load_openai_keys(self,):
        openai_keys_file = os.path.join(os.getcwd(), "llm-pddl/keys/openai_keys.txt")
        with open(openai_keys_file, "r") as f:
            context = f.read()
        context_lines = context.strip().split('\n')
        print(context_lines)
        return context_lines

    def create_llm_prompt(self, task_nl, domain_nl):
        # Baseline 1 (LLM-as-P): directly ask the LLM for plan
        prompt = f"{domain_nl} \n" + \
                 f"Now consider a planning problem. " + \
                 f"The problem description is: \n {task_nl} \n" + \
                 f"Can you provide an optimal plan, in the way of a " + \
                 f"sequence of behaviors, to solve the problem?"
        return prompt

    def create_llm_stepbystep_prompt(self, task_nl, domain_nl):
        # Baseline 1 (LLM-as-P): directly ask the LLM for plan
        prompt = f"{domain_nl} \n" + \
                 f"Now consider a planning problem. " + \
                 f"The problem description is: \n {task_nl} \n" + \
                 f"Can you provide an optimal plan, in the way of a " + \
                 f"sequence of behaviors, to solve the problem? \n" + \
                 f"Please think step by step."
        return prompt

    def create_llm_tot_ic_prompt(self, task_nl, domain_nl, context, plan):
        context_nl, context_pddl, context_sol = context
        prompt = f"Given the current state, provide the set of feasible actions and their corresponding next states, using the format 'action -> state'. \n" + \
                 f"Keep the list short. Think carefully about the requirements of the actions you select and make sure they are met in the current state. \n" + \
                 f"Start with actions that are most likely to make progress towards the goal. \n" + \
                 f"Only output one action per line. Do not return anything else. " + \
                 f"Here are the rules. \n {domain_nl} \n\n" + \
                 f"An example planning problem is: \n {context_nl} \n" + \
                 f"A plan for the example problem is: \n {context_sol} \n" + \
                 f"Now I have a new planning problem and its description is: \n {task_nl} \n" + \
                 f"You have taken the following actions: \n {plan} \n"
        # print(prompt)
        return prompt

    def create_llm_tot_ic_value_prompt(self, task_nl, domain_nl, context, plan):
        context_nl, context_pddl, context_sol = context
        context_sure_1 = context_sol.split('\n')[0]
        context_sure_2 = context_sol.split('\n')[0] + context_sol.split('\n')[1]
        context_impossible_1 = '\n'.join(context_sol.split('\n')[1:])
        context_impossible_2 = context_sol.split('\n')[-1]
        '''
        prompt = f"Evaluate if a given plan reaches the goal or is an optimal partial plan towards the goal (reached/sure/maybe/impossible). \n" + \
                 f"Only answer 'reached' if the goal conditions are reached by the exact plan in the prompt. \n" + \
                 f"Only answer 'sure' if you are sure that preconditions are satisfied for all actions in the plan, and the plan makes fast progress towards the goal. \n" + \
                 f"Answer 'impossible' if one of the actions has unmet preconditions. \n" + \
                 f"Here are the rules. \n {domain_nl} \n\n" + \
                 f"Here are some example evaluations for the planning problem: \n {context_nl} \n\n " + \
                 f"Plan: {context_sure_1} \n" + \
                 f"Answer: Sure. \n\n" + \
                 f"Plan: {context_sure_2} \n" + \
                 f"Answer: Sure. \n\n" + \
                 f"Plan: {context_sol} \n" + \
                 f"Answer: Reached. \n\n" + \
                 f"Plan: {context_impossible_1} \n" + \
                 f"Answer: Impossible. \n\n" + \
                 f"Plan: {context_impossible_2} \n" + \
                 f"Answer: Impossible. \n\n" + \
                 f"Now I have a new planning problem and its description is: \n {task_nl} \n" + \
                 f"Evaluate the following partial plan as reached/sure/maybe/impossible. DO NOT RETURN ANYTHING ELSE. DO NOT TRY TO COMPLETE THE PLAN. \n" + \
                 f"Plan: {plan} \n"
        '''
        prompt = f"Determine if a given plan reaches the goal or give your confidence score that it is an optimal partial plan towards the goal (reached/impossible/0-1). \n" + \
                 f"Only answer 'reached' if the goal conditions are reached by the exact plan in the prompt. \n" + \
                 f"Answer 'impossible' if one of the actions has unmet preconditions. \n" + \
                 f"Otherwise,give a number between 0 and 1 as your evaluation of the partial plan's progress towards the goal. \n" + \
                 f"Here are the rules. \n {domain_nl} \n\n" + \
                 f"Here are some example evaluations for the planning problem: \n {context_nl} \n\n " + \
                 f"Plan: {context_sure_1} \n" + \
                 f"Answer: 0.8. \n\n" + \
                 f"Plan: {context_sure_2} \n" + \
                 f"Answer: 0.9. \n\n" + \
                 f"Plan: {context_sol} \n" + \
                 f"Answer: Reached. \n\n" + \
                 f"Plan: {context_impossible_1} \n" + \
                 f"Answer: Impossible. \n\n" + \
                 f"Plan: {context_impossible_2} \n" + \
                 f"Answer: Impossible. \n\n" + \
                 f"Now I have a new planning problem and its description is: \n {task_nl} \n" + \
                 f"Evaluate the following partial plan as reached/impossible/0-1. DO NOT RETURN ANYTHING ELSE. DO NOT TRY TO COMPLETE THE PLAN. \n" + \
                 f"Plan: {plan} \n"

        return prompt


    def tot_bfs(self, task_nl, domain_nl, context, time_left=200, max_depth=2):
        from queue import PriorityQueue
        start_time = time.time()
        plan_queue = PriorityQueue()
        plan_queue.put((0, ""))
        while time.time() - start_time < time_left and not plan_queue.empty():
            priority, plan = plan_queue.get()
            # print (priority, plan)
            steps = plan.split('\n')
            if len(steps) > max_depth:
                return ""
            candidates_prompt = self.create_llm_tot_ic_prompt(task_nl, domain_nl, context, plan)
            candidates = self.query(candidates_prompt).strip()
            print (candidates)
            lines = candidates.split('\n')
            for line in lines:
                if time.time() - start_time > time_left:
                    break
                if len(line) > 0 and '->' in line:
                    new_plan = plan + "\n" + line
                    value_prompt = self.create_llm_tot_ic_value_prompt(task_nl, domain_nl, context, new_plan)
                    answer = self.query(value_prompt).strip().lower()
                    print(new_plan)
                    print("Response \n" + answer)

                    if "reached" in answer:
                        return new_plan

                    if "impossible" in answer:
                        continue

                    if "answer: " in answer:
                        answer = answer.split("answer: ")[1]

                    try:
                        score = float(answer)
                    except ValueError:
                        continue

                    if score > 0:
                        new_priority = priority + 1 / score
                        plan_queue.put((new_priority, new_plan))

        return ""

    def create_llm_ic_prompt(self, task_nl, domain_nl, context):
        # Baseline 2 (LLM-as-P with context): directly ask the LLM for plan
        context_nl, context_pddl, context_sol = context
        prompt = f"{domain_nl} \n" + \
                 f"An example planning problem is: \n {context_nl} \n" + \
                 f"A plan for the example problem is: \n {context_sol} \n" + \
                 f"Now I have a new planning problem and its description is: \n {task_nl} \n" + \
                 f"Can you provide an optimal plan, in the way of a " + \
                 f"sequence of behaviors, to solve the problem?"
        return prompt

    def create_llm_pddl_prompt(self, task_nl, domain_pddl, domain_nl):
        # Baseline 3 (LM+P w/o context), no context, create the problem PDDL
        prompt = f"The domain of this problem is {domain_pddl} \n" + \
                 f"Now consider a planning problem. " + \
                 f"The problem description is: \n {task_nl} \n" + \
                 f"Provide me with the problem PDDL file that describes " + \
                 f"the planning problem directly without further explanations." +\
                 f"Keep the domain name consistent in the problem PDDL. Only return the PDDL file. Do not return anything else. Ensure correct PDDL syntax with properly formatted predicates and actions. Avoid extraneous closing brackets. Answer: "
        return prompt

    def create_llm_ic_pddl_prompt(self, task_nl, domain_pddl, context):
        # our method (LM+P), create the problem PDDL given the context
        context_nl, context_pddl, context_sol = context
        prompt = f"I want you to solve planning problems. \n" + \
                 f"The domain of this problem is {domain_pddl} \n" + \
                 f"Please note that the generated predicates need to match the domain. \n" + \
                 f"An example planning problem is: \n {context_nl} \n" + \
                 f"The problem PDDL file to this problem is: \n {context_pddl} \n" + \
                 f"Now I have a new planning problem and its description is: \n {task_nl} \n" + \
                 f"Provide me with the problem PDDL file that describes " + \
                 f"the new planning problem directly without further explanations? Only return the PDDL file. Do not return anything else. Ensure correct PDDL syntax with properly formatted predicates and actions. Avoid extraneous closing brackets. Answer: "
        return prompt

    def query(self, prompt_text):
        print("prompt: ", prompt_text)
        server_flag = 0
        server_cnt = 0
        result_text = ""
        modelname=self.modelname
        if "gpt" in modelname:
            while server_cnt < 10:
                self.openai_api_keys = self.openai_api_keys[0].strip()
                # try:
                # self.update_key()
                # @backoff.on_exception(backoff.expo, openai.error.RateLimitError)
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {self.openai_api_keys}",
                    # "Content-Type": "application/json"
                }
                data = {
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "user", "content": prompt_text}
                    ],
                    "max_tokens": 5000,
                    "temperature": 0.1
                }
                response = requests.post(url, headers=headers, json=data)
                result_text = response.json()
                print(result_text)  # 打印整个响应，查看实际结构        
                result_text = response.json()["choices"][0]["message"]["content"]
                start_index = result_text.find('(')
                end_index = result_text.rfind(')')

                result_text = result_text[start_index:end_index+1]
                server_flag = 1
                if server_flag:
                    break
        elif "llama" in modelname or "mistralai" in modelname or "deepseek" in modelname:
            messages = [
                {"role": "system", "content": "You are a chatbot who always help with people!"},
                {"role": "user", "content": prompt_text}
            ]
            prompt_text = ""
            for message in messages:
                if message["role"] == "system":
                    prompt_text += "[SYSTEM] " + message["content"] + " [/SYSTEM]\n"
                elif message["role"] == "user":
                    prompt_text += "[USER] " + message["content"] + " [/USER]\n"
            prompt_text += "[ASSISTANT] "
            self.tokenizer = AutoTokenizer.from_pretrained(self.modelname, force_download=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.modelname, torch_dtype=torch.float16).to(self.device)
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)  # 将输入数据也移动到 GPU 上
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=1000,  # 控制新生成的 token 数量
                temperature=0.1
                )
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("result", result)
            result_text = result.split("[ASSISTANT]")[1].split("[/ASSISTANT]")[0].strip()
            start_index = result_text.find('(')
            end_index = result_text.rfind(')')

            result_text = result_text[start_index:end_index+1]
        print("result_text:", result_text)
        return result_text

    def update_key(self):
        curr_key = self.openai_api_keys[0]
        openai.api_key = curr_key
        self.openai_api_keys.remove(curr_key)
        self.openai_api_keys.append(curr_key)

    def parse_result(self, pddl_string):
        # remove extra texts
        #try:
        #    beg = pddl_string.find("```") + 3
        #    pddl_string = pddl_string[beg: beg + pddl_string[beg:].find("```")]
        #except:
        #    raise Exception("[error] cannot find ```pddl-file``` in the pddl string")

        # remove comments, they can cause error
        #t0 = time.time()
        #while pddl_string.find(";") >= 0:
        #    start = pddl_string.find(";")
        #    i = 0
        #    while pddl_string[start+i] != ")" and pddl_string[start+i] != "\n":
        #        i += 1
        #    pddl_string = pddl_string[:start] + pddl_string[start+i:]
        #pddl_string = pddl_string.strip() + "\n"
        #t1 = time.time()
        #print(f"[info] remove comments takes {t1-t0} sec")
        return pddl_string

    def plan_to_language(self, plan, task_nl, domain_nl, domain_pddl):
        domain_pddl_ = " ".join(domain_pddl.split())
        task_nl_ = " ".join(task_nl.split())
        prompt = f"A planning problem is described as: \n {task_nl} \n" + \
                 f"The corresponding domain PDDL file is: \n {domain_pddl_} \n" + \
                 f"The optimal PDDL plan is: \n {plan} \n" + \
                 f"Transform the PDDL plan into a sequence of behaviors without further explanation."
        res = self.query(prompt).strip() + "\n"
        return res


def llm_ic_pddl_planner(args, planner, domain):
    """
    Our method:
        context: (task natural language, task problem PDDL)
        Condition on the context (task description -> task problem PDDL),
        LLM will be asked to provide the problem PDDL of a new task description.
        Then, we use a planner to find the near optimal solution, and translate
        that back to natural language.
    """
    context          = domain.get_context()
    domain_pddl      = domain.get_domain_pddl()
    domain_pddl_file = domain.get_domain_pddl_file()
    domain_nl        = domain.get_domain_nl()
    domain_nl_file   = domain.get_domain_nl_file()

    # create the tmp / result folders
    problem_folder = f"./experiments/run{args.run}/problems/llm_ic_pddl/{domain.name}"
    plan_folder    = f"./experiments/run{args.run}/plans/llm_ic_pddl/{domain.name}"
    result_folder  = f"./experiments/run{args.run}/results/llm_ic_pddl/{domain.name}"

    if not os.path.exists(problem_folder):
        os.system(f"mkdir -p {problem_folder}")
    if not os.path.exists(plan_folder):
        os.system(f"mkdir -p {plan_folder}")
    if not os.path.exists(result_folder):
        os.system(f"mkdir -p {result_folder}")

    task = args.task

    start_time = time.time()

    # A. generate problem pddl file
    task_suffix        = domain.get_task_suffix(task)
    task_nl, task_pddl = domain.get_task(task) 
    prompt             = planner.create_llm_ic_pddl_prompt(task_nl, domain_pddl, context)
    raw_result         = planner.query(prompt)
    task_pddl_         = planner.parse_result(raw_result)

    # B. write the problem file into the problem folder
    task_pddl_file_name = f"./experiments/run{args.run}/problems/llm_ic_pddl/{task_suffix}"
    with open(task_pddl_file_name, "w") as f:
        f.write(task_pddl_)
    time.sleep(1)

    ## C. run fastforward to plan
    plan_file_name = f"./experiments/run{args.run}/plans/llm_ic_pddl/{task_suffix}"
    sas_file_name  = f"./experiments/run{args.run}/plans/llm_ic_pddl/{task_suffix}.sas"
    os.system(f"python ./llm-pddl/downward/fast-downward.py --alias {FAST_DOWNWARD_ALIAS} " + \
              f"--search-time-limit {args.time_limit} --plan-file {plan_file_name} " + \
              f"--sas-file {sas_file_name} " + \
              f"{domain_pddl_file} {task_pddl_file_name}")

    # D. collect the least cost plan
    best_cost = 1e10
    best_plan = None
    for fn in glob.glob(f"{plan_file_name}.*"):
        with open(fn, "r") as f:
            plans = f.readlines()
            cost = get_cost(plans[-1])
            if cost < best_cost:
                best_cost = cost
                best_plan = "\n".join([p.strip() for p in plans[:-1]])

    # E. translate the plan back to natural language, and write it to result
    # commented out due to exceeding token limit of gpt-4
    '''
    if best_plan:
        plans_nl = planner.plan_to_language(best_plan, task_nl, domain_nl, domain_pddl)
        plan_nl_file_name = f"./experiments/run{args.run}/results/llm_ic_pddl/{task_suffix}"
        with open(plan_nl_file_name, "w") as f:
            f.write(plans_nl)
    '''
    end_time = time.time()
    if best_plan:
        print(f"[info] task {task} takes {end_time - start_time} sec, found a plan with cost {best_cost}")
    else:
        print(f"[info] task {task} takes {end_time - start_time} sec, no solution found")


def llm_pddl_planner(args, planner, domain):
    """
    Baseline method:
        Same as ours, except that no context is given. In other words, the LLM
        will be asked to directly give a problem PDDL file without any context.
    """
    context          = domain.get_context()
    domain_pddl      = domain.get_domain_pddl()
    domain_pddl_file = domain.get_domain_pddl_file()
    domain_nl        = domain.get_domain_nl()
    domain_nl_file   = domain.get_domain_nl_file()

    # create the tmp / result folders
    problem_folder = f"./experiments/run{args.run}/problems/llm_pddl/{domain.name}"
    plan_folder    = f"./experiments/run{args.run}/plans/llm_pddl/{domain.name}"
    result_folder  = f"./experiments/run{args.run}/results/llm_pddl/{domain.name}"

    if not os.path.exists(problem_folder):
        os.system(f"mkdir -p {problem_folder}")
    if not os.path.exists(plan_folder):
        os.system(f"mkdir -p {plan_folder}")
    if not os.path.exists(result_folder):
        os.system(f"mkdir -p {result_folder}")

    task = args.task

    start_time = time.time()

    # A. generate problem pddl file
    task_suffix        = domain.get_task_suffix(task)
    task_nl, task_pddl = domain.get_task(task) 
    prompt             = planner.create_llm_pddl_prompt(task_nl, domain_pddl, domain_nl)
    raw_result         = planner.query(prompt)
    task_pddl_         = planner.parse_result(raw_result)

    # B. write the problem file into the problem folder
    task_pddl_file_name = f"./experiments/run{args.run}/problems/llm_pddl/{task_suffix}"
    with open(task_pddl_file_name, "w") as f:
        f.write(task_pddl_)
    time.sleep(1)

    # C. run fastforward to plan
    plan_file_name = f"./experiments/run{args.run}/plans/llm_pddl/{task_suffix}"
    sas_file_name  = f"./experiments/run{args.run}/plans/llm_pddl/{task_suffix}.sas"
    os.system(f"python ./llm-pddl/downward/fast-downward.py --alias {FAST_DOWNWARD_ALIAS} " + \
              f"--search-time-limit {args.time_limit} --plan-file {plan_file_name} " + \
              f"--sas-file {sas_file_name} " + \
              f"{domain_pddl_file} {task_pddl_file_name}")

    # D. collect the least cost plan
    best_cost = 1e10
    best_plan = None
    for fn in glob.glob(f"{plan_file_name}.*"):
        with open(fn, "r") as f:
            try:
                plans = f.readlines()
                cost = get_cost(plans[-1])
                if cost < best_cost:
                    best_cost = cost
                    best_plan = "\n".join([p.strip() for p in plans[:-1]])
            except:
                continue

    # E. translate the plan back to natural language, and write it to result
    # commented out due to exceeding token limit of gpt-4
    '''
    if best_plan:
        plans_nl = planner.plan_to_language(best_plan, task_nl, domain_nl, domain_pddl)
        plan_nl_file_name = f"./experiments/run{args.run}/results/llm_pddl/{task_suffix}"
        with open(plan_nl_file_name, "w") as f:
            f.write(plans_nl)
    '''
    end_time = time.time()
    if best_plan:
        print(f"[info] task {task} takes {end_time - start_time} sec, found a plan with cost {best_cost}")
        print("best plan: ", best_plan)
    else:
        print(f"[info] task {task} takes {end_time - start_time} sec, no solution found")


def llm_planner(args, planner, domain):
    """
    Baseline method:
        The LLM will be asked to directly give a plan based on the task description.
    """
    context          = domain.get_context()
    domain_pddl      = domain.get_domain_pddl()
    domain_pddl_file = domain.get_domain_pddl_file()
    domain_nl        = domain.get_domain_nl()
    domain_nl_file   = domain.get_domain_nl_file()

    # create the tmp / result folders
    problem_folder = f"./experiments/run{args.run}/problems/llm/{domain.name}"
    plan_folder    = f"./experiments/run{args.run}/plans/llm/{domain.name}"
    result_folder  = f"./experiments/run{args.run}/results/llm/{domain.name}"

    if not os.path.exists(problem_folder):
        os.system(f"mkdir -p {problem_folder}")
    if not os.path.exists(plan_folder):
        os.system(f"mkdir -p {plan_folder}")
    if not os.path.exists(result_folder):
        os.system(f"mkdir -p {result_folder}")

    task = args.task

    start_time = time.time()

    # A. generate problem pddl file
    task_suffix        = domain.get_task_suffix(task)
    task_nl, task_pddl = domain.get_task(task) 
    prompt             = planner.create_llm_prompt(task_nl, domain_nl)
    text_plan          = planner.query(prompt)

    # B. write the problem file into the problem folder
    text_plan_file_name = f"./experiments/run{args.run}/results/llm/{task_suffix}"
    with open(text_plan_file_name, "w") as f:
        f.write(text_plan)
    end_time = time.time()
    print(f"[info] task {task} takes {end_time - start_time} sec")


def llm_stepbystep_planner(args, planner, domain):
    """
    Baseline method:
        The LLM will be asked to directly give a plan based on the task description.
    """
    context          = domain.get_context()
    domain_pddl      = domain.get_domain_pddl()
    domain_pddl_file = domain.get_domain_pddl_file()
    domain_nl        = domain.get_domain_nl()
    domain_nl_file   = domain.get_domain_nl_file()

    # create the tmp / result folders
    problem_folder = f"./experiments/run{args.run}/problems/llm_step/{domain.name}"
    plan_folder    = f"./experiments/run{args.run}/plans/llm_step/{domain.name}"
    result_folder  = f"./experiments/run{args.run}/results/llm_step/{domain.name}"

    if not os.path.exists(problem_folder):
        os.system(f"mkdir -p {problem_folder}")
    if not os.path.exists(plan_folder):
        os.system(f"mkdir -p {plan_folder}")
    if not os.path.exists(result_folder):
        os.system(f"mkdir -p {result_folder}")

    task = args.task

    start_time = time.time()

    # A. generate problem pddl file
    task_suffix        = domain.get_task_suffix(task)
    task_nl, task_pddl = domain.get_task(task) 
    prompt             = planner.create_llm_stepbystep_prompt(task_nl, domain_nl)
    text_plan          = planner.query(prompt)

    # B. write the problem file into the problem folder
    text_plan_file_name = f"./experiments/run{args.run}/results/llm_step/{task_suffix}"
    with open(text_plan_file_name, "w") as f:
        f.write(text_plan)
    end_time = time.time()
    print(f"[info] task {task} takes {end_time - start_time} sec")


def llm_tot_ic_planner(args, planner, domain):
    """
    Tree of Thoughts planner
    """
    context          = domain.get_context()
    domain_pddl      = domain.get_domain_pddl()
    domain_pddl_file = domain.get_domain_pddl_file()
    domain_nl        = domain.get_domain_nl()
    domain_nl_file   = domain.get_domain_nl_file()

    # create the tmp / result folders
    problem_folder = f"./experiments/run{args.run}/problems/llm_tot_ic/{domain.name}"
    plan_folder    = f"./experiments/run{args.run}/plans/llm_tot_ic/{domain.name}"
    result_folder  = f"./experiments/run{args.run}/results/llm_tot_ic/{domain.name}"

    if not os.path.exists(problem_folder):
        os.system(f"mkdir -p {problem_folder}")
    if not os.path.exists(plan_folder):
        os.system(f"mkdir -p {plan_folder}")
    if not os.path.exists(result_folder):
        os.system(f"mkdir -p {result_folder}")

    task = args.task

    start_time = time.time()

    # A. generate problem pddl file
    task_suffix        = domain.get_task_suffix(task)
    task_nl, task_pddl = domain.get_task(task)
    text_plan = planner.tot_bfs(task_nl, domain_nl, context, time_left=200, max_depth=10)

    # B. write the problem file into the problem folder
    text_plan_file_name = f"./experiments/run{args.run}/results/llm_tot_ic/{task_suffix}"
    with open(text_plan_file_name, "w") as f:
        f.write(text_plan)
    end_time = time.time()
    print(f"[info] task {task} takes {end_time - start_time} sec")


def llm_ic_planner(args, planner, domain):
    """
    Baseline method:
        The LLM will be asked to directly give a plan based on the task description.
    """
    context          = domain.get_context()
    domain_pddl      = domain.get_domain_pddl()
    domain_pddl_file = domain.get_domain_pddl_file()
    domain_nl        = domain.get_domain_nl()
    domain_nl_file   = domain.get_domain_nl_file()

    # create the tmp / result folders
    problem_folder = f"./experiments/run{args.run}/problems/llm_ic/{domain.name}"
    plan_folder    = f"./experiments/run{args.run}/plans/llm_ic/{domain.name}"
    result_folder  = f"./experiments/run{args.run}/results/llm_ic/{domain.name}"

    if not os.path.exists(problem_folder):
        os.system(f"mkdir -p {problem_folder}")
    if not os.path.exists(plan_folder):
        os.system(f"mkdir -p {plan_folder}")
    if not os.path.exists(result_folder):
        os.system(f"mkdir -p {result_folder}")

    task = args.task

    start_time = time.time()

    # A. generate problem pddl file
    task_suffix        = domain.get_task_suffix(task)
    task_nl, task_pddl = domain.get_task(task) 
    prompt             = planner.create_llm_ic_prompt(task_nl, domain_nl, context)
    text_plan          = planner.query(prompt)

    # B. write the problem file into the problem folder
    text_plan_file_name = f"./experiments/run{args.run}/results/llm_ic/{task_suffix}"
    with open(text_plan_file_name, "w") as f:
        f.write(text_plan)
    end_time = time.time()
    print(f"[info] task {task} takes {end_time - start_time} sec")


def print_all_prompts(planner):
    for domain_name in DOMAINS:
        domain = eval(domain_name.capitalize())()
        context = domain.get_context()
        domain_pddl = domain.get_domain_pddl()
        domain_pddl_file = domain.get_domain_pddl_file()
        domain_nl = domain.get_domain_nl()
        
        for folder_name in [
            f"./prompts/llm/{domain.name}",
            f"./prompts/llm_step/{domain.name}",
            f"./prompts/llm_ic/{domain.name}",
            f"./prompts/llm_pddl/{domain.name}",
            f"./prompts/llm_ic_pddl/{domain.name}"]:
            if not os.path.exists(folder_name):
                os.system(f"mkdir -p {folder_name}")

        for task in range(len(domain)):
            task_nl_file, task_pddl_file = domain.get_task_file(task) 
            task_nl, task_pddl = domain.get_task(task) 
            task_suffix = domain.get_task_suffix(task)

            llm_prompt = planner.create_llm_prompt(task_nl, domain_nl)
            llm_stepbystep_prompt = planner.create_llm_stepbystep_prompt(task_nl, domain_nl)
            llm_ic_prompt = planner.create_llm_ic_prompt(task_nl, domain_nl, context)
            llm_pddl_prompt = planner.create_llm_pddl_prompt(task_nl, domain_pddl, domain_nl)
            llm_ic_pddl_prompt = planner.create_llm_ic_pddl_prompt(task_nl, domain_pddl, context)
            with open(f"./prompts/llm/{task_suffix}.prompt", "w") as f:
                f.write(llm_prompt)
            with open(f"./prompts/llm_step/{task_suffix}.prompt", "w") as f:
                f.write(llm_stepbystep_prompt)
            with open(f"./prompts/llm_ic/{task_suffix}.prompt", "w") as f:
                f.write(llm_ic_prompt)
            with open(f"./prompts/llm_pddl/{task_suffix}.prompt", "w") as f:
                f.write(llm_pddl_prompt)
            with open(f"./prompts/llm_ic_pddl/{task_suffix}.prompt", "w") as f:
                f.write(llm_ic_pddl_prompt)


def run_planner(domain_name="alfred", method_name="llm_pddl_planner", time_limit=200, 
                task=0, run=0, print_prompts=False, model="gpt-4"):
    """
    运行 LLM 规划器，可在其他代码中调用。
    
    :param domain_name: 任务规划的领域
    :param method_name: 选择的 LLM 规划方法
    :param time_limit: 任务的时间限制
    :param task: 任务编号
    :param run: 运行编号
    :param print_prompts: 是否打印 prompts
    :param model: 选择的 LLM 模型
    """
    # 初始化 domain
    domain = eval(domain_name.capitalize())()

    # 初始化 planner
    planner = Planner()

    # 选择执行方法
    method_dict = {
        "llm_ic_pddl_planner"   : llm_ic_pddl_planner,
        "llm_pddl_planner"      : llm_pddl_planner,
        "llm_planner"           : llm_planner,
        "llm_stepbystep_planner": llm_stepbystep_planner,
        "llm_ic_planner"        : llm_ic_planner,
        "llm_tot_ic_planner"    : llm_tot_ic_planner,
    }

    if method_name not in method_dict:
        raise ValueError(f"Invalid method: {method_name}")

    method = method_dict[method_name]

    if print_prompts:
        print_all_prompts(planner)
    else:
        method(domain=domain, planner=planner, time_limit=time_limit, 
               task=task, run=run, model=model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-Planner")
    parser.add_argument('--domain', type=str, choices=DOMAINS, default="alfred")
    parser.add_argument('--method', type=str, choices=["llm_ic_pddl_planner",
                                                       "llm_pddl_planner",
                                                       "llm_planner",
                                                       "llm_stepbystep_planner",
                                                       "llm_ic_planner",
                                                       "llm_tot_ic_planner"],
                                              default="llm_pddl_planner")
    parser.add_argument('--time-limit', type=int, default=200)
    parser.add_argument('--task', type=int, default=0)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--print-prompts', action='store_true')
    args = parser.parse_args()

    domain = eval(args.domain.capitalize())()

    # 2. initialize the planner
    planner = Planner()

    # 3. execute the llm planner
    method = {
        "llm_ic_pddl_planner"   : llm_ic_pddl_planner,
        "llm_pddl_planner"      : llm_pddl_planner,
        "llm_planner"           : llm_planner,
        "llm_stepbystep_planner": llm_stepbystep_planner,
        "llm_ic_planner"        : llm_ic_planner,
        "llm_tot_ic_planner"       : llm_tot_ic_planner,
    }[args.method]

    if args.print_prompts:
        print_all_prompts(planner)
    else:
        method(args, planner, domain)
