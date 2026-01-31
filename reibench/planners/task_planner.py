import torch
import torch._dynamo
torch._dynamo.config.verbose = True
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModel
from transformers import StoppingCriteria, StoppingCriteriaList
from torch.nn import CrossEntropyLoss
import guidance
from guidance import models, select, gen, user, assistant
import logging
import re
import os
import time
from reibench.utils.config_mapper import (
    get_planner_framework, get_model_name, get_prompting_method, get_model_config
)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def measure_tokens_and_latency(self, prompt, output_text):
    # input tokens
    if self.tokenizer is not None:
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        input_tokens = input_ids.shape[-1]
    else:
        input_tokens = -1  # 超大模型或OpenAI自动处理

    # output tokens
    if self.tokenizer is not None:
        output_ids = self.tokenizer(output_text, return_tensors="pt")["input_ids"].to(self.device)
        output_tokens = output_ids.shape[-1]
    else:
        output_tokens = -1

    return input_tokens, output_tokens, input_tokens + output_tokens

class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_token_id in input_ids[0].tolist()

class TaskPlanner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device2 = torch.device("cuda:1")
        self.max_steps = cfg.planner.max_steps
        
        # Use config mapper for new/old format compatibility
        self.planner_framework = get_planner_framework(cfg)
        model_config = get_model_config(cfg)
        self.model_name = model_config.get('model_name', get_model_name(cfg))
        method_config = get_prompting_method(cfg)
        
        self.scoring_batch_size = cfg.planner.scoring_batch_size
        self.score_function = cfg.planner.score_function
        self.scoring_mode = cfg.planner.scoring_mode
        self.use_predefined_prompt = cfg.planner.use_predefined_prompt
        
        # Get prompting method from new or old format
        self.AP = method_config.get('aware_hint', False)
        self.COT = method_config.get('COT', False)
        self.TOCC = method_config.get('TOCC', False)
        self.TOCC_referring_hint = ""
        self.tokenizer = None

        if self.planner_framework == "saycan" or self.planner_framework == "dag_plan" or self.planner_framework == "hpe_plan":
            print(f"Loading LLM and tokenizer: {self.model_name}")
            model_args = {'pretrained_model_name_or_path': self.model_name, 'trust_remote_code': True,
                        'torch_dtype': torch.float16}
            use_accelerate = model_config.get('use_accelerate_device_map', 
                                             getattr(cfg.planner, 'use_accelerate_device_map', True))
            if use_accelerate:
                model_args['device_map'] = "auto"
                load_in_8bit = model_config.get('load_in_8bit', 
                                               getattr(cfg.planner, 'load_in_8bit', False))
                if load_in_8bit:
                    model_args['load_in_8bit'] = True
            hf_auth_token = model_config.get('hf_auth_token', 
                                            getattr(cfg.planner, 'hf_auth_token', ''))
            model_args['use_auth_token'] = hf_auth_token
            
            if cfg.planner.scoring_mode == 'guidance':
                model_args.pop('pretrained_model_name_or_path')
                if "gpt" in self.model_name:
                    openai_model_name = self.model_name
                    openai_api_key = model_config.get('openai_api_key', 
                                                     getattr(cfg.planner, 'openai_api_key', ''))
                    os.environ["OPENAI_API_KEY"] = openai_api_key
                    self.planner_model = models.OpenAIChat(openai_model_name)
                else: 
                    # tokenizer 
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, force_download=True)
                    # model
                    if "meta-llama" in self.model_name or 'mistralai' in self.model_name or 'deepseek' in self.model_name or "Qwen2.5" in self.model_name:
                        self.planner_model = models.Transformers(
                            self.model_name, 
                            self.tokenizer, 
                            device=self.device, 
                            torch_dtype=torch.float16,
                        )
                    elif 'gemma' in self.model_name:
                        self.planner_model = models.Transformers(self.model_name, self.tokenizer, device=self.device, torch_dtype=torch.bfloat16)

                    print("输出所在设备:", self.planner_model.device if isinstance(self.planner_model, torch.Tensor) else "输出不是 tensor")

                logging.getLogger("guidance").setLevel(logging.WARNING)

            else:
                if "decapoda-research/llama" in self.model_name or "chainyo/alpaca" in self.model_name: 
                    self.model = LlamaForCausalLM.from_pretrained(**model_args)
                    self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name)
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(**model_args)
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

                if not use_accelerate:
                    self.model = self.model.to(self.device)
                self.model.eval()
                self.tokenizer.pad_token_id = 0
                print(f"Loading done\n")

            self.prompt = self.init_prompt(cfg)

    def reset(self, nl_act_list, nl_obj_list):
        self.nl_obj_list = nl_obj_list
        self.skill_set = self.init_skill_set(nl_act_list, nl_obj_list)

    def reset(self):
        self.skill_set = self.init_skill_set()

    def init_prompt(self, cfg):
        raise NotImplementedError()

    def init_skill_set(self, nl_act_list, nl_obj_list):
        raise NotImplementedError()

    def update_skill_set(self, previous_step, nl_obj_list):
        raise NotImplementedError()
    
    
    def count_tokens(self, prompt: str, output_text: str):
        """Count input/output/total tokens for local HF models."""

        if self.tokenizer is None:
            return -1, -1, -1

        # input tokens
        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
            input_tokens = input_ids.shape[-1]
        except:
            input_tokens = -1

        # output tokens
        try:
            output_ids = self.tokenizer(output_text, return_tensors="pt")["input_ids"]
            output_tokens = output_ids.shape[-1]
        except:
            output_tokens = -1

        return input_tokens, output_tokens, input_tokens + output_tokens


    def score(self, prompt, skill_set):
        scores = {}
        batch_skill_set_list = [skill_set[chunk:chunk + self.scoring_batch_size] for chunk in
                                range(0, len(skill_set), self.scoring_batch_size)]
        print("skill_set", skill_set, len(skill_set), type(skill_set), "done")
        torch.cuda.empty_cache()
        if self.scoring_mode == 'guidance':
            if "meta-llama" in self.model_name or 'Qwen2.5' in self.model_name or 'mistralai' in self.model_name or 'gemma' in self.model_name or 'deepseek' in self.model_name:
                out = self.planner_model + prompt + select(skill_set, name='best')
                print('best', out)
                beststep = out['best']
            elif "gpt" in self.model_name:
                out = self.planner_model + prompt + select(skill_set, name='best')
                print('best', out)
                beststep = out['best']
            else:
                out = self.guidance_program(prompt=prompt, candidates=skill_set)
            scores = out['score']
        else:
            assert False, 'unknown scoring mode'
        return beststep

    def plan_whole(self, query, prev_steps=(), prev_msgs=()):
        step_seq = []
        skill_set_size_seq = []
        print(f"Input query: {query}")

        prompt_lines = self.prompt.split('\n')
        prompt_examples = prompt_lines[2:]
        example_text = '\n'.join(prompt_examples)
        skills_text = ', '.join([x.strip() for x in self.skill_set])

        previous_plan = ""
        for i, (step, msg) in enumerate(zip(prev_steps, prev_msgs)):
            if self.use_predefined_prompt and len(msg) > 0:
                previous_plan += step + f' (this action failed: {msg.lower()}), {i + 2}. '
            else:
                previous_plan += step + f', {i + 2}. '

        if len(prev_steps) == 0:
            with user():
                lm = self.planner_model + f"""
                You are a robot operating in a home. A human user can ask you to do various tasks and you are supposed to tell the sequence of actions you would do to accomplish your task.
                Examples of human instructions and possible your (robot) answers:{example_text}

                Now please answer the sequence of actions for the input instruction.
                You should **only** use one of actions of this list: {skills_text}.
                The content in 'Human Previous Inquiry' pertains to previous tasks, and I am not required to complete them. 
                The 'Human Pending Instruction' section contains the instructions I need to follow and complete. 
                List the actions with comma seperator.
                Input user instruction:   
                {query}
                """
        else:
            with user():
                lm = self.planner_model + f"""
                You are a robot operating in a home. A human user can ask you to do various tasks and you are supposed to tell the sequence of actions you would do to accomplish your task.
                Examples of human instructions and possible your (robot) answers:{example_text}

                Now please answer the sequence of actions for the input instruction.
                You should **only** use one of actions of this list: {skills_text}.
                You should **only** use one of actions of the upper list.
                The content in 'Human Previous Inquiry' pertains to previous tasks, and I am not required to complete them. 
                The 'Human Pending Instruction' section contains the instructions I need to follow and complete. 
                Input user instruction:   
                {query}
                Your previous plan was unsuccessful. Here are your plan and the reasons for the failure {previous_plan}.
                List the actions with comma seperator again.
                """
        print("previous_plan=", previous_plan)
        print('skill set=', skills_text)
        with assistant():
            lm += gen("answer", temperature=0)

        answer = lm['answer']
        print('answer:', answer)

        answer = answer.replace('Robot: ', '')
        actions = [action.strip(' 1234567890.') for action in answer.split(',')]
        step_seq = actions

        print('step_seq', step_seq)
        return step_seq, skill_set_size_seq


    def plan_step_by_step(self, query, prev_steps=(), prev_msgs=(), wrong_steps=(), wrong_action_msg=(), reference=None):
        if len(prev_steps) >= self.max_steps:
            return None, None

        prompt = self.prompt + f'{query.strip()}\nRobot: 1. '

        if self.AP == True:
            print("use the aware_hint: ", self.AP)
            prompt = self.prompt + f'Robot: {self.AP}\n' + f'{query.strip()}\nRobot: 1. '
        elif self.TOCC == True:
            if len(prev_steps) == 0: 
                TOCC_hint= TOCC_hint = f"""
                    Human pending instruction may contain vague referring expressions, such as ``electronic devices'', ``beverages'', ``fruits'', and ``containers'', which are not specific items. \n
                    You are a robot, your task is to make the `Human Pending Instruction" clear. \n
                    Do not add extra commentary or conversation or the hole plan, only output the clear instruction. \n
                    Use the previous context below to resolve the referring expressions:\n
                    Previous context:\n
                    {query.strip()}\n
                    Please make the `Human Pending Instruction" clear:"""
                prompt_reference = self.planner_model + f"{TOCC_hint}\n" + gen(stop='.')
                prompt_references = str(prompt_reference).split('\n')
                self.prompt_reference = prompt_references[-1].strip()
                with open("output.txt", "a") as f:
                    f.write(f"{str(self.prompt_reference)}\n")
                prompt = self.prompt + f"Human:{self.prompt_reference}\nRobot: 1. "
            else: 
                prompt = self.prompt + f"Human:{self.prompt_reference}\nRobot: 1. "
        elif self.COT == True:
            if len(prev_steps) == 0: 
                cot_hint = self.prompt + f'{query.strip()}\n' \
                + """Human pending instruction may contain some descriptive referring expressions, such as ``electronic devices",  ``beverages", ``fruits", and ``containers", which cannot be identified as specific items. After identifying these referring expressions, I can determine what the referring expressions of these models are from the previous context.\
                The previous context:
                {query.strip()}\n\
                Please answer:\
                What is the object referred to by the referring expression in the ``Human Pending Instruction"?"""
                prompt_reference = self.planner_model + f"{cot_hint}\n" + gen(stop=".")
                prompt_reference = str(prompt_reference).split('\n')
                self.prompt_reference = prompt_reference[-1] 
                prompt = self.prompt + f'{query.strip()}\n' + f"Hint: {self.prompt_reference}\nRobot: 1. "
            else: 
                prompt = self.prompt + f'{query.strip()}\n' + f"{self.prompt_reference}\nRobot: 1. "

        for i, (step, msg) in enumerate(zip(prev_steps, prev_msgs)):
            if self.use_predefined_prompt and len(msg) > 0:
                prompt += step + f' (this action failed: {msg.lower()}), {i + 2}. '
            else:
                prompt += step + f', {i + 2}. '

        
        best_step = self.score(prompt, self.skill_set)
        best_step = best_step.strip() 
        input_tok, output_tok, total_tok = self.count_tokens(prompt, best_step)
        print("[TOKENS][step_by_step] in:", input_tok, "out:", output_tok, "total:", total_tok)

        return best_step, prompt
    
    def plan_dag(self, dag_query, query, prev_steps=(), prev_msgs=(), wrong_steps=(), wrong_action_msg=(), reference=None):
        if len(prev_steps) >= self.max_steps:
            return None, None

        if len(prev_steps) == 0: 
            pattern = r"\[instruction\]"
            dag_query = re.sub(pattern, query, dag_query, flags=re.IGNORECASE)
            # dag_query = dag_query + "Please ended with \" . \" "
            dag_hint = self.planner_model + dag_query + gen(max_tokens=200)
            dag_hint = str(dag_hint)[len(dag_query):]
            print(dag_hint)
            # import pdb;pdb.set_trace()
            from dag_plan.dag2plan import dag_text_to_plan
            try:
                self.hint = dag_text_to_plan(dag_hint)
            except:
                self.hint = dag_hint
            self.hint = str(self.hint)
        
        prompt = self.prompt + "hint:" + self.hint + f'{query.strip()}\nRobot: 1. '
        # prompt = self.prompt + f'{query.strip()}\nRobot: 1. '

        for i, (step, msg) in enumerate(zip(prev_steps, prev_msgs)):
            if self.use_predefined_prompt and len(msg) > 0:
                prompt += step + f' (this action failed: {msg.lower()}), {i + 2}. '
            else:
                prompt += step + f', {i + 2}. '

        
        best_step = self.score(prompt, self.skill_set)
        best_step = best_step.strip() 
        input_tok, output_tok, total_tok = self.count_tokens(prompt, best_step)
        print("[TOKENS][step_by_step] in:", input_tok, "out:", output_tok, "total:", total_tok)

        return best_step, prompt

    def plan_hpe(self, hpe_query, context, instruction, prev_steps=(), prev_msgs=(), wrong_steps=(), wrong_action_msg=(), reference=None):
        if len(prev_steps) >= self.max_steps:
            return None, None

        if len(prev_steps) == 0: 
            pattern = r"\[instruction\]"
            hpe_query = re.sub(pattern, context, hpe_query, flags=re.IGNORECASE)
            # dag_query = dag_query + "Please ended with \" . \" "
            memory_bank = self.planner_model + hpe_query + gen(max_tokens=100)
            memory_bank = str(memory_bank)[len(hpe_query):]
            print(memory_bank)
            hpe_query2 = "Please extract only the information from the Memory Bank that is most relevant and useful for accomplishing the task.Ignore any unrelated or redundant descriptions. Especially regarding the unclear parts of the instructions\n"
            hpe_query2 =  hpe_query2 + "Memory Bank:" + memory_bank + "Instruction:" + instruction
            hpe_hint = self.planner_model + hpe_query2 + gen(max_tokens=200)
            hpe_hint = str(hpe_hint)[len(hpe_query2):]
            print(hpe_hint)
            # import pdb;pdb.set_trace()
            self.hint = hpe_hint
            self.hint = str(self.hint)
        
        prompt = self.prompt + "hint:" + self.hint + f'{instruction.strip()}\nRobot: 1. '
        # prompt = self.prompt + f'{query.strip()}\nRobot: 1. '

        for i, (step, msg) in enumerate(zip(prev_steps, prev_msgs)):
            if self.use_predefined_prompt and len(msg) > 0:
                prompt += step + f' (this action failed: {msg.lower()}), {i + 2}. '
            else:
                prompt += step + f', {i + 2}. '

        
        best_step = self.score(prompt, self.skill_set)
        best_step = best_step.strip() 
        input_tok, output_tok, total_tok = self.count_tokens(prompt, best_step)
        print("[TOKENS][step_by_step] in:", input_tok, "out:", output_tok, "total:", total_tok)

        return best_step, prompt

    def duplicate_past_key_values(self, past_key_values, batch_size):
        batch_past_key_values = []
        for layer in range(len(past_key_values)):
            batch_past_key_values_layer = []
            for kv in range(len(past_key_values[layer])):
                batch_past_key_values_layer.append(past_key_values[layer][kv].repeat(batch_size, 1, 1, 1))
            batch_past_key_values_layer = tuple(batch_past_key_values_layer)
            batch_past_key_values.append(batch_past_key_values_layer)
        batch_past_key_values = tuple(batch_past_key_values)
        return batch_past_key_values
