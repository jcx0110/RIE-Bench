import math
import os, json
import pprint
import random
import textwrap
import time, datetime
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '')
sys.path.insert(0, './alfred')

from PIL import Image, ImageDraw, ImageFont

from alfred.data.preprocess import Dataset
from reibench.planners.alfred_planner import AlfredTaskPlanner
from reibench.envs.alfred.thor_connector import ThorConnector
from reibench.envs.alfred.utils import dotdict, load_task_json
from tqdm import tqdm

import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from reibench.evaluator import Evaluator

from reibench.planners.llm_p_planner import llmp_planner, Planner
from reibench.utils.config_mapper import (
    get_planner_framework, get_data_type, get_data_types, get_prompting_method
)
import re
import time

font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf", 24)
log = logging.getLogger(__name__)
log_success = logging.getLogger(f"{__name__}_success")

class AlfredEvaluator(Evaluator):
    def __init__(self, hparams):
        self.cfg = hparams
        
        # Use config mapper for new/old format compatibility
        method_config = get_prompting_method(hparams)
        self.COT = method_config.get('COT', False)
        self.TOCC = method_config.get('TOCC', False)
        self.planner_framework = get_planner_framework(hparams)
        self.split = self.cfg.split

    def evaluate(self):
        cfg = self.cfg

        log.info(OmegaConf.to_yaml(cfg))
        global splits

        if self.planner_framework == "saycan" or self.planner_framework == "dag_plan" or self.planner_framework == "hpe_plan":
            if len(cfg.planner.model_name) > 0:
                self.planner = AlfredTaskPlanner(cfg)
                self.planner.reset() 
            else:
                self.planner = None
        elif self.planner_framework == "LLM+P":
            planner = AlfredTaskPlanner(cfg)
            self.skillset = planner.init_skill_set()
            prompt = planner.init_prompt(cfg)
            prompt_lines = prompt.split('\n')
            prompt_examples = prompt_lines[2:]
            self.example_text = '\n'.join(prompt_examples)
            self.planner = Planner()

        args_dict = {'data': 'data/raw/alfred/json_2.1.0 copy', 'pframe': 300, 'fast_epoch': False,
                    'use_templated_goals': False, 'dout': 'exp/model', 'pp_folder': 'pp',
                    'reward_config': 'alfred/models/config/rewards.json', 'max_steps': 1000}
        
        splits = self.split

        with open(splits) as f:
            splits = json.load(f)
            pprint.pprint({k: len(v) for k, v in splits.items()})

        number_of_dirs = len(list(os.listdir(args_dict['data'])))
        do_preprocessing = number_of_dirs < 50 
        if do_preprocessing:
            log.info("\nPreprocessing dataset... Do this once as required:")
            vocab = None
            dataset = Dataset(dotdict(args_dict), vocab)
            dataset.preprocess_splits(splits)

        print("loading eval_set:", cfg.alfred.eval_set)
        print("usable splits:", splits.keys())
        assert cfg.alfred.eval_set in splits.keys()
        files = []

        for e in splits[cfg.alfred.eval_set]:
            if 'pick_two_obj_and_place' not in e['task']:
                files.append(e)

        if False: 
            for file in files:
                if 'trial_T20190907_012527_434405' in file['task']:
                    new_files = [file]
                    break
            files = new_files

        start = time.time()
        x_display = cfg.alfred.x_display
        save_path = cfg.out_dir
        results = self.evaluate_main(files, args_dict, self.planner, x_display, save_path)

        log.info(results)
        n = len(results)
        n_success = 0
        for e in results:
            if e['success']:
                n_success += 1
                log.info(f'{e}')
        # 计算平均 whole-plan latency
        latencies = [e['whole_plan_latency_ms'] for e in results if 'whole_plan_latency_ms' in e]
        if len(latencies) > 0:
            avg_latency = sum(latencies) / len(latencies)
            log.info(f'Average whole-plan latency: {avg_latency:.2f} ms')
        else:
            log.info('No latency data recorded.')
        log.info(f'success rate: {n_success / n * 100:.2f} % ({n_success}/{n})')
        log.info(f'elapsed: {str(datetime.timedelta(seconds=(time.time() - start)))}')
        log.info('------------------------')
        log.info(OmegaConf.to_yaml(cfg))  

    def evaluate_main(self, tasks, args_dict, planner, x_display, save_path):
        results = []
        model_args = dotdict(args_dict)
        env = ThorConnector(x_display=x_display)

        if planner is not None:
            with open(os.path.join(save_path, 'prompt.txt'), 'w') as f:
                f.write(planner.prompt)

        train_gt_steps = None
        if self.cfg.alfred.eval_set == 'train':

            train_gt_steps = {}
            with open(self.cfg.prompt.example_file_path, 'r') as f:
                train_examples = json.load(f)
            for ex in train_examples:
                train_gt_steps[ex['task id']] = ex['NL steps']

        for i, task in enumerate(tqdm(tasks)):
            log.info(f"repeat_idx:{task['repeat_idx']} task:{task['task']}")
            traj_data = load_task_json(task, args_dict["data"])
            r_idx = task['repeat_idx']
            try:
                log.info(f"repeat_idx:{task['repeat_idx']} task:{task['task']}")
                traj_data = load_task_json(task, args_dict["data"])
                r_idx = task['repeat_idx']
                log.info(f"Evaluating ({i+1}/{len(tasks)}): {traj_data['root']}")
                if self.planner_framework == "saycan":
                    result = self.evaluate_task_saycan(env, traj_data, r_idx, model_args, planner, save_path, log_prompt=(i==0), train_gt_steps=train_gt_steps)
                elif self.planner_framework == "LLM+P":
                    result = self.evaluate_task_pddl(traj_data, r_idx, save_path, self.skillset, env, model_args)
                elif self.planner_framework == "dag_plan":
                    result = self.evaluate_task_dag_plan(env, traj_data, r_idx, model_args, planner, save_path, log_prompt=(i==0), train_gt_steps=train_gt_steps)
                elif self.planner_framework == "hpe_plan":
                    result = self.evaluate_task_hpe_plan(env, traj_data, r_idx, model_args, planner, save_path, log_prompt=(i==0), train_gt_steps=train_gt_steps)
                results.append(result)
                if result['success']:
                    log_success.info(task)
                    log_success.info(f"Evaluating ({i+1}/{len(tasks)}): {traj_data['root']}")

            except Exception as e:
                import traceback
                traceback.print_exc()
                log.info("Error: " + repr(e))

        return results

    def instruction_organizing(self, traj_data, r_idx):
        # Use config mapper to get data_type from new or old format
        data_type = get_data_type(self.cfg)
        instruction_text = traj_data['turk_annotations']['anns'][r_idx][f'task_desc{data_type}']
        memory = traj_data['turk_annotations']['anns'][r_idx][f'memory{data_type}']

        memory_instruction_text = ""
        for line in memory:
            memory_instruction_text = memory_instruction_text + "\n" + "Human previous inquiry(Not Required to Execute):" + line
        memory_instruction_text = memory_instruction_text + "\n" + "Human pending instruction:" + instruction_text.replace("Human: ", "", 1)
        
        log.info("Task: %s \n" % "Human pending instruction:" + memory_instruction_text.replace("Human: ", "", 1))
        
        return memory_instruction_text, instruction_text.replace("Human: ", "", 1)

    def evaluate_task_pddl(self, traj_data, r_idx, save_path, skillset, env, model_args):
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        env.step(dict(traj_data['scene']['init_action']))
        env.set_task(traj_data, model_args, reward_type='dense')

        print("right")
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        ground_truth = traj_data['turk_annotations']['anns'][r_idx][f'high_descs']
        clear_instruction = traj_data['turk_annotations']['anns'][r_idx][f'task_desc']
        instruction_text, _ = self.instruction_organizing(traj_data, r_idx)

        success, plan = llmp_planner(instruction_text, self.planner, skillset, self.example_text)

        success = False

        steps = [re.sub(r'^\d+\.\s*', '', line).strip() for line in plan.split(', ') if line.strip()]

        for si, step in enumerate(steps):
            step_to_execute = step
            try:
                action_ret = env.llm_skill_interact(step_to_execute)
            except Exception as e:
                log.warning(e)
        
        goal_satisfied = env.get_goal_satisfied()
        log.info('ground truth: ' + str(ground_truth))
        log.info('target goal: ' + json.dumps(env.task.get_targets()))
        if goal_satisfied:
            print("Goal Reached")
            success = True


        log.info('success: ' + str(success))

        result = {'trial': traj_data['task_id'],
            'type': traj_data['task_type'],
            'repeat_idx': int(r_idx),
            'goal_instr': instruction_text,
            'plan': steps,
            'success': success}
        return result
    
    def evaluate_task_dag_plan(self, env, traj_data, r_idx, model_args, planner, save_path, log_prompt=False, train_gt_steps=None):
        with open("prompts/dag_prompt.txt", "r", encoding="utf-8") as f:
            dag_query = f.read()

        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        env.step(dict(traj_data['scene']['init_action']))
        env.set_task(traj_data, model_args, reward_type='dense')

        ground_truth = traj_data['turk_annotations']['anns'][r_idx][f'high_descs']
        clear_instruction = traj_data['turk_annotations']['anns'][r_idx][f'task_desc']
        instruction_text, _ = self.instruction_organizing(traj_data, r_idx)
        referring_expression = traj_data['turk_annotations']['anns'][r_idx]['reference']

        original_instruction = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        
        done, success = False, False
        t = 0
        reward = 0
        imgs = [Image.fromarray(env.last_event.frame)]

        if self.cfg.planner.model_name.endswith('gpt-3.5-turbo') or 'gpt-4' in self.cfg.planner.model_name:
            step_by_step_mode = False
        else:
            step_by_step_mode = True

        if step_by_step_mode:
            goal_satisfied = False
            prev_steps = []
            prev_action_msg = []
            step_num = 0
            start = time.perf_counter()
            while not done:
                if self.cfg.alfred.eval_set == 'train' and train_gt_steps is not None:
                    if t >= len(train_gt_steps[traj_data['task_id']]):
                        step = 'done'
                    else:
                        step = train_gt_steps[traj_data['task_id']][t]
                    prompt = ''
                else:
                    step, prompt = planner.plan_dag(dag_query, instruction_text, prev_steps, prev_action_msg, referring_expression)
                    if step is None:
                        log.info("\tmax step reached")
                        break

                if log_prompt:
                    log.info(prompt)
                log.info(f'{len(prev_steps) + 1}. {step}')
                prev_steps.append(step)

                if step in ['done', 'done.', 'done.\n']:
                    done = True
                    prev_action_msg.append('')
                    break

                step_to_executes = ["find a fridge", "open the fridge", "find a bread", "pick up the bread", "find a lettuce", 
                                    "pick up the lettuce", "put down the bread", "pick up the lettuce", "pick up the bread", "done"]

                step_to_execute = step
                # step_to_execute = step_to_executes[step_num]

                if step_to_execute in ['done', 'done.', 'done.\n']:
                    done = True
                    prev_action_msg.append('')
                    break
                step_num += 1
                try:
                    action_ret = env.llm_skill_interact(step_to_execute)
                    print(action_ret)
                    prev_action_msg.append(action_ret['message'])
                except Exception as e:
                    log.warning(e)
                imgs.append(env.write_step_on_img(self.cfg.planner.use_predefined_prompt, t+1, action_ret))

                if not action_ret['success']:
                    print(action_ret['message'])

                t_reward, t_done = env.get_transition_reward()
                reward += t_reward
                t += 1
            end = time.perf_counter()
            whole_latency_ms = (end - start) * 1000
            log.info(f"[Latency] Whole-plan generation time: {whole_latency_ms:.2f} ms")

        goal_satisfied = env.get_goal_satisfied()
        log.info('success: ' + str(goal_satisfied))
        log.info('ground truth: ' + str(ground_truth))
        log.info('target goal: ' + json.dumps(env.task.get_targets()))
        log.info('success: ' + str(goal_satisfied))
        if goal_satisfied:
            success = True

        log_entry = {'trial': traj_data['task_id'],
                    'scene': scene_name,
                    'type': traj_data['task_type'],
                    'repeat_idx': int(r_idx),
                    'goal_instr': instruction_text,
                    'inferred_steps': prev_steps,
                    'whole_plan_latency_ms': whole_latency_ms,
                    'success': success}

        # save img
        # self.save_result(log_entry, imgs, save_path)

        return log_entry
    
    def evaluate_task_hpe_plan(self, env, traj_data, r_idx, model_args, planner, save_path, log_prompt=False, train_gt_steps=None):
        with open("prompts/hpe_prompt.txt", "r", encoding="utf-8") as f:
            hpe_query = f.read()

        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        env.step(dict(traj_data['scene']['init_action']))
        env.set_task(traj_data, model_args, reward_type='dense')

        ground_truth = traj_data['turk_annotations']['anns'][r_idx][f'high_descs']
        clear_instruction = traj_data['turk_annotations']['anns'][r_idx][f'task_desc']
        instruction_text, last_instruction = self.instruction_organizing(traj_data, r_idx)
        referring_expression = traj_data['turk_annotations']['anns'][r_idx]['reference']
        
        done, success = False, False
        t = 0
        reward = 0
        imgs = [Image.fromarray(env.last_event.frame)]

        if self.cfg.planner.model_name.endswith('gpt-3.5-turbo') or 'gpt-4' in self.cfg.planner.model_name:
            step_by_step_mode = False
        else:
            step_by_step_mode = True

        if step_by_step_mode:
            goal_satisfied = False
            prev_steps = []
            prev_action_msg = []
            step_num = 0
            start = time.perf_counter()
            while not done:
                if self.cfg.alfred.eval_set == 'train' and train_gt_steps is not None:
                    if t >= len(train_gt_steps[traj_data['task_id']]):
                        step = 'done'
                    else:
                        step = train_gt_steps[traj_data['task_id']][t]
                    prompt = ''
                else:
                    step, prompt = planner.plan_hpe(hpe_query, instruction_text, last_instruction, prev_steps, prev_action_msg, referring_expression)
                    if step is None:
                        log.info("\tmax step reached")
                        break

                if log_prompt:
                    log.info(prompt)
                log.info(f'{len(prev_steps) + 1}. {step}')
                prev_steps.append(step)

                if step in ['done', 'done.', 'done.\n']:
                    done = True
                    prev_action_msg.append('')
                    break

                step_to_executes = ["find a fridge", "open the fridge", "find a bread", "pick up the bread", "find a lettuce", 
                                    "pick up the lettuce", "put down the bread", "pick up the lettuce", "pick up the bread", "done"]

                step_to_execute = step
                # step_to_execute = step_to_executes[step_num]

                if step_to_execute in ['done', 'done.', 'done.\n']:
                    done = True
                    prev_action_msg.append('')
                    break
                step_num += 1
                try:
                    action_ret = env.llm_skill_interact(step_to_execute)
                    print(action_ret)
                    prev_action_msg.append(action_ret['message'])
                except Exception as e:
                    log.warning(e)
                imgs.append(env.write_step_on_img(self.cfg.planner.use_predefined_prompt, t+1, action_ret))

                if not action_ret['success']:
                    print(action_ret['message'])

                t_reward, t_done = env.get_transition_reward()
                reward += t_reward
                t += 1
            end = time.perf_counter()
            whole_latency_ms = (end - start) * 1000
            log.info(f"[Latency] Whole-plan generation time: {whole_latency_ms:.2f} ms")

        goal_satisfied = env.get_goal_satisfied()
        log.info('success: ' + str(goal_satisfied))
        log.info('ground truth: ' + str(ground_truth))
        log.info('target goal: ' + json.dumps(env.task.get_targets()))
        log.info('success: ' + str(goal_satisfied))
        if goal_satisfied:
            success = True

        log_entry = {'trial': traj_data['task_id'],
                    'scene': scene_name,
                    'type': traj_data['task_type'],
                    'repeat_idx': int(r_idx),
                    'goal_instr': instruction_text,
                    'inferred_steps': prev_steps,
                    'whole_plan_latency_ms': whole_latency_ms,
                    'success': success}

        # save img
        # self.save_result(log_entry, imgs, save_path)

        return log_entry

    def evaluate_task_saycan(self, env, traj_data, r_idx, model_args, planner, save_path, log_prompt=False, train_gt_steps=None):
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        env.step(dict(traj_data['scene']['init_action']))
        env.set_task(traj_data, model_args, reward_type='dense')

        ground_truth = traj_data['turk_annotations']['anns'][r_idx][f'high_descs']
        clear_instruction = traj_data['turk_annotations']['anns'][r_idx][f'task_desc']
        instruction_text, _ = self.instruction_organizing(traj_data, r_idx)
        referring_expression = traj_data['turk_annotations']['anns'][r_idx]['reference']

        original_instruction = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        
        done, success = False, False
        t = 0
        reward = 0
        imgs = [Image.fromarray(env.last_event.frame)]

        if self.cfg.planner.model_name.endswith('gpt-3.5-turbo') or 'gpt-4' in self.cfg.planner.model_name:
            step_by_step_mode = False
        else:
            step_by_step_mode = True

        if step_by_step_mode:
            goal_satisfied = False
            prev_steps = []
            prev_action_msg = []
            step_num = 0
            start = time.perf_counter()
            while not done:
                if self.cfg.alfred.eval_set == 'train' and train_gt_steps is not None:
                    if t >= len(train_gt_steps[traj_data['task_id']]):
                        step = 'done'
                    else:
                        step = train_gt_steps[traj_data['task_id']][t]
                    prompt = ''
                else:
                    step, prompt = planner.plan_step_by_step(instruction_text, prev_steps, prev_action_msg, referring_expression)
                    if step is None:
                        log.info("\tmax step reached")
                        break

                if log_prompt:
                    log.info(prompt)
                log.info(f'{len(prev_steps) + 1}. {step}')
                prev_steps.append(step)

                if step in ['done', 'done.', 'done.\n']:
                    done = True
                    prev_action_msg.append('')
                    break

                step_to_executes = ["find a fridge", "open the fridge", "find a bread", "pick up the bread", "find a lettuce", 
                                    "pick up the lettuce", "put down the bread", "pick up the lettuce", "pick up the bread", "done"]

                step_to_execute = step
                # step_to_execute = step_to_executes[step_num]

                if step_to_execute in ['done', 'done.', 'done.\n']:
                    done = True
                    prev_action_msg.append('')
                    break
                step_num += 1
                try:
                    action_ret = env.llm_skill_interact(step_to_execute)
                    print(action_ret)
                    prev_action_msg.append(action_ret['message'])
                except Exception as e:
                    log.warning(e)
                imgs.append(env.write_step_on_img(self.cfg.planner.use_predefined_prompt, t+1, action_ret))

                if not action_ret['success']:
                    print(action_ret['message'])

                t_reward, t_done = env.get_transition_reward()
                reward += t_reward
                t += 1
            end = time.perf_counter()
            whole_latency_ms = (end - start) * 1000
            log.info(f"[Latency] Whole-plan generation time: {whole_latency_ms:.2f} ms")
        else:
            replan_times = 0
            goal_satisfied = False
            prev_steps = []
            prev_action_msg = []
            while not goal_satisfied and replan_times < 3 and goal_satisfied != "True":
                steps, prompt = planner.plan_whole(instruction_text, prev_steps, prev_action_msg)
                prev_steps = steps

                if log_prompt:
                    log.info(prompt)

                for si, step in enumerate(steps):
                    log.info(f'{si + 1}. {step}')

                    if step in ['done', 'done.', 'done.\n']:
                        done = True
                        break

                    step_to_execute = step
                    try:
                        action_ret = env.llm_skill_interact(step_to_execute)
                    except Exception as e:
                        log.warning(e)
                    imgs.append(env.write_step_on_img(self.cfg.planner.use_predefined_prompt, t + 1, action_ret))
                    prev_action_msg.append(action_ret['message'])
                    if not action_ret['success']:
                        print(action_ret['message'])

                    t_reward, t_done = env.get_transition_reward()
                    reward += t_reward
                    t += 1
                goal_satisfied = env.get_goal_satisfied()
                log.info('success: ' + str(goal_satisfied))
                replan_times += 1
        
        if step_by_step_mode and not 'DeepSeek-R1-Distill' in self.cfg.planner.model_name:
            goal_satisfied = env.get_goal_satisfied()

        goal_satisfied = env.get_goal_satisfied()
        log.info('success: ' + str(goal_satisfied))
        log.info('ground truth: ' + str(ground_truth))
        log.info('target goal: ' + json.dumps(env.task.get_targets()))
        log.info('success: ' + str(goal_satisfied))
        if goal_satisfied:
            success = True

        log_entry = {'trial': traj_data['task_id'],
                    'scene': scene_name,
                    'type': traj_data['task_type'],
                    'repeat_idx': int(r_idx),
                    'goal_instr': instruction_text,
                    'inferred_steps': prev_steps,
                    'whole_plan_latency_ms': whole_latency_ms,
                    'success': success}

        # save img
        # self.save_result(log_entry, imgs, save_path)

        return log_entry

    def save_result(self, result_dict, imgs, base_path='results'):
        if result_dict:
            filename = f"{result_dict['trial']}_{result_dict['repeat_idx']}"

            with open(os.path.join(base_path, filename + '.json'), "w") as outfile:
                json.dump(result_dict, outfile)
        else:
            filename = "images"

        widths, heights = zip(*(i.size for i in imgs))
        total_width = widths[0] * 5
        textbox_height = 70 
        total_height = math.ceil(len(imgs) / 5) * heights[0] + textbox_height
        new_im = Image.new('RGB', (total_width, total_height), color='white')

        if result_dict:
            text = 'Instruction: ' + result_dict['goal_instr']
            text_color = (0, 0, 0)  
            lines = textwrap.wrap(text, width=110)
            draw = ImageDraw.Draw(new_im)
            y_start = 10 if len(lines) > 1 else 35
            draw.multiline_text((10, y_start), '\n'.join(lines), font=font, fill=text_color)
            y_offset = textbox_height
        else:
            y_offset = 0

        x_offset = 0
        for im in imgs:
            new_im.paste(im, (x_offset, y_offset))
            x_offset += im.size[0]
            if x_offset >= total_width:
                x_offset = 0
                y_offset += im.size[1]

        success_str = 'success' if result_dict['success'] else 'fail'
        new_im.save(os.path.join(base_path, f"{filename}_{success_str}_{self.cfg.data_type}.png"))
