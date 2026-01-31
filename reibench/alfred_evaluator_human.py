
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
from reibench.utils.config_mapper import get_prompting_method
import re
from reibench.envs.alfred.utils import ithor_name_to_natural_word, find_indefinite_article

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
        self.split = self.cfg.split
        self.skillset = self.init_skill_set()
        with open("skillset.txt", "w", encoding="utf-8") as f:
            for item in self.skillset:
                f.write(str(item) + "\n")

    def evaluate(self):
        cfg = self.cfg

        log.info(OmegaConf.to_yaml(cfg))
        global splits

        args_dict = {'data': 'alfred/data/json_2.1.0 copy', 'pframe': 300, 'fast_epoch': False,
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


        start = time.time()
        x_display = cfg.alfred.x_display
        save_path = cfg.out_dir
        results = self.evaluate_main(files, args_dict, x_display, save_path)

        log.info(results)
        n = len(results)
        n_success = 0
        for e in results:
            if e['success']:
                n_success += 1
                log.info(f'{e}')
        log.info(f'success rate: {n_success / n * 100:.2f} % ({n_success}/{n})')
        log.info(f'elapsed: {str(datetime.timedelta(seconds=(time.time() - start)))}')
        log.info('------------------------')
        log.info(OmegaConf.to_yaml(cfg))  


    def evaluate_main(self, tasks, args_dict, x_display, save_path):
        results = []
        model_args = dotdict(args_dict)
        env = ThorConnector(x_display=x_display)

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
                result = self.evaluate_task_human(env, traj_data, r_idx, model_args, save_path, self.skillset, log_prompt=(i==0), train_gt_steps=train_gt_steps)
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
        data_type = self.cfg.data_type
        instruction_text = traj_data['turk_annotations']['anns'][r_idx][f'task_desc{data_type}']
        memory = traj_data['turk_annotations']['anns'][r_idx][f'memory{data_type}']

        memory_instruction_text = ""
        for line in memory:
            memory_instruction_text = memory_instruction_text + "\n" + "Human previous inquiry(Not Required to Execute):" + line
        memory_instruction_text = memory_instruction_text + "\n" + "Human pending instruction:" + instruction_text.replace("Human: ", "", 1)
        
        log.info("Task: %s \n" % "Human pending instruction:" + instruction_text.replace("Human: ", "", 1))
        
        return memory_instruction_text
    
    def init_skill_set(self):
        alfred_objs = ['Cart', 'Potato', 'Faucet', 'Ottoman', 'CoffeeMachine', 'Candle', 'CD', 'Pan', 'Watch',
                    'HandTowel', 'SprayBottle', 'BaseballBat', 'CellPhone', 'Kettle', 'Mug', 'StoveBurner', 'Bowl',
                    'Toilet', 'DiningTable', 'Spoon', 'TissueBox', 'Shelf', 'Apple', 'TennisRacket', 'SoapBar',
                    'Cloth', 'Plunger', 'FloorLamp', 'ToiletPaperHanger', 'CoffeeTable', 'Spatula', 'Plate', 'Bed',
                    'Glassbottle', 'Knife', 'Tomato', 'ButterKnife', 'Dresser', 'Microwave', 'CounterTop',
                    'GarbageCan', 'WateringCan', 'Vase', 'ArmChair', 'Safe', 'KeyChain', 'Pot', 'Pen', 'Cabinet',
                    'Desk', 'Newspaper', 'Drawer', 'Sofa', 'Bread', 'Book', 'Lettuce', 'CreditCard', 'AlarmClock',
                    'ToiletPaper', 'SideTable', 'Fork', 'Box', 'Egg', 'DeskLamp', 'Ladle', 'WineBottle', 'Pencil',
                    'Laptop', 'RemoteControl', 'BasketBall', 'DishSponge', 'Cup', 'SaltShaker', 'PepperShaker',
                    'Pillow', 'Bathtub', 'SoapBottle', 'Statue', 'Fridge', 'Sink']
        alfred_pick_obj = ['KeyChain', 'Potato', 'Pot', 'Pen', 'Candle', 'CD', 'Pan', 'Watch', 'Newspaper', 'HandTowel',
                        'SprayBottle', 'BaseballBat', 'Bread', 'CellPhone', 'Book', 'Lettuce', 'CreditCard', 'Mug',
                        'AlarmClock', 'Kettle', 'ToiletPaper', 'Bowl', 'Fork', 'Box', 'Egg', 'Spoon', 'TissueBox',
                        'Apple', 'TennisRacket', 'Ladle', 'WineBottle', 'Cloth', 'Plunger', 'SoapBar', 'Pencil',
                        'Laptop', 'RemoteControl', 'BasketBall', 'DishSponge', 'Cup', 'Spatula', 'SaltShaker',
                        'Plate', 'PepperShaker', 'Pillow', 'Glassbottle', 'SoapBottle', 'Knife', 'Statue', 'Tomato',
                        'ButterKnife', 'WateringCan', 'Vase']
        alfred_open_obj = ['Safe', 'Laptop', 'Fridge', 'Box', 'Microwave', 'Cabinet', 'Drawer']
        alfred_slice_obj = ['Potato', 'Lettuce', 'Tomato', 'Apple', 'Bread']
        alfred_toggle_obj = ['Microwave', 'DeskLamp', 'FloorLamp', 'Faucet']
        alfred_recep = ['ArmChair', 'Safe', 'Cart', 'Ottoman', 'Pot', 'CoffeeMachine', 'Desk', 'Cabinet', 'Pan',
                        'Drawer', 'Sofa', 'Mug', 'StoveBurner', 'SideTable', 'Toilet', 'Bowl', 'Box', 'DiningTable',
                        'Shelf', 'ToiletPaperHanger', 'CoffeeTable', 'Cup', 'Plate', 'Bathtub', 'Bed', 'Dresser',
                        'Fridge', 'Microwave', 'CounterTop', 'Sink', 'GarbageCan']
        alfred_objs = [ithor_name_to_natural_word(w) for w in alfred_objs]
        alfred_pick_obj = [ithor_name_to_natural_word(w) for w in alfred_pick_obj]
        alfred_open_obj = [ithor_name_to_natural_word(w) for w in alfred_open_obj]
        alfred_slice_obj = [ithor_name_to_natural_word(w) for w in alfred_slice_obj]
        alfred_toggle_obj = [ithor_name_to_natural_word(w) for w in alfred_toggle_obj]
        alfred_recep = [ithor_name_to_natural_word(w) for w in alfred_recep]

        skills = ['done']

        for o in (alfred_pick_obj + alfred_slice_obj + alfred_open_obj + alfred_toggle_obj + alfred_recep):
            article = find_indefinite_article(o)
            skills.append(f'find {article} {o}')

        for o in alfred_pick_obj:
            skills.append(f'pick up the {o}')
            skills.append(f'put down the {o}')

        for o in alfred_open_obj:
            skills.append(f'open the {o}')
            skills.append(f'close the {o}')

        for o in alfred_slice_obj:
            skills.append(f'slice the {o}')

        for o in alfred_toggle_obj:
            skills.append(f'turn on the {o}')
            skills.append(f'turn off the {o}')

        skills = [' ' + c for c in skills]

        log.info(f'# of skills: {len(skills)}')
        log.info(skills)

        return skills

    def evaluate_task_human(self, env, traj_data, r_idx, model_args, save_path, skillset, log_prompt=False, train_gt_steps=None):
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
        instruction_text = self.instruction_organizing(traj_data, r_idx)
        referring_expression = traj_data['turk_annotations']['anns'][r_idx]['reference']

        original_instruction = traj_data['turk_annotations']['anns'][r_idx]['task_desc']
        
        done, success = False, False
        t = 0
        reward = 0
        imgs = [Image.fromarray(env.last_event.frame)]

        goal_satisfied = False
        prev_steps = []
        prev_action_msg = []
        step_num = 0
        while not done:
            for i, skill in enumerate(skillset):
                print("section", i, ":", skill)
            print("the instruction is:", instruction_text)
            print("the previous steps is", prev_steps)
            print("the previous action message is", prev_action_msg)
            step_id = input("Please enter your the step section: ")
            step = skillset[int(step_id)]

            log.info(f'{len(prev_steps) + 1}. {step}')
            prev_steps.append(step)
            step = step[1:]
            if step in ['done', 'done.', 'done.\n']:
                done = True
                prev_action_msg.append('')
                break

            step_to_execute = step
            step_num += 1
            try:
                action_ret = env.llm_skill_interact(step_to_execute)
                print(action_ret)
                prev_action_msg.append(action_ret['message'])
                if not action_ret['success']:
                    print(action_ret['message'])
            except Exception as e:
                log.warning(e)
                prev_action_msg.append(e)

            t_reward, t_done = env.get_transition_reward()
            reward += t_reward
            t += 1
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
