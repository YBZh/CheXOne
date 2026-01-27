import asyncio
import os
import re
import textwrap
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Union

import json
import torch

from swift.llm import PtEngine, RequestConfig, RolloutInferRequest, Template, to_device
from swift.llm.infer.protocol import ChatCompletionResponse, ChatCompletionResponseChoice
from swift.plugin import ORM, orms, rm_plugins
# register context manager(used in gym training)
from swift.plugin.context_manager import ContextManager, context_managers
from swift.plugin.env import Env, envs
from swift.plugin.multi_turn import MultiTurnScheduler, multi_turns
from swift.plugin.rm_plugin import DefaultRMPlugin
from swift.utils import get_logger
from radcliq import CompositeMetric

logger = get_logger()
"""
TO CUSTOMIZE REWARD FUNCTION:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the model's output completions and dataset columns (passed as kwargs) as input parameters.

    Step 2: Add your reward function to the orms registry:
        orms['my_reward_function'] = MyRewardFunction

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_funcs my_reward_function
"""


import torch.nn as nn
from bert_score import BERTScorer


class BertScore(nn.Module):
    def __init__(self,
                 model_type='distilbert-base-uncased',
                 num_layers=5,
                 rescale_with_baseline=True,
                 idf=False,
                 ):
        super(BertScore, self).__init__()

        with torch.no_grad():
            self.bert_scorer = BERTScorer(model_type=model_type,
                                        num_layers=num_layers,
                                        batch_size=64,
                                        nthreads=4,
                                        all_layers=False,
                                        idf=idf,
                                        device=None,
                                        lang='en',
                                        rescale_with_baseline=rescale_with_baseline,
                                        baseline_path=None)

    def forward(self, refs, hyps):
        with torch.no_grad():
            p, r, f = self.bert_scorer.score(
            cands=hyps,
            refs=refs,
            verbose=False,
            batch_size=64,
            )
            return torch.mean(f).item(), f.tolist()

# bertscore_scorer = BertScore(model_type='distilbert-base-uncased', num_layers=5)
# print("bertscore_scorer init")
# print(list(bertscore_scorer.bert_scorer._model.named_parameters()))

# class VQA_GenerationORM(ORM):

#     def __init__(self):
#         # from transformers.modeling_utils import set_zero3_state
#         # with set_zero3_state():
#         #     self.bertscore_scorer = BertScore(model_type='distilbert-base-uncased', num_layers=5)
#         # for param in self.bertscore_scorer.bert_scorer._model.parameters():
#         #     param.requires_grad = False

#         self.radcliq_scorer = CompositeMetric()
#         # 自动适配多卡环境，将bertscore_scorer移动到与ground_truth相同的device
#         # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # self.bertscore_scorer = self.bertscore_scorer.to(device)
#         # self.bertscore_scorer = BertScore(model_type='distilbert-base-uncased', num_layers=5)
#         # print("bertscore_scorer init")
#         # self.bertscore_scorer = bertscore_scorer
#         # print(self.bertscore_scorer.bert_scorer._model.embeddings.word_embeddings.weight)
#         # ipdb.set_trace()

#     def extract_boxed_result(self, text):
#         """
#         提取 \boxed{} 前面的内容和 \boxed{} 内部的内容。
#         例如: "reason\\boxed{answer}" -> ("reason", "answer")
#         如果只有前半部分或者只有后半部分，则只取出对应部分，没有的部分用None占位

#         注意：本函数假设text为原始字符串，包含转义字符（如\\n），
#         并且\boxed{...}前面可能有多行内容。
#         """
#         # 先尝试匹配最后一个 \boxed{...}，并提取其前面的内容
#         # 支持多行内容
#         pattern = r'(?s)(.*?)\\boxed{([^}]*)}'
#         match = re.search(pattern, text)
#         if match:
#             before_boxed = match.group(1)
#             boxed_content = match.group(2)
#             before_boxed = before_boxed.strip() if before_boxed and before_boxed.strip() else None
#             boxed_content = boxed_content.strip() if boxed_content and boxed_content.strip() else None
#             return before_boxed, boxed_content
#         else:
#             # 检查是否有 \boxed{...} 但前面没有内容
#             pattern_boxed_only = r'\\boxed{([^}]*)}'
#             match_boxed_only = re.search(pattern_boxed_only, text)
#             if match_boxed_only:
#                 boxed_content = match_boxed_only.group(1).strip() if match_boxed_only.group(1).strip() else None
#                 return None, boxed_content
#             # 检查是否有前半部分但没有 \boxed{}
#             text_clean = text.strip() if text and text.strip() else None
#             return text_clean, None

#     def extract_label_option_multiple(self, label):
#         """
#         提取label中的所有选项，支持多选和多种格式。
#         输入: label字符串，如 "A, B", "(A), (B)", "(A) XX, (B) XX", "(A). XX, (B). XX", "(A): XX, (B): XX", "A XX, B XX"
#         输出: 选项列表，如 ["A"], ["A", "B"]
#         """
#         # 1. 先找所有 (A) 这种括号包裹的
#         options = re.findall(r'\(([A-Za-z0-9])\)', label)
#         if options:
#             return options
#         # 2. 找所有以 "A."、"A:"、"A："、"A "、"A,"、"A，" 开头的
#         options = re.findall(r'\b([A-Za-z0-9])[\.:：,\s]', label)
#         if options:
#             # 去重并保持顺序
#             seen = set()
#             result = []
#             for o in options:
#                 if o not in seen:
#                     seen.add(o)
#                     result.append(o)
#             return result
#         # 3. 逗号/分号/空格分隔的单字母
#         options = re.findall(r'\b([A-Za-z0-9])\b', label)
#         # 过滤掉明显不是选项的情况（如数字等），但这里保守处理
#         if options:
#             # 去重并保持顺序
#             seen = set()
#             result = []
#             for o in options:
#                 if o not in seen:
#                     seen.add(o)
#                     result.append(o)
#             return result
#         # 4. 如果都没有，返回原始label作为单元素list
#         return [label]

#     def calculate_score(self, exact_option, exact_label_option):
#         # 返回两个list的交集和并集
#         intersection = list(set(exact_option) & set(exact_label_option))
#         union = list(set(exact_option) | set(exact_label_option))
#         return len(intersection) / len(union)
    
#     def bbox_iou(self, box1, box2, x1y1x2y2=True):
#         if x1y1x2y2:
#             b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
#             b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
#         else:
#             b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
#             b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
#             b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
#             b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

#         inter_rect_x1 = torch.max(b1_x1, b2_x1)
#         inter_rect_y1 = torch.max(b1_y1, b2_y1)
#         inter_rect_x2 = torch.min(b1_x2, b2_x2)
#         inter_rect_y2 = torch.min(b1_y2, b2_y2)

#         inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)

#         b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
#         b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

#         return inter_area / (b1_area + b2_area - inter_area + 1e-16)

#     def calculate_iou(self, pred, label):
#         if "no" in label.lower() and "detected" in label.lower():
#             if pred.lower() == label.lower():
#                 return 0.3
#             elif (
#                 "detected" in pred.lower() and
#                 "no" in pred.lower() and
#                 "<|box|>" not in pred and
#                 "<|/box|>" not in pred
#             ):
#                 # import ipdb; ipdb.set_trace()
#                 return 0.2
#             else:
#                 return 0.0

#         match_pred = re.search(r'\\boxed\{([^\}]*)\}', pred)
#         pred = match_pred.group(1).strip() if match_pred else pred.strip()
#         match_sol = re.search(r'\\boxed\{([^\}]*)\}', label)
#         label = match_sol.group(1).strip() if match_sol else label.strip()
#         try:
#             # Extract box coordinates from label string like '<|ref|>...<|/ref|><|box|>(26,42),(78,79)<|/box|>'
#             # 提取 reference 中的名字和 box
#             reference_refs = [ref.lower() for ref in re.findall(r'<\|ref\|\>(.*?)<\|/ref\|>', label)]
#             reference_boxes = re.findall(r'<\|box\|\>(.*?)<\|/box\|>', label)
#             reference_pairs = list(zip(reference_refs, reference_boxes))

#             # 提取 candidate 中的名字和 box
#             candidate_refs = [ref.lower() for ref in re.findall(r'<\|ref\|\>(.*?)<\|/ref\|>', pred)]
#             candidate_boxes = re.findall(r'<\|box\|\>(.*?)<\|/box\|>', pred)
#             candidate_pairs = list(zip(candidate_refs, candidate_boxes))

#             # 跳过没有reference_box的情况
#             if len(reference_pairs) == 0:
#                 return 0.0
#             if len(candidate_pairs) == 0:  # if there is no box in the prediction, the iou is 0
#                 candidate_box = ['(0,0),(0,0)']
#                 iou = 0.0
#             else:

#                 # 取出 reference_pairs[0][0] 的名字
#                 ref_name = reference_pairs[0][0]
#                 # 在 candidate_pairs 中查找是否有匹配的名字
#                 matched_cand = None
#                 for cand in candidate_pairs:
#                     if cand[0] == ref_name:
#                         matched_cand = cand
#                         break
#                 if matched_cand is not None:
#                     # 解析坐标
#                     ref_coords = [int(cord) for cord in reference_pairs[0][1].replace("(", "").replace(")", "").split(",") if cord.strip() != ""]
#                     cand_coords = [int(cord) for cord in matched_cand[1].replace("(", "").replace(")", "").split(",") if cord.strip() != ""]
#                 else:
#                     # 没有找到匹配的名字，使用第一个 candidate_pair
#                     ref_coords = [int(cord) for cord in reference_pairs[0][1].replace("(", "").replace(")", "").split(",") if cord.strip() != ""]
#                     cand_coords = [int(cord) for cord in candidate_pairs[0][1].replace("(", "").replace(")", "").split(",") if cord.strip() != ""]
#                 # 跳过坐标数不对的情况
#                 if len(ref_coords) != 4 or len(cand_coords) != 4:
#                     return 0.0
#                 reference_box_tensor = torch.tensor([ref_coords])
#                 candidate_box_tensor = torch.tensor([cand_coords])
#                 iou = self.bbox_iou(reference_box_tensor, candidate_box_tensor).item()
#                 # ipdb.set_trace()
#             return iou
#         except Exception as e:
#             # 跳过解析失败的样本
#             return 0.0


#     def __call__(self, infer_requests: List[Union['InferRequest', Dict]],
#                  **kwargs) -> List[float]:
#         # ipdb.set_trace()
#         rewards = []
#         # for idx, req in enumerate(infer_requests):
#         #     print(f'infer_requests[{idx}]:', req)
#         # for i, uid in enumerate(kwargs['unique_id']):
#         #     print(f'kwargs unique_id[{i}]:', uid)
#         # print('--------------------------------')
#         ground_truths = kwargs['solution']
#         task_names = kwargs['task_name']
#         predictions = infer_requests
#         for prediction, ground_truth, task_name in zip(predictions, ground_truths, task_names):
#             predicted_reason, prediction_boxed = self.extract_boxed_result(prediction)

#             if task_name == 'Impression Generation' or task_name == 'Findings Generation' or task_name == 'Findings Summarization':
#                 # print("bertscore_scorer forward")
#                 # print(list(self.bertscore_scorer.bert_scorer._model.named_parameters()))
#                 # ipdb.set_trace()
#                 # print(f"Current device: {torch.cuda.current_device()}")
#                 # import ipdb; ipdb.set_trace()
#                 if prediction_boxed is not None and predicted_reason is not None:
#                     reward = self.radcliq_scorer.predict(refs=[ground_truth], hyps=[prediction_boxed])[0]
#                     reward_reason = self.radcliq_scorer.predict_rouge1(refs=[ground_truth], hyps=[predicted_reason])[0]
#                     reward_reason = 0.2 if reward_reason > 0.2 else reward_reason
#                     reward = reward * 0.8 + reward_reason
#                 elif prediction_boxed is not None:
#                     reward = self.radcliq_scorer.predict(refs=[ground_truth], hyps=[prediction_boxed])[0]
#                     reward = reward * 0.8
#                 elif predicted_reason is not None:
#                     reward_reason = self.radcliq_scorer.predict_rouge1(refs=[ground_truth], hyps=[predicted_reason])[0]
#                     reward_reason = 0.2 if reward_reason > 0.2 else reward_reason
#                     reward = reward_reason
#                 else:
#                     reward = 0.0

#             elif task_name == 'Closed-Ended VQA' or task_name == 'Close-Ended VQA' or task_name == 'Image Classification' or task_name == 'Temporal Image Classification' or task_name == 'View Classification':
#                 # ipdb.set_trace()
#                 prediction_option = self.extract_label_option_multiple(prediction_boxed)
#                 ground_truth_option = self.extract_label_option_multiple(ground_truth)
#                 reward = self.calculate_score(prediction_option, ground_truth_option)
#             elif task_name == 'Abnormality Grounding' or task_name == 'Chest Tube Segmentation' or task_name == 'Phrase Grounding' or task_name == 'Pneumothorax Segmentation':
#                 reward = self.calculate_iou(prediction_boxed, ground_truth)
#             else:
#                 raise ValueError(f'Unsupported task name: {task_name}')
#             rewards.append(reward)
#         return rewards
# orms['external_vqa_generation_reward'] = VQA_GenerationORM


class VQA_ORM(ORM):

    def __init__(self):


        self.temp_list = []
        self.max_len = 1000

    def extract_boxed_result(self, text):
        """
        提取 \boxed{} 前面的内容和 \boxed{} 内部的内容。
        例如: "reason\\boxed{answer}" -> ("reason", "answer")
        如果只有前半部分或者只有后半部分，则只取出对应部分，没有的部分用None占位

        注意：本函数假设text为原始字符串，包含转义字符（如\\n），
        并且\boxed{...}前面可能有多行内容。
        """
        # 先尝试匹配最后一个 \boxed{...}，并提取其前面的内容
        # 支持多行内容
        pattern = r'(?s)(.*?)\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            before_boxed = match.group(1)
            boxed_content = match.group(2)
            before_boxed = before_boxed.strip() if before_boxed and before_boxed.strip() else None
            boxed_content = boxed_content.strip() if boxed_content and boxed_content.strip() else None
            return before_boxed, boxed_content
        else:
            # 检查是否有 \boxed{...} 但前面没有内容
            pattern_boxed_only = r'\\boxed{([^}]*)}'
            match_boxed_only = re.search(pattern_boxed_only, text)
            if match_boxed_only:
                boxed_content = match_boxed_only.group(1).strip() if match_boxed_only.group(1).strip() else None
                return None, boxed_content
            # 检查是否有前半部分但没有 \boxed{}
            text_clean = text.strip() if text and text.strip() else None
            return text_clean, None

    def extract_label_option_multiple(self, label):
        """
        提取label中的所有选项，支持多选和多种格式。
        输入: label字符串，如 "A, B", "(A), (B)", "(A) XX, (B) XX", "(A). XX, (B). XX", "(A): XX, (B): XX", "A XX, B XX"
        输出: 选项列表，如 ["A"], ["A", "B"]
        """
        # 1. 先找所有 (A) 这种括号包裹的
        options = re.findall(r'\(([A-Za-z0-9])\)', label)
        if options:
            return options
        # 2. 找所有以 "A."、"A:"、"A："、"A "、"A,"、"A，" 开头的
        options = re.findall(r'\b([A-Za-z0-9])[\.:：,\s]', label)
        if options:
            # 去重并保持顺序
            seen = set()
            result = []
            for o in options:
                if o not in seen:
                    seen.add(o)
                    result.append(o)
            return result
        # 3. 逗号/分号/空格分隔的单字母
        options = re.findall(r'\b([A-Za-z0-9])\b', label)
        # 过滤掉明显不是选项的情况（如数字等），但这里保守处理
        if options:
            # 去重并保持顺序
            seen = set()
            result = []
            for o in options:
                if o not in seen:
                    seen.add(o)
                    result.append(o)
            return result
        # 4. 如果都没有，返回原始label作为单元素list
        return [label]

    def calculate_score(self, exact_option, exact_label_option):
        # 返回两个list的交集和并集
        intersection = list(set(exact_option) & set(exact_label_option))
        union = list(set(exact_option) | set(exact_label_option))
        return len(intersection) / len(union)
    


    def __call__(self, infer_requests: List[Union['InferRequest', Dict]],
                 **kwargs) -> List[float]:
        # ipdb.set_trace()
        rewards = []
        # for idx, req in enumerate(infer_requests):
        #     print(f'infer_requests[{idx}]:', req)
        # for i, uid in enumerate(kwargs['unique_id']):
        #     print(f'kwargs unique_id[{i}]:', uid)
        # print('--------------------------------')
        ground_truths = kwargs['solution']
        task_names = kwargs['task_name']
        predictions = infer_requests
        for prediction, ground_truth, task_name in zip(predictions, ground_truths, task_names):
            predicted_reason, prediction_boxed = self.extract_boxed_result(prediction)

            if 'Impression Generation' in task_name or 'Findings Generation' in task_name:
                if len(self.temp_list) > 0:
                    reward = sum(self.temp_list) / len(self.temp_list)
                else:
                    reward = 0.0

            elif task_name == 'Closed-Ended VQA' or task_name == 'Close-Ended VQA' or task_name == 'Image Classification' or task_name == 'Temporal Image Classification' or task_name == 'View Classification' or task_name == 'Disease Classification' or task_name == 'Temporal Disease Change Detection':
                # ipdb.set_trace()
                if prediction_boxed is not None:
                    prediction_option = self.extract_label_option_multiple(prediction_boxed)
                    ground_truth_option = self.extract_label_option_multiple(ground_truth)
                    reward = self.calculate_score(prediction_option, ground_truth_option)
                    self.temp_list.append(reward)
                    if len(self.temp_list) > self.max_len:
                        self.temp_list.pop(0)
                else:
                    reward = 0.0
            elif task_name == 'Abnormality Grounding' or task_name == 'Chest Tube Segmentation' or task_name == 'Phrase Grounding' or task_name == 'Pneumothorax Segmentation':
                if len(self.temp_list) > 0:
                    reward = sum(self.temp_list) / len(self.temp_list)
                else:
                    reward = 0.0
            else:
                raise ValueError(f'Unsupported task name: {task_name}')
            rewards.append(reward)
        return rewards
orms['external_vqa_orm'] = VQA_ORM

class Iou_ORM(ORM):

    def __init__(self):
        self.temp_list = []
        self.max_len = 1000

    def extract_boxed_result(self, text):
        """
        提取 \boxed{} 前面的内容和 \boxed{} 内部的内容。
        例如: "reason\\boxed{answer}" -> ("reason", "answer")
        如果只有前半部分或者只有后半部分，则只取出对应部分，没有的部分用None占位

        注意：本函数假设text为原始字符串，包含转义字符（如\\n），
        并且\boxed{...}前面可能有多行内容。
        """
        # 先尝试匹配最后一个 \boxed{...}，并提取其前面的内容
        # 支持多行内容
        pattern = r'(?s)(.*?)\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            before_boxed = match.group(1)
            boxed_content = match.group(2)
            before_boxed = before_boxed.strip() if before_boxed and before_boxed.strip() else None
            boxed_content = boxed_content.strip() if boxed_content and boxed_content.strip() else None
            return before_boxed, boxed_content
        else:
            # 检查是否有 \boxed{...} 但前面没有内容
            pattern_boxed_only = r'\\boxed{([^}]*)}'
            match_boxed_only = re.search(pattern_boxed_only, text)
            if match_boxed_only:
                boxed_content = match_boxed_only.group(1).strip() if match_boxed_only.group(1).strip() else None
                return None, boxed_content
            # 检查是否有前半部分但没有 \boxed{}
            text_clean = text.strip() if text and text.strip() else None
            return text_clean, None

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        if x1y1x2y2:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        else:
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

        inter_rect_x1 = torch.max(b1_x1, b2_x1)
        inter_rect_y1 = torch.max(b1_y1, b2_y1)
        inter_rect_x2 = torch.min(b1_x2, b2_x2)
        inter_rect_y2 = torch.min(b1_y2, b2_y2)

        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)

        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        return inter_area / (b1_area + b2_area - inter_area + 1e-16)

    def calculate_iou(self, pred, label):
        # if "no" in label.lower() and "detected" in label.lower():
        #     if pred.lower() == label.lower():
        #         return 0.3
        #     elif (
        #         "detected" in pred.lower() and
        #         "no" in pred.lower() and
        #         "<|box|>" not in pred and
        #         "<|/box|>" not in pred
        #     ):
        #         # import ipdb; ipdb.set_trace()
        #         return 0.2
        #     else:
        #         return 0.0

        match_pred = re.search(r'\\boxed\{([^\}]*)\}', pred)
        pred = match_pred.group(1).strip() if match_pred else pred.strip()
        match_sol = re.search(r'\\boxed\{([^\}]*)\}', label)
        label = match_sol.group(1).strip() if match_sol else label.strip()
        try:
            # Extract box coordinates from label string like '<|ref|>...<|/ref|><|box|>(26,42),(78,79)<|/box|>'
            # 提取 reference 中的名字和 box
            # 为啥label可以提取到box, 但pred提取不到？
            # 这通常是因为pred字符串中没包含匹配的 <|box|>...</|box|> 模式
            # 你可以单独打印调试 re.findall 的结果
            reference_refs = [ref.lower() for ref in re.findall(r'<\|ref\|\>(.*?)<\|/ref\|>', label)]
            reference_boxes = re.findall(r'<\|box\|\>(.*?)<\|/box\|>', label)
            reference_pairs = list(zip(reference_refs, reference_boxes))
            # candidate里如果没box，re.findall会返回空list
            candidate_refs = [ref.lower() for ref in re.findall(r'<\|ref\|\>(.*?)<\|/ref\|>', pred)]
            candidate_boxes = re.findall(r'<\|box\|\>(.*?)<\|/box\|>', pred)
            candidate_pairs = list(zip(candidate_refs, candidate_boxes))
            # import ipdb; ipdb.set_trace()
            # 跳过没有reference_box的情况
            if len(reference_pairs) == 0:
                return 0.0
            if len(candidate_pairs) == 0:  # if there is no box in the prediction, the iou is 0
                candidate_box = ['(0,0),(0,0)']
                iou = 0.0
            else:

                # 取出 reference_pairs[0][0] 的名字
                ref_name = reference_pairs[0][0]
                # 在 candidate_pairs 中查找是否有匹配的名字
                can_name = candidate_pairs[0][0]
                if ref_name == can_name:
                    ref_rate = 1.0
                else:
                    ref_rate = 0.95
                # matched_cand = None
                # for cand in candidate_pairs:
                #     if cand[0] == ref_name:
                #         matched_cand = cand
                #         break
                # if matched_cand is not None:
                #     ref_rate = 1.0
                #     # 解析坐标
                #     ref_coords = [int(cord) for cord in reference_pairs[0][1].replace("(", "").replace(")", "").split(",") if cord.strip() != ""]
                #     cand_coords = [int(cord) for cord in matched_cand[1].replace("(", "").replace(")", "").split(",") if cord.strip() != ""]
                # else:
                #     # 没有找到匹配的名字，使用第一个 candidate_pair
                #     ref_rate = 0.95
                ref_coords = [int(cord) for cord in reference_pairs[0][1].replace("(", "").replace(")", "").split(",") if cord.strip() != ""]
                cand_coords = [int(cord) for cord in candidate_pairs[0][1].replace("(", "").replace(")", "").split(",") if cord.strip() != ""]
                # 跳过坐标数不对的情况
                if len(ref_coords) != 4 or len(cand_coords) != 4:
                    return 0.0
                reference_box_tensor = torch.tensor([ref_coords])
                candidate_box_tensor = torch.tensor([cand_coords])
                iou = self.bbox_iou(reference_box_tensor, candidate_box_tensor).item()
                iou = iou * ref_rate
                # ipdb.set_trace()
            return iou
        except Exception as e:
            # 跳过解析失败的样本
            return 0.0


    def __call__(self, infer_requests: List[Union['InferRequest', Dict]],
                 **kwargs) -> List[float]:
        # ipdb.set_trace()
        rewards = []
        # for idx, req in enumerate(infer_requests):
        #     print(f'infer_requests[{idx}]:', req)
        # for i, uid in enumerate(kwargs['unique_id']):
        #     print(f'kwargs unique_id[{i}]:', uid)
        # print('--------------------------------')
        ground_truths = kwargs['solution']
        task_names = kwargs['task_name']
        predictions = infer_requests
        for prediction, ground_truth, task_name in zip(predictions, ground_truths, task_names):
            predicted_reason, prediction_boxed = self.extract_boxed_result(prediction)

            if 'Impression Generation' in task_name or 'Findings Generation' in task_name:
                if len(self.temp_list) > 0:
                    reward = sum(self.temp_list) / len(self.temp_list)
                else:
                    reward = 0.0

            elif task_name == 'Closed-Ended VQA' or task_name == 'Close-Ended VQA' or task_name == 'Image Classification' or task_name == 'Temporal Image Classification' or task_name == 'View Classification' or task_name == 'Disease Classification' or task_name == 'Temporal Disease Change Detection':
                # ipdb.set_trace()
                if len(self.temp_list) > 0:
                    reward = sum(self.temp_list) / len(self.temp_list)
                else:
                    reward = 0.0
            elif task_name == 'Abnormality Grounding' or task_name == 'Chest Tube Segmentation' or task_name == 'Phrase Grounding' or task_name == 'Pneumothorax Segmentation':
                if prediction_boxed is not None:
                    reward = self.calculate_iou(prediction_boxed, ground_truth)
                    self.temp_list.append(reward)
                    if len(self.temp_list) > self.max_len:
                        self.temp_list.pop(0)
                else:
                    reward = 0.0
            else:
                raise ValueError(f'Unsupported task name: {task_name}')
            rewards.append(reward)
        return rewards
orms['external_iou_reward'] = Iou_ORM


class GenerationWithReasonORM(ORM):

    def __init__(self):

        self.radcliq_scorer = CompositeMetric()
        self.temp_list = []
        self.max_len = 1000

    def extract_boxed_result(self, text):
        """
        提取 \boxed{} 前面的内容和 \boxed{} 内部的内容。
        例如: "reason\\boxed{answer}" -> ("reason", "answer")
        如果只有前半部分或者只有后半部分，则只取出对应部分，没有的部分用None占位

        注意：本函数假设text为原始字符串，包含转义字符（如\\n），
        并且\boxed{...}前面可能有多行内容。
        """
        # 先尝试匹配最后一个 \boxed{...}，并提取其前面的内容
        # 支持多行内容
        pattern = r'(?s)(.*?)\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            before_boxed = match.group(1)
            boxed_content = match.group(2)
            before_boxed = before_boxed.strip() if before_boxed and before_boxed.strip() else None
            boxed_content = boxed_content.strip() if boxed_content and boxed_content.strip() else None
            return before_boxed, boxed_content
        else:
            # 检查是否有 \boxed{...} 但前面没有内容
            pattern_boxed_only = r'\\boxed{([^}]*)}'
            match_boxed_only = re.search(pattern_boxed_only, text)
            if match_boxed_only:
                boxed_content = match_boxed_only.group(1).strip() if match_boxed_only.group(1).strip() else None
                return None, boxed_content
            # 检查是否有前半部分但没有 \boxed{}
            text_clean = text.strip() if text and text.strip() else None
            return text_clean, None

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]],
                 **kwargs) -> List[float]:
        # ipdb.set_trace()
        rewards = []
        # for idx, req in enumerate(infer_requests):
        #     print(f'infer_requests[{idx}]:', req)
        # for i, uid in enumerate(kwargs['unique_id']):
        #     print(f'kwargs unique_id[{i}]:', uid)
        # print('--------------------------------')
        ground_truths = kwargs['solution']
        task_names = kwargs['task_name']
        predictions = infer_requests
        for prediction, ground_truth, task_name in zip(predictions, ground_truths, task_names):
            predicted_reason, prediction_boxed = self.extract_boxed_result(prediction)

            if 'Impression Generation' in task_name or 'Findings Generation' in task_name:
                # print("bertscore_scorer forward")
                # print(list(self.bertscore_scorer.bert_scorer._model.named_parameters()))
                # ipdb.set_trace()
                # print(f"Current device: {torch.cuda.current_device()}")
                # import ipdb; ipdb.set_trace()
                if prediction_boxed is not None and predicted_reason is not None:
                    reward = self.radcliq_scorer.predict(refs=[ground_truth], hyps=[prediction_boxed])[0]
                    # reward_reason = self.radcliq_scorer.predict_rouge1(refs=[ground_truth], hyps=[predicted_reason])[0]
                    # reward_reason = 0.05 if reward_reason > 0.05 else reward_reason
                    # reward = reward * 0.95 + reward_reason
                elif prediction_boxed is not None:
                    reward = self.radcliq_scorer.predict(refs=[ground_truth], hyps=[prediction_boxed])[0]
                    reward = reward * 0.95
                # elif predicted_reason is not None:
                #     reward_reason = self.radcliq_scorer.predict_rouge1(refs=[ground_truth], hyps=[predicted_reason])[0]
                #     reward_reason = 0.05 if reward_reason > 0.05 else reward_reason
                #     reward = reward_reason
                else:
                    reward = 0.0
                self.temp_list.append(reward)
                if len(self.temp_list) > self.max_len:
                    self.temp_list.pop(0)
            
            elif task_name == 'Closed-Ended VQA' or task_name == 'Close-Ended VQA' or task_name == 'Image Classification' or task_name == 'Temporal Image Classification' or task_name == 'View Classification' or task_name == 'Disease Classification' or task_name == 'Temporal Disease Change Detection':
                # ipdb.set_trace()
                if len(self.temp_list) > 0:
                    reward = sum(self.temp_list) / len(self.temp_list)
                else:
                    reward = 0.0
            elif task_name == 'Abnormality Grounding' or task_name == 'Chest Tube Segmentation' or task_name == 'Phrase Grounding' or task_name == 'Pneumothorax Segmentation':
                if len(self.temp_list) > 0:
                    reward = sum(self.temp_list) / len(self.temp_list)
                else:
                    reward = 0.0
            else:
                raise ValueError(f'Unsupported task name: {task_name}')
            rewards.append(reward)
        return rewards
orms['external_generation_with_reason_reward'] = GenerationWithReasonORM

class Format_boxed(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion has a specific format:
        - \boxed{} must not be empty
        - There must be non-whitespace characters before \boxed{}
        - There must be exactly one \boxed{} occurrence
        - The part before \boxed{} must not be empty or only whitespace
        """
        scores = []
        # Pattern: non-whitespace before \boxed{...} and non-empty inside
        pattern = r'(.*?)\\boxed\{([^\}]*)\}'
        boxed_pattern = r'\\boxed\{([^\}]*)\}'
        for content in completions:
            # Count all \boxed{} occurrences
            boxed_matches = re.findall(boxed_pattern, content, re.DOTALL)
            if len(boxed_matches) != 1:
                scores.append(0.0)
                continue
            # Now check the format and non-empty inside
            match = re.search(pattern, content, re.DOTALL)
            if match:
                before_boxed = match.group(1)
                inside_boxed = match.group(2)
                if before_boxed is not None and before_boxed.strip() != "" and inside_boxed.strip() != "":
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

orms['external_format_reward_boxed'] = Format_boxed

# For additional reward functions, refer to swift/plugin/orm.py.
class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


orms['external_countdown'] = CountdownORM


class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


orms['external_r1v_acc'] = MultiModalAccuracyORM


class MultiTurnThinkingTips(ORM):
    """
    A reward function example designed for use with the `ThinkingTipsScheduler`.

    This class demonstrates how to handle reward computation when a single
    training sample (or request) is split into multiple "turns" or steps.
    Specifically, it computes the reward based on the **last turn** of each
    multi-turn trajectory using a math accuracy function.

    NOTE
    ----
    If you feed fragments of the *same* trajectory as independent samples, this
    function **must return an identical reward for every fragment**
    """

    def __init__(self):
        from swift.plugin.orm import MathAccuracy
        self.acc_func = MathAccuracy()

    def __call__(self, completions, **kwargs) -> List[float]:
        trajectory_ids: List[str] = kwargs.get('request_id')

        global_trajectorys: Dict[str, List[Dict]] = kwargs.get('trajectory_inputs')

        rewards = []
        for local_tra_id in trajectory_ids:
            total_trajectory_inputs = global_trajectorys[local_tra_id]
            # For reward calculation, we use the entire trajectory of this sample.
            # Here, we specifically evaluate only the last turn.
            last_turn_messages = total_trajectory_inputs[-1]['messages']
            last_turn_completion = last_turn_messages[-1]['content']
            last_turn_solution = total_trajectory_inputs[-1]['solution']
            # Compute reward based on math accuracy for the final completion.
            reward = self.acc_func([last_turn_completion], [last_turn_solution])[0]
            rewards.append(reward)
        return rewards


orms['thinking_tips'] = MultiTurnThinkingTips


# ref implementation: https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
class CodeReward(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('e2b') is not None, (
            "The e2b package is required but not installed. Please install it using 'pip install e2b-code-interpreter'."
        )
        from dotenv import load_dotenv
        load_dotenv()

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    def run_async_from_sync(self, scripts: List[str], languages: List[str]) -> List[float]:
        """Function wrapping the `run_async` function."""
        # Create a new event loop and set it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Run the async function and get the result
            rewards = loop.run_until_complete(self.run_async(scripts, languages))
        finally:
            loop.close()

        return rewards

    async def run_async(self, scripts: List[str], languages: List[str]) -> List[float]:
        from e2b_code_interpreter import AsyncSandbox

        # Create the sandbox by hand, currently there's no context manager for this version
        try:
            sbx = await AsyncSandbox.create(timeout=30, request_timeout=3)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return [0.0] * len(scripts)
        # Create a list of tasks for running scripts concurrently
        tasks = [self.run_script(sbx, script, language) for script, language in zip(scripts, languages)]

        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results

        # Kill the sandbox after all the tasks are complete
        await sbx.kill()

        return rewards

    async def run_script(self, sbx, script: str, language: str) -> float:
        try:
            execution = await sbx.run_code(script, language=language, timeout=30)
        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            return 0.0
        try:
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that evaluates code snippets using the E2B code interpreter.

        Assumes the dataset contains a `verification_info` column with test cases.
        """
        evaluation_script_template = """
        import subprocess
        import json

        def evaluate_code(code, test_cases):
            passed = 0
            total = len(test_cases)
            exec_timeout = 5

            for case in test_cases:
                process = subprocess.run(
                    ["python3", "-c", code],
                    input=case["input"],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )

                if process.returncode != 0:  # Error in execution
                    continue

                output = process.stdout.strip()
                if output.strip() == case["output"].strip():
                    passed += 1

            success_rate = (passed / total)
            return success_rate

        code_snippet = {code}
        test_cases = json.loads({test_cases})

        evaluate_code(code_snippet, test_cases)
        """
        verification_info = kwargs['verification_info']
        languages = [info['language'] for info in verification_info]
        code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]
        scripts = [
            evaluation_script_template.format(
                code=json.dumps(code), test_cases=json.dumps(json.dumps(info['test_cases'])))
            for code, info in zip(code_snippets, verification_info)
        ]
        try:
            rewards = self.run_async_from_sync(scripts, languages)

        except Exception as e:
            logger.warning(f'Error from E2B executor: {e}')
            rewards = [0.0] * len(completions)

        return rewards


orms['external_code_reward'] = CodeReward


class CodeFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        verification_info = kwargs['verification_info']
        rewards = []
        for content, info in zip(completions, verification_info):
            pattern = r'^<think>.*?</think>\s*<answer>.*?```{}.*?```.*?</answer>(?![\s\S])'.format(info['language'])
            match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
            reward = 1.0 if match else 0.0
            rewards.append(reward)
        return rewards


orms['external_code_format'] = CodeFormat


class CodeRewardByJudge0(ORM):
    LANGUAGE_ID_MAP = {
        'assembly': 45,
        'bash': 46,
        'basic': 47,
        'c': 50,
        'c++': 54,
        'clojure': 86,
        'c#': 51,
        'cobol': 77,
        'common lisp': 55,
        'd': 56,
        'elixir': 57,
        'erlang': 58,
        'executable': 44,
        'f#': 87,
        'fortran': 59,
        'go': 60,
        'groovy': 88,
        'haskell': 61,
        'java': 62,
        'javascript': 63,
        'kotlin': 78,
        'lua': 64,
        'multi-file program': 89,
        'objective-c': 79,
        'ocaml': 65,
        'octave': 66,
        'pascal': 67,
        'perl': 85,
        'php': 68,
        'plain text': 43,
        'prolog': 69,
        'python': 71,
        'python2': 70,
        'python3': 71,
        'r': 80,
        'ruby': 72,
        'rust': 73,
        'scala': 81,
        'sql': 82,
        'swift': 83,
        'typescript': 74,
        'visual basic.net': 84
    }
    PYTHON_ID = 71

    def __init__(self):
        self.endpoint = os.getenv('JUDGE0_ENDPOINT')
        assert self.endpoint is not None, (
            'Judge0 endpoint is not set. Please set the JUDGE0_ENDPOINT environment variable.')
        x_auth_token = os.getenv('JUDGE0_X_AUTH_TOKEN')
        self.headers = {'Content-Type': 'application/json'}
        if x_auth_token is not None:
            self.headers['X-Auth-Token'] = x_auth_token

    @staticmethod
    def extract_code(completion: str, language: str) -> str:
        pattern = re.compile(rf'```{language}\n(.*?)```', re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ''
        return extracted_answer

    @classmethod
    def get_language_id(cls, language):
        if language is None:
            return cls.PYTHON_ID
        return cls.LANGUAGE_ID_MAP.get(language.lower().strip(), cls.PYTHON_ID)

    async def _evaluate_code(self, code, test_cases, language_id):
        import aiohttp
        try:
            passed = 0
            total = len(test_cases)

            for case in test_cases:
                if code is not None and code != '':
                    async with aiohttp.ClientSession() as session:
                        payload = {
                            'source_code': code,
                            'language_id': language_id,
                            'stdin': case['input'],
                            'expected_output': case['output']
                        }
                        logger.debug(f'Payload: {payload}')
                        async with session.post(
                                self.endpoint + '/submissions/?wait=true', json=payload,
                                headers=self.headers) as response:
                            response_json = await response.json()
                            logger.debug(f'Response: {response_json}')
                            if response_json['status']['description'] == 'Accepted':
                                passed += 1

            success_rate = (passed / total)
            return success_rate
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            return 0.0

    def run_async_from_sync(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            rewards = loop.run_until_complete(self.run_async())
        finally:
            loop.close()
        return rewards

    async def run_async(self):
        tasks = [
            self._evaluate_code(code, info['test_cases'], CodeRewardByJudge0.get_language_id(info['language']))
            for code, info in zip(self.code_snippets, self.verification_info)
        ]
        results = await asyncio.gather(*tasks)
        rewards = list(results)
        return rewards

    def __call__(self, completions, **kwargs) -> List[float]:
        self.verification_info = kwargs['verification_info']

        languages = [info['language'] for info in self.verification_info]
        self.code_snippets = [
            self.extract_code(completion, language) for completion, language in zip(completions, languages)
        ]

        try:
            rewards = self.run_async_from_sync()
        except Exception as e:
            logger.warning(f'Error from Judge0 executor: {e}')
            rewards = [0.0] * len(completions)
        return rewards


orms['external_code_reward_by_judge0'] = CodeRewardByJudge0


# ref implementation: https://github.com/qiancheng0/ToolRL/blob/main/verl/utils/reward_score/rlla.py
# arxiv paper: https://arxiv.org/abs/2504.13958
# MAX1STEP30MAX3: enable Two stage reward Setting include Format and Correctness
# SCHEDULEREWARD: enable Dynamic (Finegrained) reward Setting include Format and Correctness
# Correctness Reward Granularity:
# COARSEREWARD -> Coarse, INTERMEDIATEREWARD -> Intermediate, REFINEDREWARD -> Finegrained
class ToolUseFormatReward(ORM):

    def __init__(self):
        self.format_max_possible = 1.0
        self.format_min_possible = 0.0

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        max_possible_reward = self.format_max_possible
        min_possible_reward = self.format_min_possible
        # Two stage (Coarse) Setting, divide training into two phases. Format Reward in [0,0.5] if step < 30 else [0,1]
        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1':
            if global_step >= 30:
                max_possible_reward = self.format_max_possible / 2
                min_possible_reward = self.format_min_possible / 2
            else:
                max_possible_reward = self.format_max_possible
                min_possible_reward = self.format_min_possible

        # apply continuous interpolation between the two reward scales throughout training.
        if str(os.getenv('SCHEDULEREWARD', 0)) == '1':
            max_possible_reward = 2 - (2 - max_possible_reward) * global_step / 150
            min_possible_reward = -2 + (2 + min_possible_reward) * global_step / 150
            if max_possible_reward < 1.0:
                max_possible_reward = 1.0
            if min_possible_reward > -1.0:
                min_possible_reward = -1.0

        rewards = []
        responses = completions

        for response, ans in zip(responses, solution):
            reward = min_possible_reward
            if '<response>' in ans and '<tool_call>' not in ans:
                pattern = r'^<think>.*?</think>\s*<response>.*?</response>$'
                if re.search(pattern, response,
                             re.DOTALL) and response.count('<response>') == 1 and response.count('</response>') == 1:
                    reward = max_possible_reward
            elif '<response>' not in ans and '<tool_call>' in ans:
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>$'
                if re.search(pattern, response,
                             re.DOTALL) and response.count('<tool_call>') == 1 and response.count('</tool_call>') == 1:
                    reward = max_possible_reward
            elif '<response>' in ans and '<tool_call>' in ans:
                pattern = r'^<think>.*?</think>\s*<tool_call>.*?</tool_call>\s*<response>.*?</response>$'
                if (re.search(pattern, response, re.DOTALL) and response.count('<tool_call>') == 1
                        and response.count('</tool_call>') == 1 and response.count('<response>') == 1
                        and response.count('</response>') == 1):
                    reward = max_possible_reward
            else:
                pattern = r'^<think>.*?</think>$'
                if re.search(pattern, response, re.DOTALL):
                    reward = max_possible_reward

            rewards.append(reward)

        return rewards


orms['external_tooluse_format_reward'] = ToolUseFormatReward


class ToolUseLengthReward(ORM):

    def __init__(self):
        self.length_max_possible = 1.0
        self.length_min_possible = 0.0

    # customized reward functions: length
    def __call__(self, completions, solution, **kwargs):
        max_possible_reward = self.length_max_possible
        min_possible_reward = self.length_min_possible
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        # SCHEDULELENGTH: enable Dynamic Length Reward
        if os.getenv('SCHEDULELENGTH', 0) == '1':
            max_reward_len = (640 - 384) * global_step / 105 + 384
        else:
            max_reward_len = 512
        """Reward function that gives higher scores to longer completions."""
        responses = completions
        rewards = []

        for response, ans in zip(responses, solution):
            if '<think>' not in response or '</think>' not in response:
                rewards.append(min_possible_reward)
                continue
            think_responses = response.split('<think>')[-1].split('</think>')[0].strip()
            reward = round(len(think_responses.split()) / max_reward_len, 2)
            if reward > 1.0:
                reward = 1.0

            final_reward = reward * (max_possible_reward - min_possible_reward) + min_possible_reward
            rewards.append(final_reward)

        return rewards


orms['external_tooluse_length_reward'] = ToolUseLengthReward


class ToolUseCorrectnessReward(ORM):

    def __init__(self):
        if str(os.getenv('CORRECTMAX1', 0)) == '1':
            self.tool_max_possible = 1.0
            self.tool_min_possible = -1.0
        else:
            self.tool_max_possible = 3.0
            self.tool_min_possible = -3.0

    def match_score(self, list1, list2):
        if list1 == list2:
            return 1.0

        if os.getenv('REFINEDREWARD', 0) == '1':
            if list1 != list2:
                return 0.0

        if not list1 or not list2:
            return 0.0

        count1 = Counter(list1)  # Frequency count for list1
        count2 = Counter(list2)  # Frequency count for list2

        intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
        max_possible = len(list1) + len(list2) - intersection

        return intersection / max_possible if max_possible > 0 else 0.0

    def compute_tool_call_reward(self, gt_tools, pd_tools, max_possible_reward, min_possible_reward):
        if gt_tools == pd_tools:
            return max_possible_reward

        if os.getenv('COARSEREWARD', 0) == '1':
            if gt_tools != pd_tools:
                return min_possible_reward

        gt_names = [tool['name'] for tool in gt_tools]
        pd_names = [tool['name'] for tool in pd_tools]
        score = self.match_score(list(gt_names), list(pd_names))

        local_max_possible = 1.0
        used_pd_indices = set()  # Keep track of matched pd_tools

        for gt_tool in gt_tools:
            gt_name = gt_tool['name']
            gt_params = gt_tool['parameters']

            if str(os.getenv('INTERMEDIATEREWARD', 0)) == '1':
                local_max_possible += 1.0
            else:
                local_max_possible += 1.0 + len(gt_params)

            best_match = None
            best_match_score = 0.0
            best_match_index = -1

            # Find the best matching unused pd_tool
            for i, pd_tool in enumerate(pd_tools):
                if i in used_pd_indices or pd_tool['name'] != gt_name:
                    continue

                if str(os.getenv('INTERMEDIATEREWARD', 0)) == '1':
                    if gt_tool == pd_tool:
                        best_match = pd_tool
                        best_match_index = i
                        best_match_score = 1.0
                        break
                    else:
                        continue

                pd_params = pd_tool['parameters']
                param_score = self.match_score(list(gt_params.keys()), list(pd_params.keys()))

                # Calculate correctness score for parameter values
                correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

                total_score = param_score + correctness_score

                if total_score > best_match_score:
                    best_match_score = total_score
                    best_match = pd_tool
                    best_match_index = i

            if best_match:
                used_pd_indices.add(best_match_index)
                score += best_match_score

        return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward

    # custoimzed reward functions: tool call correctness
    def __call__(self, completions, solution, **kwargs):
        trainer_state = kwargs.get('trainer_state')
        global_step = trainer_state.global_step
        max_possible_reward = self.tool_max_possible
        min_possible_reward = self.tool_min_possible
        # two stage (Coarse) Setting, divide training into two phases.
        if str(os.getenv('MAX1STEP30MAX3', 0)) == '1':
            if global_step < 30:
                max_possible_reward = max_possible_reward / 3
                min_possible_reward = min_possible_reward / 3
            else:
                max_possible_reward = max_possible_reward
                min_possible_reward = min_possible_reward
        # apply continuous interpolation between the two reward scales throughout training.
        if str(os.getenv('SCHEDULEREWARD', 0)) == '1':
            max_possible_reward = (max_possible_reward - 2) * global_step / 150 + 2
            min_possible_reward = (min_possible_reward + 2) * global_step / 150 - 2
            if max_possible_reward > 3.0:
                max_possible_reward = 3.0
            if min_possible_reward < -3.0:
                min_possible_reward = -3.0

        responses = completions
        rewards = []

        for response, ans in zip(responses, solution):
            reward = 0.0

            if '<tool_call>' not in ans:
                # if "<tool_call>" not in response and "</tool_call>" not in response:
                #     reward = max_possible_reward
                # else:
                #     reward = min_possible_reward
                rewards.append(reward)
                continue

            gt_tool_call = ans.split('<tool_call>')[1].split('</tool_call>')[0].strip()
            gt_tools = gt_tool_call.split('\n')
            gt_tools = [json.loads(tool) for tool in gt_tools]  # each diction contains "name" and "parameter"

            try:
                # if the format is not correct, directly give the lowest possible score
                assert '<tool_call>' in response
                assert '</tool_call>' in response
                pd_tools = response.split('<tool_call>')[1].split('</tool_call>')[0].strip().split('\n')
                pd_tools = [json.loads(tool) for tool in pd_tools]
                reward = self.compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward,
                                                       min_possible_reward)  # top reward is 2
            except (ValueError, IndexError, AssertionError):
                reward = min_possible_reward

            rewards.append(reward)

        return rewards


orms['external_tooluse_correct_reward'] = ToolUseCorrectnessReward
"""
TO CUSTOMIZE REWARD MODEL:
    Step 1: Define a Reward Class
        Implement your custom reward calculation logic within the __call__ method.
        The method accepts the messages generated by the model during interactions
        and dataset columns as inputs parameters.

    Step 2: Add your reward model plugin to the rm_plugins registry:
        rm_plugins['my_rm_plugin'] = MyRMPlugin

    Step 3: Configure the Arguments
        Run the script with:
        --external_plugins /path/to/plugin.py \
        --reward_model_plugin my_rm_plugin

For GenRM you can refer to swift/llm/plugin/rm_plugin/GenRMPlugin
"""


class CustomizedRMPlugin:
    """
    Customized Reward Model Plugin, same to DefaultRMPlugin

    It assumes that `self.model` is a classification model with a value head(output dimmension 1).
    The first logits value from the model's output is used as the reward score.
    """

    def __init__(self, model, template):
        self.model = model
        self.template: Template = template

    def __call__(self, inputs, **kwargs):
        batched_inputs = [self.template.encode(deepcopy(infer_request)) for infer_request in inputs]
        reward_inputs = to_device(self.template.data_collator(batched_inputs), self.model.device)

        with torch.inference_mode():
            return self.model(**reward_inputs).logits[:, 0]


class QwenLongPlugin(DefaultRMPlugin):
    # https://arxiv.org/abs/2505.17667
    # NOTE: you should customize the verified reward function, you can refer to
    # https://github.com/Tongyi-Zhiwen/QwenLong-L1/tree/main/verl/verl/utils/reward_score
    # hf_dataset: https://huggingface.co/datasets/Tongyi-Zhiwen/DocQA-RL-1.6K/viewer/default/train
    # ms_dataset: https://modelscope.cn/datasets/iic/DocQA-RL-1.6K
    def __init__(self, model, template, accuracy_orm=None):
        super().__init__(model, template)
        # initilize PTEngine to infer
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)  # 0: no limit
        self.request_config = RequestConfig(temperature=0)  # customise your request config here
        self.system = textwrap.dedent("""
            You are an expert in verifying if two answers are the same.

            Your input consists of a problem and two answers: Answer 1 and Answer 2.
            You need to check if they are equivalent.

            Your task is to determine if the two answers are equivalent, without attempting to solve the original problem.
            Compare the answers to verify they represent identical values or meanings,
            even when expressed in different forms or notations.

            Your output must follow this format:
            1) Provide an explanation for why the answers are equivalent or not.
            2) Then provide your final answer in the form of: [[YES]] or [[NO]]

            Problem: {problem_placeholder}
            Answer 1: {answer1_placeholder}
            Answer 2: {answer2_placeholder}
        """)  # noqa
        self.accuracy_orm = accuracy_orm

    def __call__(self, inputs, **kwargs):
        completions = [example['messages'][-1]['content'] for example in inputs]
        ground_truths = [example['reward_model']['ground_truth'] for example in inputs]
        rm_inputs = self.prepare_rm_inputs(inputs, completions, ground_truths)

        results = self.engine.infer(rm_inputs, self.request_config, use_tqdm=False)
        llm_rewards = self.compute_rewards(results)

        if self.accuracy_orm:
            verified_rewards = self.accuracy_orm(completions, ground_truths)
        else:
            verified_rewards = [0.0] * len(llm_rewards)

        rewards = [max(r1, r2) for r1, r2 in zip(llm_rewards, verified_rewards)]
        return torch.tensor(rewards, dtype=torch.float32)

    def prepare_rm_inputs(self, inputs: List[Dict], completions, ground_truths) -> List[Dict]:
        rm_inputs = []
        for infer_request, completion, ground_truth in zip(inputs, completions, ground_truths):
            # Deep copy to prevent modification of original input
            rm_infer_request = deepcopy(infer_request)
            problem = infer_request['messages'][0]['content']
            start_index = problem.index('</text>')
            end_index = problem.index('Format your response as follows:')
            question = problem[start_index:end_index].replace('</text>', '').strip()
            prompt = self.system.format(
                problem_placeholder=question, answer1_placeholder=completion, answer2_placeholder=ground_truth)

            # Construct new messages tailored for the reward model
            rm_messages = [{'role': 'user', 'content': prompt}]

            # Update the messages in the reward infer request
            rm_infer_request['messages'] = rm_messages
            rm_inputs.append(rm_infer_request)
        return rm_inputs

    @staticmethod
    def extract_reward(model_output: str) -> float:
        match = re.search(r'\[([A-Z]+)\]', model_output)
        if match:
            answer = match.group(1)
            if answer == 'YES':
                return 1.0
            elif answer == 'NO':
                return 0.0
            else:
                logger.warning("Unexpected answer, expected 'YES' or 'NO'.")
                return 0.0
        else:
            logger.warning("Unable to extract reward score from the model's output, setting reward to 0")
            return 0.0  # Or raise ValueError("Format incorrect")

    def compute_rewards(self, results: List[ChatCompletionResponse]) -> List[float]:
        """
        Compute average reward scores from the reward model's outputs.

        Args:
            results (List[ChatCompletionResponse]): A list of results from the reward model.

        Returns:
            List[float]: A list of average reward scores.
        """
        rewards = []
        for idx, output in enumerate(results):
            try:
                cur_rewards = []
                for choice in output.choices:
                    response = choice.message.content
                    reward = self.extract_reward(response)
                    cur_rewards.append(reward)
                cur_rewards = [r for r in cur_rewards if r is not None]
                if cur_rewards:
                    average_reward = sum(cur_rewards) / len(cur_rewards)
                else:
                    average_reward = 0.0
                    logger.warning('No valid rewards extracted. Assigning reward score of 0.0.')

                rewards.append(average_reward)
            except Exception as e:
                logger.error(f'Error computing reward: {e}')
                rewards.append(0.0)  # Assign default reward score on failure
        return rewards


rm_plugins['my_rmplugin'] = CustomizedRMPlugin
rm_plugins['qwenlong'] = QwenLongPlugin
"""
TO CUSTOMIZE MULTITURN SCHEDULER:
    Step 1: Define a Scheduler Class
        Implement your custom scheduler with the following methods:
            - step (Required): Constructs the next round of the infer request.
            - check_finished (Optional): Determines whether the current round has finished,
                which defaults to ending when the inference result is truncated (over length) or
                when the maximum number of rounds is reached.
            or override run method in MultiTurnScheduler class.

        Both methods accept:
            - the last turn's InferRequest/response_choice
            - the current turn count

    Step 2: Add your scheduler to the multi_turns registry:
        multi_turns['my_scheduler'] = MyScheduler

    Step 3: Configure the Arguments
        Run the script with:
        swift rollout \
            --external_plugins /path/to/plugin.py \
            --multi_turn_scheduler my_scheduler
"""


class ToolCallScheduler(MultiTurnScheduler):
    # A simple scheduler that supports tool calls by overriding the `step` method
    # Tool parsing uses the ReAct format
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A simple tool registry. Extend or replace with your own tools as needed.
        self.tools = {
            'calculator': self._calculator_tool,
        }

    def _calculator_tool(self, expression: str) -> str:
        # A very small sandboxed calculator
        # The calculator tool implemented here can perform only basic arithmetic operations and
        # may not be able to solve all math problems in the dataset.
        import ast
        import operator

        def _evaluate_ast_node(node) -> Union[int, float]:
            operators = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }

            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                else:
                    raise TypeError(f'Unsupported constant type: {type(node.value)}')

            elif isinstance(node, ast.Num):
                return node.n

            elif isinstance(node, ast.BinOp):
                left = _evaluate_ast_node(node.left)
                right = _evaluate_ast_node(node.right)
                op = operators.get(type(node.op))

                if op is None:
                    raise TypeError(f'Unsupported operation: {type(node.op).__name__}')

                if isinstance(node.op, ast.Div) and right == 0:
                    raise ZeroDivisionError('Division by zero')

                return op(left, right)

            elif isinstance(node, ast.UnaryOp):
                operand = _evaluate_ast_node(node.operand)
                op = operators.get(type(node.op))

                if op is None:
                    raise TypeError(f'Unsupported unary operation: {type(node.op).__name__}')

                return op(operand)

            else:
                raise TypeError(f'Unsupported AST node type: {type(node).__name__}')

        try:
            expression = expression.strip().replace(' ', '')

            if not re.match(r'^[0-9+\-*/().\s]+$', expression):
                return 'Error: expression contains disallowed characters.'

            if expression.count('(') != expression.count(')'):
                return 'Error: unmatched parentheses.'

            try:
                result = ast.literal_eval(expression)
                return f'Result: {result}'
            except (ValueError, SyntaxError):
                node = ast.parse(expression, mode='eval')
                result = _evaluate_ast_node(node.body)
                return f'Result: {result}'

        except Exception as e:
            return f'Calculation error: {e}'

    def _extract_tool_calls(self, text: str):
        """
        Parse tool-call patterns using ReAct format from model output.
        Format: Action: tool_name\nAction Input: parameters
        """
        import re

        pattern = r'Action:\s*(.*?)\s*\nAction Input:\s*(.*?)(?:\n|$)'
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            return None
        return [{'tool': name.strip(), 'params': params.strip()} for name, params in matches]

    def _execute_tools(self, tool_calls):
        """Run each requested tool and collect its observation string."""
        results = []
        for call in tool_calls:
            name, params = call['tool'], call['params']
            if name in self.tools:
                try:
                    result = self.tools[name](params)
                    results.append(result)
                except Exception as e:
                    results.append(f'tool error {e}')
            else:
                results.append(f'unknown tool {name}')
        return results

    def check_finished(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
                       current_turn: int) -> bool:
        completion = response_choice.message.content
        tool_calls = self._extract_tool_calls(completion)
        if tool_calls is None:
            return True

        return super().check_finished(infer_request, response_choice, current_turn)

    def step(self, infer_request: 'RolloutInferRequest', response_choice: 'ChatCompletionResponseChoice',
             current_turn: int) -> Dict:
        completion = response_choice.message.content
        token_ids = response_choice.token_ids
        loss_mask = [1] * len(token_ids)
        tool_calls = self._extract_tool_calls(completion)
        # assert len(tool_calls) == 1, 'this scheduler is designed for one tool call per turn'
        tool_results = self._execute_tools(tool_calls)
        # append tool result to the completion
        infer_request.messages[-1]['content'] += (tool_results[0])

        tokenizer = self.infer_engine.default_template.tokenizer
        result_tokens = tokenizer.encode(tool_results[0], add_special_tokens=False)
        token_ids.extend(result_tokens)
        loss_mask.extend([0] * len(result_tokens))

        return {
            'infer_request': infer_request,
            'response_token_ids': token_ids,
            'response_loss_mask': loss_mask,
            'rollout_infos': {
                'tool_results': tool_results[0],
                'num_turns': current_turn,
            }
        }


multi_turns['tool_call_scheduler'] = ToolCallScheduler


# register GYM env
class CustomEnv(Env):
    pass


envs['custom_env'] = CustomEnv


class CustomCtxManager(ContextManager):
    pass


context_managers['custom_ctx'] = CustomCtxManager
