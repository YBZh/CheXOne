import json
# path = '/home/zhangyabin/project/data/chestxagent_sft_train_all_tasks_Close-Ended_VQA_subset.json'
# new_path = '/home/zhangyabin/project/data/grpo_prepare/chestxagent_sft_train_all_tasks_Close-Ended_VQA_subset_standardized.json' # 356018
# path = '/home/zhangyabin/project/data/chestxagent_sft_train_all_tasks_View_Classification_subset.json'
# new_path = '/home/zhangyabin/project/data/grpo_prepare/chestxagent_sft_train_all_tasks_View_Classification_subset_standardized.json' # 348297
# path = '/home/zhangyabin/project/data/chestxagent_sft_train_all_tasks_Temporal_Image_Classification_subset.json'
# new_path = '/home/zhangyabin/project/data/grpo_prepare/chestxagent_sft_train_all_tasks_Temporal_Image_Classification_subset_standardized.json' # 13806, including 1177 multiple options
# path = '/home/zhangyabin/project/data/chestxagent_sft_train_all_tasks_Image_Classification_subset.json'
# new_path = '/home/zhangyabin/project/data/grpo_prepare/chestxagent_sft_train_all_tasks_Image_Classification_subset_standardized.json' # 525693, including 128342 multiple options

vqa_set = [
    # '/home/zhangyabin/project/data/grpo_prepare/vqa_set/chestxagent_sft_train_all_tasks_Close-Ended_VQA_subset_standardized.json', # 356018
    # '/home/zhangyabin/project/data/grpo_prepare/vqa_set/chestxagent_sft_train_all_tasks_View_Classification_subset_standardized.json', # 348297
    # '/home/zhangyabin/project/data/grpo_prepare/vqa_set/chestxagent_sft_train_all_tasks_Temporal_Image_Classification_subset_standardized.json', # 621, single options
    # '/home/zhangyabin/project/data/grpo_prepare/vqa_set/chestxagent_sft_train_all_tasks_Image_Classification_subset_standardized.json', # 525688, including 128342 multiple options
    # '/home/zhangyabin/project/data/grpo_prepare/vqa_set/VQA_ReXGradient_160K_train_valid.json', # 612976
    # '/home/zhangyabin/project/data/grpo_prepare/generation/chestxagent_sft_train_all_tasks_Findings_Generation_subset_standardized.json',  # 201540
    # '/home/zhangyabin/project/data/grpo_prepare/generation/chestxagent_sft_train_all_tasks_Impression_Generation_subset_standardized.json',  # 393237
    # '/home/zhangyabin/project/data/grpo_prepare/generation/chestxagent_sft_train_all_tasks_Findings_Summarization_subset_standardized.json',  # 165414
    # "/home/zhangyabin/project/data/grpo_prepare/generation/ReXGradient_160K_Findings_Generation_subset_standardized.json", # 149998
    # "/home/zhangyabin/project/data/grpo_prepare/generation/ReXGradient_160K_Impression_Generation_subset_standardized.json", # 149998
    # "/home/zhangyabin/project/data/grpo_prepare/generation/ReXGradient_160K_Findings_Summarization_subset_standardized.json" # 149998
    # '/home/zhangyabin/project/data/grpo_prepare/iou_set_qwenvl3/chestxagent_sft_train_all_tasks_Abnormality_Grounding_subset_standardized_18465.json', 
    # '/home/zhangyabin/project/data/grpo_prepare/iou_set_qwenvl3/chestxagent_sft_train_all_tasks_Chest_Tube_Segmentation_subset_standardized_1423.json', #
    # '/home/zhangyabin/project/data/grpo_prepare/iou_set_qwenvl3/chestxagent_sft_train_all_tasks_Phrase_Grounding_subset_standardized_961.json', # 
    # '/home/zhangyabin/project/data/grpo_prepare/iou_set_qwenvl3/chestxagent_sft_train_all_tasks_Pneumothorax_Segmentation_subset_standardized_4985.json', # 
    './data/grpo_prepare/generation/Findings_Generation_ReXGradient_160K_train_with_indication_139999.json', 
    './data/grpo_prepare/generation/Impression_Generation_ReXGradient_160K_train_with_indication_139999.json', #
    './data/grpo_prepare/generation/chestxagent_sft_train_all_tasks_Findings_Generation_with_Indication_subset_148090.json', # 
    './data/grpo_prepare/generation/chestxagent_sft_train_all_tasks_Impression_Generation_with_Indication_subset_172993.json', # 
    './data/grpo_prepare/generation/chestxagent_sft_train_all_tasks_Progression_Findings_Generation_subset_108796.json', # 
    './data/grpo_prepare/generation/chestxagent_sft_train_all_tasks_Progression_Impression_Generation_subset_193079.json', # 
]
# path = '/home/zhangyabin/project/data/chestxagent_sft_train_all_tasks_Abnormality_Grounding_subset.json'
# new_path = '/home/zhangyabin/project/data/grpo_prepare/iou_set/chestxagent_sft_train_all_tasks_Abnormality_Grounding_subset_standardized.json' # 32727
# path = '/home/zhangyabin/project/data/chestxagent_sft_train_all_tasks_Chest_Tube_Segmentation_subset.json'
# new_path = '/home/zhangyabin/project/data/grpo_prepare/iou_set/chestxagent_sft_train_all_tasks_Chest_Tube_Segmentation_subset_standardized.json' # 2685
# path = '/home/zhangyabin/project/data/chestxagent_sft_train_all_tasks_Phrase_Grounding_subset.json'
# new_path = '/home/zhangyabin/project/data/grpo_prepare/iou_set/chestxagent_sft_train_all_tasks_Phrase_Grounding_subset_standardized.json' # 
# path = '/home/zhangyabin/project/data/chestxagent_sft_train_all_tasks_Pneumothorax_Segmentation_subset.json'
# new_path = '/home/zhangyabin/project/data/grpo_prepare/iou_set/chestxagent_sft_train_all_tasks_Pneumothorax_Segmentation_subset_standardized.json' # 15129

repeat_num = 8

for path in vqa_set:
    repeat_data = []
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {path}")
    for item in data:
        for i in range(repeat_num):
            import copy
            item_copy = copy.deepcopy(item)
            item_copy['unique_id'] = item_copy['unique_id'] + ' [repeat' + str(i) + ']'
            # The previous implementation caused multiple duplicates because:
            # 1. The outer loop iterates over each dataset file
            # 2. The inner loop repeats each data item repeat_num times (8 times)
            # 3. Each repetition appends the same text to the original value
            # 4. If the original data has already been processed, the text will accumulate repeatedly
            # Solution: Check whether the text has already been added to avoid adding repeatedly
            if 'content' not in item_copy['messages'][0] and 'value' in item_copy['messages'][0]:
                item_copy['messages'][0]['content'] = item_copy['messages'][0]['value']
                del item_copy['messages'][0]['value']
            if 'content' not in item_copy['messages'][1] and 'value' in item_copy['messages'][1]:
                item_copy['messages'][1]['content'] = item_copy['messages'][1]['value']
                del item_copy['messages'][1]['value']    

            if not item_copy['messages'][0]['content'].endswith(r' Please reason step by step, and put your final answer within \boxed{{}}.'):
                item_copy['messages'][0]['content'] = item_copy['messages'][0]['content'] + r' Please reason step by step, and put your final answer within \boxed{{}}.'
            repeat_data.append(item_copy)
    with open(path.replace('.json', '_repeat.json'), 'w') as f:
        json.dump(repeat_data, f, indent=4, ensure_ascii=False)