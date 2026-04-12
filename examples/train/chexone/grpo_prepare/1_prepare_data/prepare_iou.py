# Standardize the IOU dataset for GRPO preparation
import json
import re
    
validation_path = '/home/zhangyabin/project/data/chestxagent_sft_val_all_tasks.json'
validation_file = json.load(open(validation_path, 'r'))

path = '/home/zhangyabin/project/data/chestxagent_sft_train_all_tasks_Abnormality_Grounding_subset.json'
new_path = '/home/zhangyabin/project/data/grpo_prepare/iou_set/chestxagent_sft_train_all_tasks_Abnormality_Grounding_subset_standardized.json' # 32727
# path = '/home/zhangyabin/project/data/chestxagent_sft_train_all_tasks_Chest_Tube_Segmentation_subset.json'
# new_path = '/home/zhangyabin/project/data/grpo_prepare/iou_set/chestxagent_sft_train_all_tasks_Chest_Tube_Segmentation_subset_standardized.json' ## 2685
# path = '/home/zhangyabin/project/data/chestxagent_sft_train_all_tasks_Phrase_Grounding_subset.json'
# new_path = '/home/zhangyabin/project/data/grpo_prepare/iou_set/chestxagent_sft_train_all_tasks_Phrase_Grounding_subset_standardized.json' ## 
# path = '/home/zhangyabin/project/data/chestxagent_sft_train_all_tasks_Pneumothorax_Segmentation_subset.json'
# new_path = '/home/zhangyabin/project/data/grpo_prepare/iou_set/chestxagent_sft_train_all_tasks_Pneumothorax_Segmentation_subset_standardized.json' ## 15129

print('loading validation file...',path)
selected_file = []
for item in validation_file:
    unique_id = item['unique_id']
    if 'Abnormality_Grounding' in unique_id:
        selected_file.append(item)
    # if 'Chest Tube Segmentation' in unique_id:
    #     selected_file.append(item)
    # if 'Phrase_Grounding' in unique_id:
    #     selected_file.append(item)
    # if 'Pneumothorax_Segmentation' in unique_id:
    #     selected_file.append(item)
# print('selected count:', len(selected_file))  

file = json.load(open(path, 'r'))

print('length of selected validation file:', len(selected_file))
print('length of train file:', len(file))
merged_file = selected_file + file
# print('length of merged file:', len(merged_file))

multi_option_count = 0


def standardize_options_and_answer(q, a):
    global multi_option_count
    # Match all options (digit/letter) and their content
    option_pattern = re.compile(r"\(([^)]+)\)\s*([^\n]+)")
    options = option_pattern.findall(q)
    if not options:
        # print('no options', q, a)
        return None, None

    abcd = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    option_map = {}
    new_options_strs = []
    for idx, (orig_key, value) in enumerate(options):
        if idx >= len(abcd):
            break
        letter = abcd[idx]
        option_map[orig_key.strip()] = letter
        new_options_strs.append(f"({letter}) {value.strip()}")

    # Replace the Options section - handle the case with and without the "Options:" separator
    q_split = q.split("Options:")
    if len(q_split) == 2:
        # Case with "Options:" separator
        before, after = q_split
        after_split = after.split('\n')
        option_lines = []
        rest_lines = []
        for line in after_split:
            if re.match(r"\([^)]+\)\s*[^\n]+", line.strip()):
                option_lines.append(line)
            else:
                rest_lines.append(line)
        processed_options = []
        for opt in new_options_strs:
            m = re.match(r"\(([^)]+)\)\s*([A-Z]):\s*(.+)", opt)
            if m:
                letter, _, content = m.groups()
                processed_options.append(f"({letter}) {content}")
            else:
                processed_options.append(opt)
        new_q = before + "Options:\n" + "\n".join(processed_options)
        if rest_lines:
            new_q += "\n" + "\n".join(rest_lines)
    else:
        # Case without the "Options:" separator - directly replace all options
        new_q = q
        for orig_key, value in options:
            orig_key = orig_key.strip()
            if orig_key in option_map:
                mapped_letter = option_map[orig_key]
                # Replace the original option format with the new alphabet letter format
                old_option = f"({orig_key}) {value.strip()}"
                new_option = f"({mapped_letter}) {value.strip()}"
                new_q = new_q.replace(old_option, new_option)

    # Answer standardization - handle various formats for options (number, lowercase, uppercase letters)
    answer_pattern = re.compile(r"\(([^)]+)\)\s*([^\(\),]*)")
    answer_matches = []
    for match in answer_pattern.finditer(a):
        key, value = match.group(1).strip(), match.group(2)
        if key in option_map:
            answer_matches.append((key, value))
        else:
            if answer_matches and key.strip():
                prev_key, prev_value = answer_matches[-1]
                answer_matches[-1] = (prev_key, (prev_value + f"({key}) {value}").strip())

    if answer_matches:
        new_a_parts = []
        for orig_key, value in answer_matches:
            orig_key = orig_key.strip()
            if orig_key in option_map:
                mapped_letter = option_map[orig_key]
                if value.strip():
                    new_a_parts.append(f"({mapped_letter}) {value.strip()}")
                else:
                    new_a_parts.append(f"({mapped_letter})")
            else:
                if value.strip():
                    new_a_parts.append(f"({orig_key}) {value.strip()}")
                else:
                    new_a_parts.append(f"({orig_key})")
        processed_new_a_parts = []
        for part in new_a_parts:
            m = re.match(r"\(([^)]+)\)\s*[A-Z]:\s*(.+)", part)
            if m:
                letter, content = m.groups()
                processed_new_a_parts.append(f"({letter}) {content}")
            else:
                processed_new_a_parts.append(part)
        new_a = " , ".join(processed_new_a_parts)
    else:
        new_a = a
    # If there are more than 1 (X) in new_a, print
    num_options = len(re.findall(r"\([A-Z]\)", new_a))
    if num_options > 1:
        multi_option_count += 1
        # return None, None
        # print(f"Multiple options detected in answer: new_a: {new_a}, q: {new_q}")
    # A more elegant way: when writing to json, enforce ensure_ascii=False, to ensure all characters (including quotes) are output as original unicode.
    # No need to do any string replacement here, just return directly
    return new_q, new_a

# 1. Because the SFT model has never seen the box format under reasoning prompt, you need to add in_context_prompt, otherwise the model output format will have problems  
# 2. The "not detected" format is removed, because the GRPO reward function is difficult to balance the reward between box and not detected, which leads to the model tending to output "not detected".
# in_context_prompt = "\nOutput specification:\n\nIf a target is detected, output:\n<|ref|>target name<|/ref|><|box|>(x1,y1),(x2,y2)<|/box|>\nExample: <|ref|>pneumothorax<|/ref|><|box|>(12,60),(18,87)<|/box|>\nIf no target is detected, output:\nNo target name detected.\nExample: No pneumothorax detected."
in_context_prompt = "\nOutput format:\n<|ref|>target name<|/ref|><|box|>(x1,y1),(x2,y2)<|/box|>\nExample: <|ref|>pneumothorax<|/ref|><|box|>(12,60),(18,87)<|/box|>\n"


count = 0
updated_file = []
for item in merged_file:
    # import ipdb; ipdb.set_trace()
    # Judge assistant value
    # import ipdb; ipdb.set_trace()

    item['messages'][0]['value'] = item['messages'][0]['value'] + in_context_prompt
    assistant_value = item['messages'][1]['value']
    # Check whether it is "No ... detected"
    if isinstance(assistant_value, str) and assistant_value.strip().lower().startswith("no ") and assistant_value.strip().lower().endswith(" detected."):
        updated_file.append(item)
        continue
    # Check the number of complete <|box|>...<|/box|> pairs
    import re
    box_matches = re.findall(r"<\|box\|>.*?<\|/box\|>", assistant_value)
    box_count = len(box_matches)
    if box_count == 1:
        updated_file.append(item)
        continue
    else:
        print(assistant_value)
        count += 1
print('skip count:', count)
print('updated count:', len(updated_file))



with open(new_path, 'w', encoding='utf-8') as f:
    json.dump(updated_file, f, indent=4, ensure_ascii=False)