# Standardize the VQA dataset for GRPO preparation
import json
import re
    
validation_path = '/home/zhangyabin/project/data/chestxagent_sft_val_all_tasks.json'
validation_file = json.load(open(validation_path, 'r'))

# path = '/home/zhangyabin/project/data/chestxagent_sft_train_all_tasks_Close-Ended_VQA_subset.json'
# new_path = '/home/zhangyabin/project/data/grpo_prepare/vqa_set/chestxagent_sft_train_all_tasks_Close-Ended_VQA_subset_standardized.json' # 356018
# path = '/home/zhangyabin/project/data/chestxagent_sft_train_all_tasks_View_Classification_subset.json'
# new_path = '/home/zhangyabin/project/data/grpo_prepare/vqa_set/chestxagent_sft_train_all_tasks_View_Classification_subset_standardized.json' ## 348297
# path = '/home/zhangyabin/project/data/chestxagent_sft_train_all_tasks_Temporal_Image_Classification_subset.json'
# new_path = '/home/zhangyabin/project/data/grpo_prepare/vqa_set/chestxagent_sft_train_all_tasks_Temporal_Image_Classification_subset_standardized.json' ## 621, single options
path = '/home/zhangyabin/project/data/chestxagent_sft_train_all_tasks_Image_Classification_subset.json'
new_path = '/home/zhangyabin/project/data/grpo_prepare/vqa_set/chestxagent_sft_train_all_tasks_Image_Classification_subset_standardized.json' ## 525688, including 128342 multiple options

print('loading validation file...', path)
selected_file = []
for item in validation_file:
    unique_id = item['unique_id']
    # if 'Close-Ended VQA' in unique_id:
    #     selected_file.append(item)
    # if 'View Classification' in unique_id:
    #     selected_file.append(item)
    # if 'Temporal Image Classification' in unique_id:
    #     selected_file.append(item)
    if 'Image Classification' in unique_id and 'Temporal Image Classification' not in unique_id:
        selected_file.append(item)
# print('selected count:', len(selected_file))

file = json.load(open(path, 'r'))

print('length of selected validation file:', len(selected_file))
print('length of train file:', len(file))
merged_file = selected_file + file
print('length of merged file:', len(merged_file))

multi_option_count = 0


def standardize_options_and_answer(q, a):
    global multi_option_count
    # Match all options (number/letter) and their content
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

    # Replace the Options part - handle cases with and without the "Options:" separator
    q_split = q.split("Options:")
    if len(q_split) == 2:
        # With "Options:" separator
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
        # Without "Options:" separator - directly replace all options
        new_q = q
        for orig_key, value in options:
            orig_key = orig_key.strip()
            if orig_key in option_map:
                mapped_letter = option_map[orig_key]
                # Replace original option format with new letter format
                old_option = f"({orig_key}) {value.strip()}"
                new_option = f"({mapped_letter}) {value.strip()}"
                new_q = new_q.replace(old_option, new_option)

    # Standardize answer - handle various formats of options (number, lowercase letter, uppercase letter)
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
    # If the number of (X) in new_a is greater than 1, print
    num_options = len(re.findall(r"\([A-Z]\)", new_a))
    if num_options > 1:
        multi_option_count += 1
        # return None, None
        # print(f"Multiple options detected in answer: new_a: {new_a}, q: {new_q}")
    # A more elegant way: when writing to json, enforce ensure_ascii=False to ensure all characters (including quotes) are output as original unicode.
    # No need to do any string replacement here, just return directly
    return new_q, new_a

count = 0
updated_file = []
for item in merged_file:
    # import ipdb; ipdb.set_trace()
    # print(item['unique_id'])
    # if 'Temporal Image Classification' in item['unique_id']:
    if 'Image Classification' in item['unique_id'] and 'Temporal Image Classification' not in item['unique_id']:
        messages = item['messages']
        q = messages[0]['value']
        a = messages[1]['value']
        new_q, new_a = standardize_options_and_answer(q, a)
        if new_q is None:
            count += 1
            continue    

        item['messages'][0]['value'] = new_q
        item['messages'][1]['value'] = new_a
        item['solution'] = new_a
        updated_file.append(item)
print('skip count:', count)
print('updated count:', len(updated_file))
print('multi option count:', multi_option_count)


with open(new_path, 'w', encoding='utf-8') as f:
    json.dump(updated_file, f, indent=4, ensure_ascii=False)