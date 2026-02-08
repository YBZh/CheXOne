import json
import csv

# raw_data_path = 'draft_report/raw_data_mimic_report/1.csv'
# residents_data_path = 'draft_report/responses/reader-1.json'
# # 打开原csv，准备写新csv文件（11.csv）
# output_path = 'eval_report/raw_data_mimic_report/11.csv'


# raw_data_path = 'draft_report/raw_data_mimic_report/2.csv'
# residents_data_path = 'draft_report/responses/reader-2.json'
# output_path = 'eval_report/raw_data_mimic_report/12.csv'

# raw_data_path = 'draft_report/raw_data_mimic_report/3.csv'
# residents_data_path = 'draft_report/responses/reader-3.json'
# output_path = 'eval_report/raw_data_mimic_report/13.csv'

# raw_data_path = 'draft_report/raw_data_mimic_report/4.csv'
# residents_data_path = 'draft_report/responses/reader-4.json'
# output_path = 'eval_report/raw_data_mimic_report/14.csv'

raw_data_path = 'draft_report/raw_data_mimic_report/5.csv'
residents_data_path = 'draft_report/responses/reader-5.json'
output_path = 'eval_report/raw_data_mimic_report/15.csv'


# Load resident's data (the JSON with AI report responses)
with open(residents_data_path, 'r', encoding='utf-8') as f:
    resident_data = json.load(f)


with open(raw_data_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8', newline='') as f_out:
    reader = csv.DictReader(f_in)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()
    import random
    for row in reader:
        unique_id = row['unique_id']
        # 获取该行对应的 resident write
        resident_entry = resident_data.get(unique_id, {})
        if not resident_entry or 'Write' not in resident_entry:
            raise ValueError(f'unique_id: {unique_id} not found in resident data')
        # 构建新的一行
        new_row = row.copy()
        new_row['unique_id'] = unique_id + ' residents'
        new_row['candidate'] = resident_entry['Write']
        rows_to_write = [row, new_row]
        random.shuffle(rows_to_write)
        for r in rows_to_write:
            writer.writerow(r)
