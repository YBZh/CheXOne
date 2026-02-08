import json
import csv

raw_data_path = 'draft_report/raw_data_mimic_report/1.csv'
residents_data_path = 'draft_report/responses/reader-1.json'
output_path = 'blind_compare/raw_data_mimic_report/1.csv'


raw_data_path = 'draft_report/raw_data_mimic_report/2.csv'
residents_data_path = 'draft_report/responses/reader-2.json'
output_path = 'blind_compare/raw_data_mimic_report/2.csv'

# raw_data_path = 'draft_report/raw_data_mimic_report/3.csv'
# residents_data_path = 'draft_report/responses/reader-3.json'
# output_path = 'blind_compare/raw_data_mimic_report/3.csv'

raw_data_path = 'draft_report/raw_data_mimic_report/4.csv'
residents_data_path = 'draft_report/responses/reader-4.json'
output_path = 'blind_compare/raw_data_mimic_report/4.csv'

raw_data_path = 'draft_report/raw_data_mimic_report/5.csv'
residents_data_path = 'draft_report/responses/reader-5.json'
output_path = 'blind_compare/raw_data_mimic_report/5.csv'


# Load resident's data (the JSON with AI report responses)
with open(residents_data_path, 'r', encoding='utf-8') as f:
    resident_data = json.load(f)

with open(raw_data_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8', newline='') as f_out:
    reader = csv.DictReader(f_in)
    fieldnames = reader.fieldnames.copy() if reader.fieldnames else []
    # Ensure 'resident' is present in fieldnames exactly once
    if 'resident' not in fieldnames:
        fieldnames.append('resident')
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()
    for row in reader:
        unique_id = row['unique_id']
        resident_entry = resident_data.get(unique_id, {})
        if not resident_entry or 'Write' not in resident_entry:
            raise ValueError(f'unique_id: {unique_id} not found in resident data')
        row_with_resident = row.copy()
        row_with_resident['resident'] = resident_entry['Write'].replace('\n', ' ').replace('\r', ' ')
        writer.writerow(row_with_resident)
