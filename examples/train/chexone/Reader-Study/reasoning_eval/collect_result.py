import json

json_files = [
    "responses/reader-1.json",
    "responses/reader-2.json",
    "responses/reader-3.json",
    "responses/reader-4.json",
    "responses/reader-5.json",
]
import os

existing_json_files = []
for file in json_files:
    if os.path.exists(file):
        existing_json_files.append(file)
json_files = existing_json_files

# 注意：candidate 顺序应该固定，不能用sorted
candidate = ["Strongly Agree", "Agree", "Neutral/Not Applicable", "Disagree", "Strongly Disagree"]

from collections import Counter, defaultdict

all_counts = defaultdict(Counter)
all_totals = defaultdict(int)

for json_file in json_files:
    print(f"Stats for file: {json_file}")
    file_counts = defaultdict(Counter)
    file_totals = defaultdict(int)
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # for each entry, count values for 'factuality' and 'causal_support'
    for sample in data.values():
        for k in ['factuality', 'causal_support']:
            if k in sample:
                file_counts[k][sample[k]] += 1
                all_counts[k][sample[k]] += 1
                file_totals[k] += 1
                all_totals[k] += 1
    for category in file_counts:
        print(f"  {category}:")
        total = file_totals[category]
        for resp in candidate:
            count = file_counts[category][resp] if resp in file_counts[category] else 0
            ratio = (count / total) if total > 0 else 0
            print(f"    {resp}: {count} ({ratio:.2%})")
    print()

# 汇总全部文件的统计信息
print("===== 汇总所有文件统计结果 =====")
# 对应的分数
score_map = {
    "Strongly Agree": 10,
    "Agree": 5,
    "Neutral/Not Applicable": 0,
    "Disagree": -5,
    "Strongly Disagree": -10,
}
for category in all_counts:
    print(f"Summary for {category}:")
    total = all_totals[category]
    weighted_sum = 0.0
    for resp in candidate:
        count = all_counts[category][resp] if resp in all_counts[category] else 0
        ratio = (count / total) if total > 0 else 0
        print(f"  {resp}: {count} ({ratio:.2%})")
        weighted_sum += ratio * score_map[resp]
    avg_score = weighted_sum if total > 0 else 0
    print(f"  加权平均分 (Weighted Average Score): {avg_score:.2f}")
    print()