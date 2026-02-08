import json

json_files = [
    "responses/reader-1.json",
    "responses/reader-2.json",
    "responses/reader-3.json",
    "responses/reader-4.json",
    "responses/reader-5.json",
    "responses/reader-6.json",
]
import os

existing_json_files = []
for file in json_files:
    if os.path.exists(file):
        existing_json_files.append(file)
json_files = existing_json_files

# 注意：candidate 顺序应该固定，不能用sorted
candidate = ["candidate", "resident", "Comparable (Equivalent Quality)"]

from collections import Counter, defaultdict

all_counts = defaultdict(Counter)
all_totals = defaultdict(int)

# 由于"Comparable (Equivalent Quality)"的统计有时出现在preferred_source，有时只出现在preferred字段里，
# 我们需要专门处理一下。如果preferred_source不存在且preferred是"Comparable (Equivalent Quality)"，也算上。
for json_file in json_files:
    print(f"Stats for file: {json_file}")
    file_counts = defaultdict(Counter)
    file_totals = defaultdict(int)
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 依次统计
    for sample in data.values():
        # 新增处理：如果preferred_source不存在、但是preferred=="Comparable (Equivalent Quality)"，则也计入
        if "preferred_source" in sample:
            value = sample["preferred_source"]
            # 如果有人写错 preferred_source，且把 "Comparable (Equivalent Quality)" 记在了 preferred 字段里，也补一下
            if value not in candidate and "preferred" in sample and sample["preferred"] == "Comparable (Equivalent Quality)":
                value = "Comparable (Equivalent Quality)"
            file_counts['preferred_source'][value] += 1
            all_counts['preferred_source'][value] += 1
            file_totals['preferred_source'] += 1
            all_totals['preferred_source'] += 1
        else:
            # 如果找不到 preferred_source 但preferred是"Comparable (Equivalent Quality)", 也统计
            if sample.get("preferred") == "Comparable (Equivalent Quality)":
                value = "Comparable (Equivalent Quality)"
                file_counts['preferred_source'][value] += 1
                all_counts['preferred_source'][value] += 1
                file_totals['preferred_source'] += 1
                all_totals['preferred_source'] += 1
            # 其它情况就不计了
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
for category in all_counts:
    print(f"Summary for {category}:")
    total = all_totals[category]
    for resp in candidate:
        count = all_counts[category][resp] if resp in all_counts[category] else 0
        ratio = (count / total) if total > 0 else 0
        print(f"  {resp}: {count} ({ratio:.2%})")
    print()