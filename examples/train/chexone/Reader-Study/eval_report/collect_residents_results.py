import json



attending_editing_json_files = [
    "responses/reader-1.json",
    "responses/reader-2.json",
    "responses/reader-3.json",
    "responses/reader-4.json",
    "responses/reader-5.json",
    "responses/reader-6.json",
]

import os

existing_json_files = []
for file in attending_editing_json_files:
    if os.path.exists(file):
        existing_json_files.append(file)
attending_editing_json_files = existing_json_files


# 根据 key 里面有没有 "residents" 分组统计 attending/residents 结果
# 对如下字段分别统计分布（如 likert-scale 一致的顺序）:
#   1. Why did you make those edits?** (Required)
#   2. Applicability to exam indication:
#   3. Writing efficiency:
#   4. Interpretation efficiency:
#   5. duration

# Import likert-candidate order from reader_study/reasoning_eval/collect_result.py
LIKERT_CANDIDATE = [
    "Strongly Agree",
    "Agree",
    "Neutral/Not Applicable",
    "Disagree",
    "Strongly Disagree",
]
LIKERT_SCORE_MAP = {
    "Strongly Agree": 10,
    "Agree": 5,
    "Neutral/Not Applicable": 0,
    "Disagree": -5,
    "Strongly Disagree": -10,
}

FIELD_NAMES = [
    "**3.1: Why did you make those edits?** (Required)",
    "**3.2: Applicability to exam indication:  The drafted report helps answer the exam indication:** (Required)",
    "**3.3: Writing efficiency:  The drafted report improves report writing efficiency:** (Required)",
    "**3.4: Interpretation efficiency:  The drafted report improves CXR interpretation efficiency:** (Required)",
    "duration",
]

def summarize_field_values(all_values):
    from collections import Counter
    import math
    result = {}
    for k, vlist in all_values.items():
        if k == "duration":
            nums = [float(x) for x in vlist if isinstance(x, (int, float)) or (isinstance(x, str) and x not in ("", None, "NaN"))]
            if nums:
                result[k] = {
                    "mean": sum(nums) / len(nums),
                    "max": max(nums),
                    "min": min(nums),
                    "num": len(nums),
                    "list": nums,
                }
            else:
                result[k] = None
        elif k in (
            "**3.2: Applicability to exam indication:  The drafted report helps answer the exam indication:** (Required)",
            "**3.3: Writing efficiency:  The drafted report improves report writing efficiency:** (Required)",
            "**3.4: Interpretation efficiency:  The drafted report improves CXR interpretation efficiency:** (Required)"
        ):
            counts = Counter(vlist)
            total = sum(counts[cat] for cat in LIKERT_CANDIDATE)
            likert_counts = {cat: counts.get(cat, 0) for cat in LIKERT_CANDIDATE}
            # Calculate average and stddev
            if total > 0:
                # Make expanded list of scores for each value
                all_scores = []
                for cat in LIKERT_CANDIDATE:
                    all_scores.extend([LIKERT_SCORE_MAP[cat]] * counts.get(cat, 0))
                avg = sum(all_scores) / len(all_scores)
                if len(all_scores) > 1:
                    mean = avg
                    var = sum((x - mean) ** 2 for x in all_scores) / len(all_scores)
                    std = math.sqrt(var)
                else:
                    std = 0.0
            else:
                avg = 0.0
                std = 0.0
            result[k] = (likert_counts, total, avg, std)
        elif k == "**3.1: Why did you make those edits?** (Required)":
            # Fixed candidate order from example (file_context_0 snippet)
            FIELD_CANDIDATE = [
                'Both content and style need improvement',
                'No editing needed (good report)',
                '[Content] False / Missing report of a finding in the image',
                '[Style] Poor report writing style',
            ]
            counts = Counter(vlist)
            total = sum(counts[cat] for cat in FIELD_CANDIDATE)
            ordered_counts = {cat: counts.get(cat, 0) for cat in FIELD_CANDIDATE}
            result[k] = (ordered_counts, total)
        else:
            counts = Counter(vlist)
            result[k] = dict(counts)
    return result

def print_attending_stats(groupname, results):
    print(f"=== {groupname} ===")
    for field in FIELD_NAMES:
        if not results.get(field):
            print(f"{field}: No entry.")
            continue
        if field == "duration":
            durinfo = results[field]
            if durinfo:
                print(f"{field}:")
                print(f"  N={durinfo['num']}")
                print(f"  Mean: {durinfo['mean']:.2f} s")
                print(f"  Max:  {durinfo['max']:.2f} s")
                print(f"  Min:  {durinfo['min']:.2f} s")
                # To show all: print(f"  All durations: {', '.join(f'{d:.6f}' for d in durinfo['list'])}")
            else:
                print(f"{field}: No timing data.")
        elif field == "**3.1: Why did you make those edits?** (Required)":
            counts_and_total = results[field]
            if counts_and_total is None:
                print(f"{field}: No entry.")
                continue
            counts, total = counts_and_total
            print(f"{field}:")
            candidate_order = [
                'Both content and style need improvement',
                'No editing needed (good report)',
                '[Content] False / Missing report of a finding in the image',
                '[Style] Poor report writing style',
            ]
            for ans in candidate_order:
                cnt = counts.get(ans, 0)
                prob = (cnt / total) if total > 0 else 0
                print(f"   {repr(ans)}: {cnt} ({prob:.2%})")
        elif field in (
            "**3.2: Applicability to exam indication:  The drafted report helps answer the exam indication:** (Required)",
            "**3.3: Writing efficiency:  The drafted report improves report writing efficiency:** (Required)",
            "**3.4: Interpretation efficiency:  The drafted report improves CXR interpretation efficiency:** (Required)"
        ):
            counts_and_total_avgstd = results[field]
            if counts_and_total_avgstd is None:
                print(f"{field}: No entry.")
                continue
            likert_counts, total, avg, std = counts_and_total_avgstd
            print(f"{field}:")
            for ans in LIKERT_CANDIDATE:
                cnt = likert_counts.get(ans, 0)
                prob = (cnt / total) if total > 0 else 0
                print(f"   {repr(ans)}: {cnt} ({prob:.2%})")
            print(f"   平均分 (Weighted Average Score): {avg:.2f}")
            print(f"   标准差 (Std Dev): {std:.2f}")
        else:
            counts = results[field]
            print(f"{field}:")
            # Just print as-is for other fields
            for answer, count in counts.items():
                print(f"   {repr(answer)}: {count}")
    print()

# 新增：收集所有文件的统计数据（按ChexOne和Res各自合并全局统计）
all_group_values = {
    "ChexOne": {key: [] for key in FIELD_NAMES},
    "residents": {key: [] for key in FIELD_NAMES},
}

for json_file in attending_editing_json_files:
    print(f"\n===== {json_file} =====")
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    group_values = {
        "ChexOne": {key: [] for key in FIELD_NAMES},
        "residents": {key: [] for key in FIELD_NAMES},
    }
    for k, v in data.items():
        group = "residents" if "residents" in k else "ChexOne"
        for field in FIELD_NAMES:
            val = v.get(field, None)
            if val is not None:
                group_values[group][field].append(val)
                all_group_values[group][field].append(val)  # 收集到全局合计
    att_sum = summarize_field_values(group_values["ChexOne"])
    print_attending_stats("ChexOne", att_sum)
    res_sum = summarize_field_values(group_values["residents"])
    print_attending_stats("Residents", res_sum)

# 最终统计，跨所有文件合计
print("\n================== OVERALL (All attending files combined) ==================")
att_sum_all = summarize_field_values(all_group_values["ChexOne"])
print_attending_stats("ChexOne OVERALL", att_sum_all)
res_sum_all = summarize_field_values(all_group_values["residents"])
print_attending_stats("Residents OVERALL", res_sum_all)