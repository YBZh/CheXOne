import json

resident_draft_json_files = [
    "../draft_report/responses/reader-1.json",
    "../draft_report/responses/reader-2.json",
    "../draft_report/responses/reader-3.json",
    "../draft_report/responses/reader-4.json",
    "../draft_report/responses/reader-5.json",
    "../draft_report/responses/reader-6.json",
]

resident_editing_json_files = [
    "responses/reader-1.json",
    "responses/reader-2.json",
    "responses/reader-3.json",
    "responses/reader-4.json",
    "responses/reader-5.json",
    "responses/reader-6.json",
]

import os

existing_json_files = []
for file in resident_draft_json_files:
    if os.path.exists(file):
        existing_json_files.append(file)
resident_draft_json_files = existing_json_files

existing_json_files = []
for file in resident_editing_json_files:
    if os.path.exists(file):
        existing_json_files.append(file)
resident_editing_json_files = existing_json_files

# 统计 duration 字段，输出均值、最大、最小，并且把所有的数值 print 出来，方便画散点图，draft 和 editing 分开统计

def compute_stats(durations):
    if not durations:
        return {"mean": None, "max": None, "min": None, "num": 0}
    return {
        "mean": sum(durations) / len(durations),
        "max": max(durations),
        "min": min(durations),
        "num": len(durations),
    }

def print_stats(name, stats):
    print(f"{name}:")
    if stats["num"] == 0:
        print("  No entries.")
    else:
        print(f"  N={stats['num']}")
        print(f"  Mean: {stats['mean']:.2f} s")
        print(f"  Max:  {stats['max']:.2f} s")
        print(f"  Min:  {stats['min']:.2f} s")
    print()

for tag, filelist in [
    ("Draft", resident_draft_json_files),
    ("Editing", resident_editing_json_files)
]:
    all_durations = []
    print(f"==== {tag} duration statistics by file ====")
    for json_file in filelist:
        durations = []
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for v in data.values():
            if isinstance(v, dict) and "duration" in v:
                try:
                    durations.append(float(v["duration"]))
                except Exception:
                    pass
        stats = compute_stats(durations)
        print_stats(f"File: {json_file}", stats)
        print(f"  All durations for {json_file}:")
        if durations:
            print("   " + ", ".join(f"{d:.6f}" for d in durations))
        else:
            print("   (none)")
        print()
        all_durations.extend(durations)
    print(f"==== {tag} overall ====")
    stats = compute_stats(all_durations)
    print_stats(f"{tag} overall", stats)
    print(f"  All durations for {tag} overall:")
    if all_durations:
        print("   " + ", ".join(f"{d:.6f}" for d in all_durations))
    else:
        print("   (none)")
    print()