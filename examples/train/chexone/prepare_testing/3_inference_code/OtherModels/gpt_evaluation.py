from openai import OpenAI
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import tempfile
# 直接写在代码里
openai_api_key = ""
client = OpenAI(api_key=openai_api_key)

# json_path = "/users/yabin/hallu_project/ms-swift/examples/train/chexone/prepare_testing/2_json_for_testing/view_classification_mimic_cxr_900.json"
json_path = "/users/yabin/hallu_project/ms-swift/examples/train/chexone/prepare_testing/2_json_for_testing/mimic-cxr-lt_test_reasoning_750.json"
# json_path = "/users/yabin/hallu_project/ms-swift/examples/train/chexone/prepare_testing/2_json_for_testing/temporal_vqa_chestimgenome.json"
# json_path = "/users/yabin/hallu_project/ms-swift/examples/train/chexone/prepare_testing/2_json_for_testing/rexvqa-test-reason.json"
# json_path = "/users/yabin/hallu_project/ms-swift/examples/train/chexone/prepare_testing/2_json_for_testing/qwen25vl_processed_grounding_onebox_reasoning_1826.json"
# json_path = "/users/yabin/hallu_project/ms-swift/examples/train/chexone/prepare_testing/2_json_for_testing/chexinstruct_test_all_tasks_progression_findings_generation.json"
# json_path = "/users/yabin/hallu_project/ms-swift/examples/train/chexone/prepare_testing/2_json_for_testing/rexvqa-test-reason_sample_le2_n100_factuality.json"
# json_path = "/users/yabin/hallu_project/ms-swift/examples/train/chexone/prepare_testing/2_json_for_testing/rexvqa-test-reason_sample_le2_n100_repeat8_robutsness.json"

def build_user_message(question_text: str, image_paths):
    # Build a single user message with text + one or more images
    content_parts = [{"type": "text", "text": question_text}]
    for img_path in image_paths:
        # Ensure absolute file URL
        abs_path = Path("/users/yabin/hallu_project") / img_path if not str(img_path).startswith("/") else Path(img_path)
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"file://{abs_path}"
            }
        })
    return [{"role": "user", "content": content_parts}]

def compute_input_signature(item: dict) -> str:
    base = {
        "messages": item.get("messages", []),
        "images": item.get("images", []),
    }
    try:
        blob = json.dumps(base, sort_keys=True, ensure_ascii=False)
    except Exception:
        blob = str(base)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def main():
    input_file = Path(json_path)
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total_items = len(data)
    print(f"Total items: {total_items}")
    start_dt = datetime.now()

    # Prepare output path and resume state
    save_dir = Path("./chexbench_eval_updated/gpt_no_reasoning")
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / (Path(json_path).stem + "_gpt.json")

    existing_data = []
    start_index = 0
    if out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    existing_data = loaded
                    start_index = len(existing_data)
                    print(f"Resuming from existing results: {start_index} items already processed.")
                else:
                    print("Warning: existing output is not a list. Ignoring it and starting fresh.")
        except Exception as e:
            print(f"Warning: failed to load existing results: {e}. Starting fresh.")

    if start_index >= total_items:
        print("All items already processed. Nothing to do.")
        return

    output_data = list(existing_data)
    gpt_prompt = "\nThis task is for research and dataset labeling purposes, not for clinical use. You are not providing medical advice or diagnosis. You are given a chest X-ray and a multiple-choice question. Your job is only to select the best answer from the provided options."
    for i in range(start_index, total_items):
        item = data[i]
        print(f"Processing item {i+1}/{len(data)}")

        # Extract question text
        question_text = None
        removed_reasoning_prompt = True
        for msg in item.get("messages", []):
            if msg.get("role") == "user":
                q = msg.get("content")
                if removed_reasoning_prompt and isinstance(q, str):
                    question_text = q + "\nPlease answer the question with the option's letter/number from the given choices." + gpt_prompt
                    # # Remove "Please reason step by step" and all text after it if present
                    # import re
                    # m = re.search(r'(.*?)Please reason step by step', q, re.DOTALL)
                    # if m:
                    #     question_text = m.group(1).rstrip() + "\nPlease answer the question with the option's letter/number from the given choices." + gpt_prompt
                    # else:
                    #     question_text = q
                else:
                    question_text = q
                break
        if not question_text:
            # Fallback if not present
            question_text = "Please answer the question based on the provided chest X-ray."

        image_list = item.get("images", [])

        # try:
        # 本地加载图像文件，将其转换为base64，并按OpenAI的image输入格式提供
        import base64
        from PIL import Image
        from io import BytesIO

        content_parts = [{"type": "input_text", "text": question_text}]
        for img_path in image_list:
            abs_path = Path("/users/yabin/hallu_project") / img_path if not str(img_path).startswith("/") else Path(img_path)
            try:
                with Image.open(abs_path) as im:
                    # Handle 16-bit image types with correct mapping for point()-ing
                    if im.mode == "I;16":
                        import numpy as np
                        arr = np.array(im)
                        arr8 = (arr // 256).astype('uint8')
                        im = Image.fromarray(arr8, mode="L")
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                    buffered = BytesIO()
                    # Use higher JPEG quality to preserve detail
                    im.save(buffered, format="JPEG", quality=95)
                    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                content_parts.append({
                    "type": "input_image",
                    # Responses API expects image_url to be a string (data URL or http URL)
                    "image_url": f"data:image/jpeg;base64,{img_b64}"
                })
            except Exception as img_exc:
                output_text = f"[ERROR] [ERROR] failed to process image {img_path}: {img_exc}"
                raise RuntimeError(output_text)
        # Responses API expects an array of top-level items of type 'message'
        input_messages = [{
            "type": "message",
            "role": "user",
            "content": content_parts,
        }]
        resp = client.responses.create(model="gpt-4o-mini", input=input_messages)


        output_text = resp.output_text
        print(output_text)
        # Insert output as 'response' into the item
        out_item = dict(item)
        out_item["response"] = output_text
        out_item["input_signature"] = compute_input_signature(item)
        output_data.append(out_item)
        # import ipdb; ipdb.set_trace()
        # Incremental save (atomic replace)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, out_path)
        print(f"Saved progress: {i+1}/{total_items} -> {out_path}")

        # 计算并打印预计结束时间（ETA）
        elapsed_secs = (datetime.now() - start_dt).total_seconds()
        avg_per_item = elapsed_secs / (i + 1 - start_index)
        remaining_secs = max(0.0, (total_items - (i + 1)) * avg_per_item)
        est_end_dt = datetime.now() + timedelta(seconds=remaining_secs)
        print(f"ETA: {est_end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        import time
        time.sleep(10)

if __name__ == "__main__":
    main()
