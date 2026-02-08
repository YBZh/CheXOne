"""
python axis_all_chexagent.py
自己写的demo, 纯串行处理
"""
import json
import os

from rich import print

from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
import argparse

def evaluate_medgemma(data_path: str, save_path: str, temperature: float = 0.0):
    """
    Run Med-Gemma evaluation for the dataset at data_path.
    Returns a list of task samples with an added 'response' field.
    """
    # Load data
    bench = json.load(open(data_path))

    # Load Med-Gemma model and processor
    # pip install accelerate
    model_id = "google/medgemma-4b-it"
    print(f"Loading model: {model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # Batched evaluation for better GPU utilization
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "500"))
    print(f"Running batched inference with BATCH_SIZE={batch_size}, MAX_NEW_TOKENS={max_new_tokens}")

    results = []
    total = len(bench)
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_samples = [bench[i] for i in range(start_idx, end_idx)]

        # Prepare batch chats (texts + images)
        chats_batch = []
        for task_samples in batch_samples:
            images = task_samples.get("images") or []
            messages = task_samples.get("messages")

            # Build chat with image + prompt
            prompt = messages[0].get("content") if messages else ""

            pil_images = []
            if isinstance(images, list) and len(images) > 0:
                for img_item in images:
                    if isinstance(img_item, str):
                        try:
                            # Prefer relative path loading from current working directory
                            pil = Image.open(img_item).convert("RGB")
                            pil_images.append(pil)
                            continue
                        except Exception:
                            # Fallback: try resolving relative to repo root (script directory 4 levels up)
                            try:
                                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
                                alt_path = os.path.join(repo_root, img_item)
                                pil = Image.open(alt_path).convert("RGB")
                                pil_images.append(pil)
                                continue
                            except Exception:
                                pass
                    elif isinstance(img_item, Image.Image):
                        pil_images.append(img_item)

            chat_messages = []
            chat_messages.append({
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert radiologist."}]
            })
            user_content = [{"type": "text", "text": prompt}]
            for pil in pil_images:
                user_content.append({"type": "image", "image": pil})
            chat_messages.append({"role": "user", "content": user_content})
            chats_batch.append(chat_messages)

        # Tokenize/process batch
        inputs = processor.apply_chat_template(
            chats_batch,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        # Compute input lengths to strip prompt from generations
        pad_token_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(getattr(model, "config", None), "pad_token_id", 0)
        input_ids = inputs["input_ids"]
        input_lengths = (input_ids != pad_token_id).sum(dim=-1).tolist()

        # Generate for the whole batch
        with torch.inference_mode():
            generations = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        # Decode each item in the batch
        for local_idx, task_samples in enumerate(batch_samples):
            gen_seq = generations[local_idx]
            prompt_len = input_lengths[local_idx]
            new_tokens = gen_seq[prompt_len:]
            decoded = processor.decode(new_tokens, skip_special_tokens=True)
            print(decoded)
            task_samples["response"] = decoded
            results.append(task_samples)

        print(f"Processed items {start_idx} - {end_idx - 1} / {total}")

    with open(os.path.join(save_path, os.path.basename(data_path).replace(".json", "_medgemma.json")), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Med-Gemma evaluation")
    parser.add_argument("--data_path", type=str, 
                       default="./examples/train/vlm_medical/prepare_testing/json_for_testing/qwen25vl_processed_grounding_onebox_reasoning_1826.json",
                       help="Path to the test data JSON file")
    parser.add_argument("--save_path", type=str, default="./chexbench_eval_updated/medgemma/",
                       help="Path to save the results")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    # default_data_path = "/users/yabin/hallu_project/ms-swift/examples/train/vlm_medical/prepare_testing/json_for_testing/view_classification_mimic_cxr_900.json"
    evaluate_medgemma(parser.parse_args().data_path, parser.parse_args().save_path, parser.parse_args().temperature)