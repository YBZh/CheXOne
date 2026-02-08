"""
Qwen3-VL evaluation script
"""
import json
import os

from rich import print

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image
import torch
import argparse

def evaluate_qwen3vl(data_path: str, save_path: str, temperature: float):
    """
    Run Qwen3-VL evaluation for the dataset at data_path.
    Returns a list of task samples with an added 'response' field.
    """
    # Load data
    bench = json.load(open(data_path))

    # Load Qwen3-VL model and processor
    # model_id = "Qwen/Qwen3-VL-4B-Thinking"
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    print(f"Loading model: {model_id}")
    # We recommend enabling flash_attention_2 for better acceleration and memory saving.
    attn_impl = os.environ.get("ATTN_IMPL", "").strip()
    attn_kwargs = {}
    if attn_impl:
        attn_kwargs["attn_implementation"] = attn_impl
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        **attn_kwargs,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    # Ensure left padding for decoder-only generation
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"

    # Batched evaluation for better GPU utilization
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
    print(f"Running batched inference with BATCH_SIZE={batch_size}, MAX_NEW_TOKENS={max_new_tokens}")

    results = []
    total = len(bench)
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_samples = [bench[i] for i in range(start_idx, end_idx)]

        # Prepare batch chats (texts + images) according to Qwen3-VL chat template
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

        # Tokenize/process batch using Qwen3-VL processor chat template
        inputs = processor.apply_chat_template(
            chats_batch,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        # Generate for the whole batch
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                # temperature=temperature,
                # follow Qwen demo defaults; customize via HF generation kwargs if needed
            )

        # Trim prompts and decode each item in the batch following Qwen demo
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        decoded_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        for local_idx, task_samples in enumerate(batch_samples):
            decoded = decoded_texts[local_idx]
            print(decoded)
            task_samples["response"] = decoded
            results.append(task_samples)

        print(f"Processed items {start_idx} - {end_idx - 1} / {total}")

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, os.path.basename(data_path).replace(".json", "_qwen3vl.json")), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Qwen3-VL evaluation")
    parser.add_argument("--data_path", type=str, 
                    #    default="./examples/train/vlm_medical/prepare_testing/json_for_testing/qwen25vl_processed_grounding_onebox_reasoning_1826.json",
                        default="./examples/train/vlm_medical/prepare_testing/json_for_testing/chexinstruct_test_all_tasks_progression_findings_generation.json",
                       help="Path to the test data JSON file")
    parser.add_argument("--save_path", type=str, default="./chexbench_eval_updated/qwen3vl_thinking/",
                       help="Path to save the results")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    # default_data_path = "/users/yabin/hallu_project/ms-swift/examples/train/vlm_medical/prepare_testing/json_for_testing/view_classification_mimic_cxr_900.json"
    evaluate_qwen3vl(parser.parse_args().data_path, parser.parse_args().save_path, parser.parse_args().temperature)