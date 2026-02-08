"""
ChestXReasoner (Qwen2-VL-based) evaluation script
"""
import json
import os

from rich import print

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, GenerationConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import argparse
import ipdb

def evaluate_qwen2vl(data_path: str, save_path: str, temperature: float):
    """
    Run Qwen2-VL evaluation for the dataset at data_path.
    Returns a list of task samples with an added 'response' field.
    """
    # Load data
    bench = json.load(open(data_path))

    # Load Qwen2-VL model and processor
    model_id = "byrLLCC/ChestX-Reasoner"
    print(f"Loading model: {model_id}")
    # Optional: enable flash_attention_2 via env: ATTN_IMPL=flash_attention_2
    attn_impl = os.environ.get("ATTN_IMPL", "").strip()
    attn_kwargs = {}
    if attn_impl:
        attn_kwargs["attn_implementation"] = attn_impl
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        **attn_kwargs,
    )
    processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct')
    # Ensure left padding for decoder-only generation
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"

    # Explicit generation config to ensure sampling args are honored
    gen_config = GenerationConfig.from_model_config(model.config)
    # Allow overriding via env; fall back to reasonable sampling defaults
    gen_config.do_sample = False
    gen_config.temperature = float(os.environ.get("TEMPERATURE", "0.0"))
    gen_config.top_p = float(os.environ.get("TOP_P", "0.9"))
    gen_config.top_k = int(os.environ.get("TOP_K", "50"))
    gen_config.repetition_penalty = float(os.environ.get("REPETITION_PENALTY", "1.0"))

    # Batched evaluation for better GPU utilization
    batch_size = int(os.environ.get("BATCH_SIZE", "8"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
    print(f"Running batched inference with BATCH_SIZE={batch_size}, MAX_NEW_TOKENS={max_new_tokens}")

    results = []
    total = len(bench)
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch_samples = [bench[i] for i in range(start_idx, end_idx)]

        # Prepare batch chats (texts + images) according to Qwen2-VL chat template
        chats_batch = []
        for task_samples in batch_samples:
            images = task_samples.get("images") or []
            messages = task_samples.get("messages")

            # Build chat with image + prompt
            prompt = messages[0].get("content") if messages else ""
                        # Remove "Please reason step by step" and all text after it if present
            import re
            m = re.search(r'(.*?)Please reason step by step', prompt, re.DOTALL)
            if m:
                prompt = m.group(1).rstrip() ##+ "\nPlease answer the question with the option's letter/number from the given choices."
            else:
                prompt = prompt

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
            chat_messages.append({
                "role": "system",
                "content": [{"type": "text", "text": (
                    "The user asks a question, and the Assistant solves it.\n"
                    "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
                    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively.\n"
                )}]
            })
            user_content = [{"type": "text", "text": prompt}]
            for pil in pil_images:
                user_content.append({"type": "image", "image": pil})
            chat_messages.append({"role": "user", "content": user_content})
            chats_batch.append(chat_messages)

        # Prepare batched processor inputs per Qwen2-VL demo (texts + processed vision inputs)
        texts = []
        images_batch = []
        videos_batch = []
        for chat in chats_batch:
            text = processor.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(chat)
            texts.append(text)
            images_batch.append(image_inputs)
            videos_batch.append(video_inputs)

        # Only pass videos when at least one sample contains video frames
        any_videos = any(isinstance(v, (list, tuple)) and len(v) > 0 for v in videos_batch)
        if any_videos:
            inputs = processor(
                text=texts,
                images=images_batch,
                videos=videos_batch,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
        else:
            inputs = processor(
                text=texts,
                images=images_batch,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

        # Generate for the whole batch
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                generation_config=gen_config,
                max_new_tokens=max_new_tokens,
            )

        # Trim prompts and decode each item in the batch
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
        # import ipdb; ipdb.set_trace()
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, os.path.basename(data_path).replace(".json", "_chestxreasoner.json")), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ChestXReasoner (Qwen2-VL) evaluation")
    parser.add_argument("--data_path", type=str, 
                        default="./examples/train/chexone/prepare_testing/json_for_testing/mimic-cxr-lt_test_reasoning_750.json",
                        help="Path to the test data JSON file")
    parser.add_argument("--save_path", type=str, default="./chexbench_eval_updated/chestxreasoner/",
                        help="Path to save the results")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    evaluate_qwen2vl(parser.parse_args().data_path, parser.parse_args().save_path, parser.parse_args().temperature)