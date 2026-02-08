"""
python axis_all_chexagent.py
自己写的demo, 纯串行处理
"""
import json
import os

from rich import print

from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch


def main():
    # Constant
    data_path = "/users/yabin/hallu_project/ms-swift/examples/train/vlm_medical/prepare_testing/json_for_testing/qwen25vl_processed_grounding_onebox_1826.json"
    save_dir = "./chexbench_eval_updated/maira2/"
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    bench = json.load(open(data_path))

    # Load MAIRA-2 model and processor
    model_id = "microsoft/maira-2"
    print(f"Loading model: {model_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(device).eval()

    def _extract_phrase_from_solution(sample: dict) -> str:
        # Prefer extracting phrase from <|ref|>...<|/ref|> in solution
        solution = sample.get("solution") or ""
        ref_start = "<|ref|>"
        ref_end = "<|/ref|>"
        if ref_start in solution and ref_end in solution:
            try:
                start_idx = solution.index(ref_start) + len(ref_start)
                end_idx = solution.index(ref_end, start_idx)
                phrase = solution[start_idx:end_idx].strip()
                if phrase:
                    return phrase
            except Exception:
                pass
        # Fallback: try to parse from first user message after a colon
        messages = sample.get("messages") or []
        if messages and isinstance(messages, list):
            content = messages[0].get("content") or ""
            if ":" in content:
                return content.split(":", 1)[1].strip()
            return content.strip()
        return ""

    # Evaluate phrase grounding on each task
    results = []
    for i in range(len(bench)):
        print(f"Evaluating {i}:")
        sample = bench[i]

        # Load first image
        pil_image = None
        images = sample.get("images") or []
        if isinstance(images, list) and len(images) > 0 and isinstance(images[0], str):
            image_path = images[0]
            try:
                pil_image = Image.open(image_path).convert("RGB")
            except Exception:
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
                alt_path = os.path.join(repo_root, image_path)
                pil_image = Image.open(alt_path).convert("RGB")

        # Extract phrase to ground
        phrase = _extract_phrase_from_solution(sample)

        # Prepare inputs using MAIRA-2 phrase grounding API
        if pil_image is not None and phrase:
            inputs = processor.format_and_preprocess_phrase_grounding_input(
                frontal_image=pil_image,
                phrase=phrase,
                return_tensors="pt",
            )
        else:
            # Fallback to text-only if something is missing (should rarely happen)
            inputs = processor(text=phrase or "", return_tensors="pt")
        inputs = inputs.to(device)
        print(inputs)
        import ipdb; ipdb.set_trace()
        # Generate
        with torch.inference_mode():
            output_decoding = model.generate(
                **inputs,
                max_new_tokens=150,
                use_cache=True,
            )

        # Decode and convert to grounded output
        prompt_length = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        if prompt_length > 0:
            gen_ids = output_decoding[0][prompt_length:]
        else:
            gen_ids = output_decoding[0]
        decoded_text = processor.decode(gen_ids, skip_special_tokens=True)  # type: ignore[attr-defined]
        prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)  # type: ignore[attr-defined]

        # Record results
        sample["response_text"] = decoded_text
        sample["response"] = prediction
        results.append(sample)

    out_path = os.path.join(save_dir, "phrase_grounding_1826_maira2.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)




if __name__ == '__main__':
    main()