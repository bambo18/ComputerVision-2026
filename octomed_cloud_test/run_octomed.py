"""OctoMed + SLAKE: cloud GPU inference smoke test (not benchmark reproduction)."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    if args.index < 0 or args.max_new_tokens < 1:
        parser.error("index must be >= 0 and max-new-tokens must be >= 1")

    import torch
    from datasets import load_dataset
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. Run on the cloud GPU with CUDA-enabled PyTorch.")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    model_id = "OctoMed/OctoMed-7B"
    data_id = "OctoMed/SLAKE"
    data = load_dataset(data_id, split="test")
    if args.index >= len(data):
        parser.error(f"index must be less than {len(data)}")
    sample = data[args.index]
    image = sample["image"].convert("RGB")
    question = sample["question"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"sample_{args.index}.png"
    image.save(image_path)
    print("Question:", question, flush=True)
    print("Loading model...", flush=True)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=262144, max_pixels=262144
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, dtype=dtype, device_map="auto", attn_implementation="sdpa"
    ).eval()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[prompt], images=[image], padding=True, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
        )
    new_tokens = output[:, inputs["input_ids"].shape[1]:]
    prediction = processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    limit_reached = new_tokens.shape[1] >= args.max_new_tokens
    result = {
        "model": model_id, "dataset": data_id, "split": "test",
        "index": args.index, "question": question,
        "reference_answer": sample["answer"], "prediction_raw": prediction,
        "answer_type": sample.get("answer_type"),
        "generated_tokens": new_tokens.shape[1],
        "generation_limit_reached": limit_reached,
        "max_new_tokens": args.max_new_tokens, "do_sample": False,
        "image_path": str(image_path),
    }
    result_path = output_dir / f"result_{args.index}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Reference:", sample["answer"])
    print("Model output:", prediction)
    print("Saved:", result_path.resolve())
    if limit_reached:
        print("Token limit reached; increase --max-new-tokens if the answer is incomplete.")


if __name__ == "__main__":
    main()
