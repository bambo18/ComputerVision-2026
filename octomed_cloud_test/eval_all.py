import json
import re
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def normalize(value):
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")

    model_id = "OctoMed/OctoMed-7B"
    dataset_id = "OctoMed/SLAKE"
    max_new_tokens = 512
    started = time.perf_counter()

    output_dir = Path("outputs") / datetime.now().strftime(
        "evalall_%Y%m%d_%H%M%S"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    print("Output directory:", output_dir.resolve(), flush=True)

    data = load_dataset(dataset_id, split="test")
    count = len(data)

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=262144, max_pixels=262144
    )

    print("Loading model...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    ).eval()

    print("Device map:", model.hf_device_map, flush=True)

    correct_count = 0
    truncated_count = 0
    inference_times = []
    peak_memory_gib = 0.0

    with (output_dir / "predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as result_file:
        for index in range(count):
            sample = data[index]
            image = sample["image"].convert("RGB")
            question = sample["question"]
            reference = str(sample["answer"])

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }]
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            sample_started = time.perf_counter()

            inputs = processor(
                text=[prompt],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            new_tokens = output[:, inputs["input_ids"].shape[1]:]
            prediction = processor.batch_decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - sample_started
            memory_gib = (
                torch.cuda.max_memory_allocated() / 1024**3
            )
            generated_tokens = int(new_tokens.shape[1])
            # 保守적으로 토큰 한도에 도달한 출력을 표시한다.
            limit_reached = generated_tokens >= max_new_tokens
            matched = normalize(prediction) == normalize(reference)

            correct_count += int(matched)
            truncated_count += int(limit_reached)
            inference_times.append(elapsed)
            peak_memory_gib = max(peak_memory_gib, memory_gib)

            record = {
                "index": index,
                "question": question,
                "reference_answer": reference,
                "prediction_raw": prediction,
                "answer_type": sample.get("answer_type"),
                "normalized_exact_match": matched,
                "inference_seconds": elapsed,
                "generated_tokens": generated_tokens,
                "generation_limit_reached": limit_reached,
                "peak_gpu_allocated_gib": memory_gib,
            }
            result_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            result_file.flush()

            average = sum(inference_times) / len(inference_times)
            remaining_minutes = (
                (count - index - 1) * average / 60
            )
            print(
                f"[{index + 1}/{count}] "
                f"{'MATCH' if matched else 'MISMATCH'} | "
                f"{elapsed:.2f}s | "
                f"EM {correct_count / (index + 1):.1%} | "
                f"ETA {remaining_minutes:.1f} min",
                flush=True,
            )
            print(
                f"  Reference: {reference}\n"
                f"  Prediction: {prediction[:300]}",
                flush=True,
            )

            del inputs, output, new_tokens

    summary = {
        "model": model_id,
        "dataset": dataset_id,
        "split": "test",
        "selection": "all test rows",
        "num_samples": count,
        "metric": "casefold + whitespace normalized exact match",
        "correct": correct_count,
        "normalized_exact_match_percent": (
            100 * correct_count / count if count else None
        ),
        "average_inference_seconds": (
            sum(inference_times) / count if count else None
        ),
        "inference_total_seconds": sum(inference_times),
        "total_seconds_including_loading": (
            time.perf_counter() - started
        ),
        "generation_limit_reached_count": truncated_count,
        "peak_gpu_allocated_gib": peak_memory_gib,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "dtype": str(dtype),
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "min_pixels": 262144,
        "max_pixels": 262144,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summarize_by_type(output_dir)

    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved:", output_dir.resolve(), flush=True)



def summarize_by_type(output_dir):
    from collections import Counter, defaultdict

    def token_f1(prediction, reference):
        # Same case/whitespace normalization as EM.
        # Punctuation and articles are retained.
        predicted = normalize(prediction).split()
        expected = normalize(reference).split()
        if not predicted or not expected:
            return float(predicted == expected)
        overlap = sum(
            (Counter(predicted) & Counter(expected)).values()
        )
        return 2 * overlap / (len(predicted) + len(expected))

    groups = defaultdict(list)
    raw_types = Counter()
    mismatches = []

    with (output_dir / "predictions.jsonl").open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            record = json.loads(line)
            raw_type = record.get("answer_type")
            raw_types[str(raw_type)] += 1
            kind = str(raw_type).strip().casefold()

            if kind in {"open", "open-ended", "open_ended"}:
                group = "open"
            elif kind in {"closed", "close", "closed-ended", "closed_ended"}:
                group = "closed"
            else:
                group = "unknown"

            record["token_f1"] = token_f1(
                record["prediction_raw"],
                record["reference_answer"],
            )
            groups["all"].append(record)
            groups[group].append(record)

            if not record["normalized_exact_match"]:
                mismatches.append(record)

    report = {
        "note": (
            "Exploratory evaluation, not official paper reproduction. "
            "EM and token F1 use casefold and whitespace normalization. "
            "Tokenization uses whitespace; punctuation/articles retained."
        ),
        "raw_answer_type_counts": dict(raw_types),
        "groups": {},
    }

    for name in ("all", "open", "closed", "unknown"):
        rows = groups[name]
        count = len(rows)
        matched = sum(
            int(row["normalized_exact_match"]) for row in rows
        )
        report["groups"][name] = {
            "count": count,
            "exact_match_correct": matched,
            "exact_match_percent": (
                100 * matched / count if count else None
            ),
            "mean_token_f1_percent": (
                100 * sum(row["token_f1"] for row in rows) / count
                if count else None
            ),
            "average_inference_seconds": (
                sum(row["inference_seconds"] for row in rows) / count
                if count else None
            ),
            "generation_limit_reached_count": sum(
                int(row["generation_limit_reached"]) for row in rows
            ),
        }

    (output_dir / "metrics_by_type.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "mismatches.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for record in mismatches:
            handle.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print("\n=== METRICS BY TYPE ===", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
