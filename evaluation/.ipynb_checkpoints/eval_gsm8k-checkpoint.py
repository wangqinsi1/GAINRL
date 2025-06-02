import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ──────────────────────────────  CLI  ──────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a model on GSM8K with VLLM and report accuracy."
    )
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Model checkpoint directory or HuggingFace model name",
    )
    p.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="JSON file containing GSM8K evaluation data "
        "(list of dicts with keys `problem`, `solution`)",
    )
    p.add_argument("--batch_size", type=int, default=4, help="Batch size")
    p.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Limit evaluation to first N samples (None = all)",
    )
    p.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.3,
        help="Portion of a single GPU memory for VLLM (0–1)",
    )
    p.add_argument(
        "--save_results",
        action="store_true",
        help="Save per‑example outputs + metrics to a timestamped JSON",
    )
    return p


# ────────────────────  Prompt & answer helpers  ────────────────────
R1_STYLE_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it.\n"
    "The assistant first thinks about the reasoning process in the mind and then "
    "provides the user with the answer. Let's think step by step and output the "
    "final answer within \\boxed{}."
)


def last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i, right_brace_idx, n_left = idx, None, 0
    while i < len(string):
        if string[i] == "{":
            n_left += 1
        if string[i] == "}":
            n_left -= 1
            if n_left == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def remove_boxed(s: str | None) -> str:
    if not s:
        return ""
    if s.startswith("\\boxed "):
        return s[len("\\boxed ") :]
    left = "\\boxed{"
    return s[len(left) : -1] if s.startswith(left) and s.endswith("}") else s


def safe_float_equal(a, b, tol=1e-5) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (ValueError, TypeError):
        return False


def extract_xml_answer(text: str) -> str:
    """Our fine‑tuned model boxes answers in \\boxed{}."""
    return remove_boxed(last_boxed_only_string(text))


def extract_hash_answer(text: str) -> str | None:
    """Ground‑truth in GSM8K uses '#### <answer>'."""
    try:
        return text.split("####")[1].strip()
    except IndexError:
        return None


# ───────────────────────────  Evaluation  ──────────────────────────
def evaluate_model(
    model_path: str,
    dataset_path: str,
    batch_size: int,
    num_samples: int | None,
    gpu_memory_utilization: float,
    save_results: bool,
) -> Dict:
    logging.getLogger("vllm").setLevel(logging.WARNING)
    print("Initializing evaluation...")

    # Load model & tokenizer with small progress
    with tqdm(total=2, desc="Loading model components") as pbar:
        llm = LLM(
            model=model_path,
            dtype="half",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=768,
            device="cuda:0",
            enable_chunked_prefill=True,
        )
        pbar.update(1)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=768,
            padding_side="right",
            truncation_side="right",
        )
        pbar.update(1)

    # Sampling config
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    # Load dataset
    print(f"Loading dataset from {dataset_path} ...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    dataset = Dataset.from_list(raw_data)
    if num_samples:
        dataset = dataset.select(range(num_samples))
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples")

    # Prepare for evaluation loop
    correct, total = 0, 0
    results: List[Dict] = []
    progress_bar = tqdm(
        total=total_samples,
        desc="Processing samples",
        unit="examples",
        dynamic_ncols=True,
    )
    progress_bar.set_postfix(acc="0.00%", correct="0")

    # Batch inference
    for start in range(0, total_samples, batch_size):
        batch = dataset[start : start + batch_size]
        bs = len(batch["problem"])

        # Build chat prompts
        prompts = [
            [
                {"role": "system", "content": R1_STYLE_SYSTEM_PROMPT},
                {"role": "user", "content": "What is 2+2?"},
                {
                    "role": "assistant",
                    "content": (
                        "To calculate 2+2, we simply add the numbers together: "
                        "2 + 2 = 4. The answer is \\boxed{4}"
                    ),
                },
                {"role": "user", "content": q.strip()},
            ]
            for q in batch["problem"]
        ]
        formatted_prompts = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]

        # Generate
        outputs = llm.generate(formatted_prompts, sampling_params)

        for j, out in enumerate(outputs):
            response = out.outputs[0].text
            pred_ans = extract_xml_answer(response)
            true_ans = extract_hash_answer(batch["solution"][j])
            is_correct = safe_float_equal(pred_ans, true_ans)

            results.append(
                {
                    "question": batch["problem"][j],
                    "true_answer": true_ans,
                    "generated_answer": pred_ans,
                    "full_response": response,
                    "correct": is_correct,
                }
            )

            correct += int(is_correct)
            total += 1

        # Update progress bar
        progress_bar.update(bs)
        progress_bar.set_postfix(
            acc=f"{(correct/total)*100:.2f}%", correct=f"{correct}/{total}"
        )

    progress_bar.close()

    metrics = {
        "accuracy": correct / total if total else 0,
        "correct": correct,
        "total": total,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
    }

    if save_results:
        fname = f"gsm8k_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(fname, "w") as fp:
            json.dump({"metrics": metrics, "results": results}, fp, indent=2)
        print(f"\nResults saved to {fname}")

    return metrics


# ──────────────────────────────  MAIN  ─────────────────────────────
if __name__ == "__main__":
    args = build_parser().parse_args()
    metrics = evaluate_model(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        gpu_memory_utilization=args.gpu_memory_utilization,
        save_results=args.save_results,
    )

    print("\nFinal Evaluation Results")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Correct:  {metrics['correct']}/{metrics['total']}")
