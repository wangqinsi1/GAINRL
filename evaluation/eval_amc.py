#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate a model on the AMC benchmark with VLLM.

Example
-------
python eval_amc.py \
    --model_path Qwen/Qwen2.5-Math-7B \
    --dataset_path amc_eval.json \
    --batch_size 4 \
    --gpu_memory_utilization 0.35 \
    --save_results
"""

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

# ──────────────────── silence VLLM logs ────────────────────
logging.getLogger("vllm").setLevel(logging.WARNING)

# ──────────────────── prompt helpers ────────────────────
R1_STYLE_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it.\n"
    "The assistant first thinks about the reasoning process in the mind and then "
    "provides the user with the answer. Let's think step by step and output the "
    "final answer within \\boxed{}."
)

def last_boxed_only_string(text: str):
    idx = text.rfind("\\boxed")
    if "\\boxed " in text:
        return "\\boxed " + text.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    i, rb, depth = idx, None, 0
    while i < len(text):
        if text[i] == "{":
            depth += 1
        if text[i] == "}":
            depth -= 1
            if depth == 0:
                rb = i
                break
        i += 1
    return None if rb is None else text[idx : rb + 1]

def remove_boxed(s: str | None) -> str:
    if not s:
        return ""
    if s.startswith("\\boxed "):
        return s[len("\\boxed ") :]
    left = "\\boxed{"
    return s[len(left) : -1] if s.startswith(left) and s.endswith("}") else s

def extract_xml_answer(text: str) -> str:
    return remove_boxed(last_boxed_only_string(text))

def safe_float_equal(a, b, tol: float = 1e-5) -> bool:
    """Try numeric equality fallback to string equality."""
    try:
        return abs(float(a) - float(b)) <= tol
    except (ValueError, TypeError):
        return str(a).strip() == str(b).strip()

# ──────────────────── core evaluation ────────────────────
def evaluate_model(
    model_path: str,
    dataset_path: str,
    batch_size: int,
    num_samples: int | None,
    gpu_memory_utilization: float,
    save_results: bool,
) -> Dict:
    print("Initializing evaluation...")

    # 1) load model & tokenizer
    with tqdm(total=2, desc="Loading model components") as pbar:
        llm = LLM(
            model=model_path,
            dtype="half",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=1024,
            device="cuda:0",
            enable_chunked_prefill=True,
        )
        pbar.update(1)

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=2048,
            padding_side="right",
            truncation_side="right",
        )
        pbar.update(1)

    # 2) sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=3000,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    # 3) load dataset
    print(f"Loading dataset from {dataset_path} ...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    dataset = Dataset.from_list(raw)
    if num_samples:
        dataset = dataset.select(range(num_samples))
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples")

    # 4) evaluation loop
    correct, total = 0, 0
    results: List[Dict] = []
    progress = tqdm(total=total_samples, desc="Processing", unit="ex", dynamic_ncols=True)

    for start in range(0, total_samples, batch_size):
        batch = dataset[start : start + batch_size]
        bs = len(batch["problem"])

        # build prompts
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
        formatted = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]

        # generate outputs
        outputs = llm.generate(formatted, sampling_params)

        # analyse outputs
        for j, out in enumerate(outputs):
            response = out.outputs[0].text
            pred = extract_xml_answer(response)
            truth = batch["answer"][j]

            is_ok = safe_float_equal(pred, truth)
            correct += int(is_ok)
            total += 1

            results.append(
                {
                    "question": batch["problem"][j],
                    "true_answer": truth,
                    "generated_answer": pred,
                    "full_response": response,
                    "correct": is_ok,
                }
            )

        progress.update(bs)
        progress.set_postfix(acc=f"{(correct/total)*100:.2f}%", corr=f"{correct}/{total}")

    progress.close()

    # 5) compute metrics
    metrics = {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
    }

    # 6) save results
    if save_results:
        fname = f"amc_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(fname, "w") as fp:
            json.dump({"metrics": metrics, "results": results}, fp, indent=2)
        print(f"Results saved to {fname}")

    return metrics

# ──────────────────── CLI ────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate model on AMC benchmark (VLLM).")
    p.add_argument("--model_path", type=str, required=True, help="HF model or checkpoint dir")
    p.add_argument("--dataset_path", type=str, required=True, help="AMC eval JSON file")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_samples", type=int, default=None, help="Evaluate first N examples")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.3)
    p.add_argument("--save_results", action="store_true", help="Save per‑example outputs")
    return p

# ──────────────────── main ────────────────────
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
