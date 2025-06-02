import argparse, json, logging, re
from datetime import datetime
from typing import Dict, List

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logging.getLogger("vllm").setLevel(logging.WARNING)

# ───────── prompt helpers ─────────
R1_STYLE_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it.\n"
    "The assistant first thinks about the reasoning process in the mind and then "
    "provides the user with the answer. Let's think step by step and output the "
    "final answer within \\boxed{}."
)

def last_boxed_only_string(txt: str):
    idx = txt.rfind("\\boxed")
    if "\\boxed " in txt:
        return "\\boxed " + txt.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = txt.rfind("\\fbox")
        if idx < 0:
            return None
    i, rb, depth = idx, None, 0
    while i < len(txt):
        if txt[i] == "{": depth += 1
        if txt[i] == "}":
            depth -= 1
            if depth == 0: rb = i; break
        i += 1
    return None if rb is None else txt[idx:rb + 1]

def remove_boxed(s: str | None) -> str:
    if not s: return ""
    if s.startswith("\\boxed "): return s[len("\\boxed "):]
    left = "\\boxed{"
    return s[len(left):-1] if s.startswith(left) and s.endswith("}") else s

def extract_xml_answer(text: str) -> str:
    return remove_boxed(last_boxed_only_string(text))

# loose equality: numeric fallback else string
def is_equiv(a, b, tol=1e-5):
    try:
        return abs(float(a) - float(b)) <= tol
    except (ValueError, TypeError):
        return str(a).strip() == str(b).strip()

# ───────── evaluation ─────────
def evaluate_model(
    model_path: str,
    dataset_path: str,
    batch_size: int,
    num_samples: int | None,
    gpu_memory_utilization: float,
    save_results: bool,
) -> Dict:
    print("Initializing evaluation...")

    # 1) model + tokenizer
    with tqdm(total=2, desc="Loading model components") as pbar:
        llm = LLM(
            model=model_path,
            dtype="half",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=1024,
            device="cuda:0",
            enable_chunked_prefill=False,
        ); pbar.update(1)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=2048,
            padding_side="right",
            truncation_side="right",
        ); pbar.update(1)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=3000,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    # 2) dataset
    print(f"Loading dataset from {dataset_path} ...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    dataset = Dataset.from_list(raw)
    if num_samples:
        dataset = dataset.select(range(num_samples))
    total_samples = len(dataset)
    print(f"Loaded {total_samples} samples")

    correct, total = 0, 0
    results: List[Dict] = []
    pbar = tqdm(total=total_samples, desc="Processing", unit="ex", dynamic_ncols=True)

    # 3) batch inference
    for start in range(0, total_samples, batch_size):
        batch = dataset[start : start + batch_size]
        bs = len(batch["problem"])

        prompts = [
            [
                {"role": "system", "content": R1_STYLE_SYSTEM_PROMPT},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant",
                 "content": "To calculate 2+2, we simply add them: 4. The answer is \\boxed{4}"},
                {"role": "user", "content": q.strip()},
            ]
            for q in batch["problem"]
        ]
        formatted = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]

        outputs = llm.generate(formatted, sampling_params)

        for j, out in enumerate(outputs):
            response = out.outputs[0].text
            pred = extract_xml_answer(response).replace(" ", "")
            truth = re.sub(r'\$(.*?)\$', r'\1', batch["answer"][j][0]).replace(" ", "")

            ok = is_equiv(pred, truth)
            correct += int(ok); total += 1

            results.append({
                "question": batch["problem"][j],
                "true_answer": truth,
                "generated_answer": pred,
                "full_response": response,
                "correct": ok,
            })

        pbar.update(bs)
        pbar.set_postfix(acc=f"{(correct/total)*100:.2f}%", corr=f"{correct}/{total}")
    pbar.close()

    metrics = {
        "accuracy": correct / total if total else 0.0,
        "correct": correct, "total": total,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
    }

    if save_results:
        fname = f"olympiad_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(fname, "w") as fp:
            json.dump({"metrics": metrics, "results": results}, fp, indent=2)
        print(f"Results saved to {fname}")

    return metrics

# ───────── CLI ─────────
def build_parser():
    p = argparse.ArgumentParser(description="Evaluate model on Olympiad‑Bench (VLLM).")
    p.add_argument("--model_path", required=True, help="HF model or local checkpoint")
    p.add_argument("--dataset_path", required=True, help="olympiad_bench.json")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.3)
    p.add_argument("--save_results", action="store_true")
    return p

if __name__ == "__main__":
    args = build_parser().parse_args()
    stats = evaluate_model(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        gpu_memory_utilization=args.gpu_memory_utilization,
        save_results=args.save_results,
    )

    print("\nFinal Evaluation Results")
    print(f"Accuracy: {stats['accuracy']:.2%}")
    print(f"Correct:  {stats['correct']}/{stats['total']}")
