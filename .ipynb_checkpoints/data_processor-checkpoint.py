import argparse
import json
import os
import re
from typing import Dict, List

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- CLI ---------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect activation metric for the final MLP.up_proj layer"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="🤖 HuggingFace model identifier or local path",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="🗃  JSON file containing the dataset (list of dicts with keys `problem` & `answer`)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="💾 Where to save the resulting .pt file with sorted indices",
    )
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="Visible CUDA devices (e.g. `0,1`). Use `-1` for CPU‑only.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=500,
        help="Batch size for dataset preprocessing",
    )
    return parser


# ---------- Pre‑/Post‑processing helpers ---------- #
R1_STYLE_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the "
    "Assistant solves it.\n"
    "The assistant first thinks about the reasoning process in the mind and then "
    "provides the user with the answer. Let's think step by step and output the "
    "final answer within \\boxed{}."
)


def preprocess_dataset(data: Dataset, chunk_size: int = 1000) -> Dataset:
    """Wrap each math problem into a chat prompt expected by Qwen models."""
    def process_batch(batch):
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
        return {"prompt": prompts, "answer": batch["answer"]}

    return data.map(process_batch, batched=True, batch_size=chunk_size)


def register_act_hooks(model, target_layer_name: str, store: Dict[str, torch.Tensor]):
    """Attach a forward hook that grabs the *input* to `target_layer_name`."""
    hooks = []

    def _get_hook(name):
        def hook(_, inputs, __):
            # inputs is a tuple; we want the Tensor containing [batch, hidden]
            store[name] = inputs[0].detach()
        return hook

    for name, module in model.named_modules():
        if name == target_layer_name:
            hooks.append(module.register_forward_hook(_get_hook(name)))
            break  # only one layer should match
    return hooks


def remove_hooks(hooks: List[torch.utils.hooks.RemovableHandle]):
    for h in hooks:
        h.remove()


# ---------- Main ---------- #
def main():
    args = build_parser().parse_args()

    # device setup
    if args.gpu_id != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        device = "cuda"
    else:
        device = "cpu"

    # 1) Load model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Identify the last layer’s up_proj (e.g. `model.layers.<N-1>.mlp.up_proj`)
    num_layers = len(model.model.layers)
    target_layer = f"model.layers.{num_layers-1}.mlp.up_proj"
    model.eval()

    # 2) Load dataset
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    hf_dataset = Dataset.from_list(raw_data)
    dataset = preprocess_dataset(hf_dataset, chunk_size=args.chunk_size)

    # Only keep prompts (list of messages)
    prompts_list = [rec["prompt"] for rec in dataset]

    # 3) Collect activations
    act_inputs: List[Dict[str, torch.Tensor]] = []
    for prompt in tqdm(prompts_list, desc="Collecting angles"):
        text = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)

        store: Dict[str, torch.Tensor] = {}
        hooks = register_act_hooks(model, target_layer, store)

        with torch.no_grad():
            _ = model(**model_inputs)

        act_inputs.append(store)
        remove_hooks(hooks)

    # 4) Compute cosine‑similarity metric
    metrics = []
    for sample_store in act_inputs:
        inp = sample_store[target_layer].squeeze(0).float()  # [seq_len, hidden]
        normalized = F.normalize(inp, p=2, dim=1)
        cos = normalized @ normalized.T
        cos = cos * torch.tril(torch.ones_like(cos), diagonal=1)
        val = cos[110:-6][110:-6].mean() + 8 * cos[110:-6][:110].mean()
        metrics.append(val)

    indices = torch.argsort(torch.tensor(metrics), descending=True)

    # 5) Save result
    torch.save(indices, args.save_path)
    print(f"✅ Saved sorted indices to: {args.save_path}")


if __name__ == "__main__":
    main()
