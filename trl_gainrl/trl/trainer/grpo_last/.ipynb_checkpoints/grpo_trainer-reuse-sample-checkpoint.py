# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from unittest.mock import patch
import torch.nn.functional as F
import time
import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather_object
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..import_utils import is_vllm_available
from ..models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from .grpo_config import GRPOConfig
from .utils import compute_logps_with_prompt_cache, generate_model_card, get_comet_experiment_url, pad


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class GRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.use_vllm = args.use_vllm
        self.gradient_checkpointing = args.gradient_checkpointing

        self.beta = args.beta
        self.bad_logit_cache =None
        self.good_logit_cache =None
        self.all_logit_cache =None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_init_kwargs.get("device")
                if vllm_device == "auto":
                    vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also used for training. This may lead to unexpected "
                        "behavior. It is recommended to use a dedicated device for vLLM."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )
                with world_size_patch, profiling_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        **self.args.vllm_init_kwargs,
                    )
                self.sampling_params = SamplingParams(
                    n=self.num_generations,
                    temperature=args.temperature,
                    max_tokens=self.max_completion_length,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad checkpointing

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                num_return_sequences=self.num_generations,
                pad_token_id=processing_class.pad_token_id,
            )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs




    
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        start_time = time.perf_counter()
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    state_dict = unwrapped_model.state_dict()
                if self.accelerator.is_main_process:
                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                    llm_model.load_weights(state_dict.items())
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            
            if self.accelerator.is_main_process:
                # print(len(all_prompts_text))
                # print(self.sampling_params)
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                # print(len(outputs))
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
                # print(len(completion_ids))
            else:
                completion_ids = [None] * len(all_prompts_text) * self.num_generations
            
            
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts) * self.num_generations,
                (self.accelerator.process_index + 1) * len(prompts) * self.num_generations,
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_inputs_repeated = torch.repeat_interleave(prompt_inputs["input_ids"], self.num_generations, dim=0)
            prompt_completion_ids = torch.cat([prompt_inputs_repeated, completion_ids], dim=1)
        else:
            # Regular generation path
            with torch.no_grad(), unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    **prompt_inputs, generation_config=self.generation_config
                )

        end_time = time.perf_counter()
        # if self.accelerator.is_main_process and self.args.enable_profiling:
        print(f"Generation took {end_time - start_time:0.4f} seconds")
        # print(prompt_completion_ids.size())

        # Compute prompt length and extract completion ids
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Get the logit computation mini-batch size from the config
        mini_batch_size = self.args.logit_computation_mini_batch_size
        start_time = time.perf_counter()
        if not self.gradient_checkpointing:
            # Current policy logprobs (with grad)
            per_token_logps = compute_logps_with_prompt_cache(
                model=model,
                prompt_inputs=prompt_inputs,
                completion_ids=completion_ids,
                mini_batch_size=mini_batch_size,
                requires_grad_for_completion=True,
            )

            # Reference model logprobs (no grad)
            if self.ref_model is not None:
                ref_per_token_logps = compute_logps_with_prompt_cache(
                    model=self.ref_model,
                    prompt_inputs=prompt_inputs,
                    completion_ids=completion_ids,
                    mini_batch_size=mini_batch_size,
                    requires_grad_for_completion=False,
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = compute_logps_with_prompt_cache(
                        model=model,
                        prompt_inputs=prompt_inputs,
                        completion_ids=completion_ids,
                        mini_batch_size=mini_batch_size,
                        requires_grad_for_completion=False,
                    )
        # If gradient checkpointing is used, fall back to original implementation
        else:
            # Concatenate prompt_mask with completion_mask for logit computation
            prompt_mask_repeated = prompt_inputs["attention_mask"].repeat_interleave(self.num_generations, dim=0)
            attention_mask = torch.cat([prompt_mask_repeated, completion_mask], dim=1)  # (B*G, P+C)

            # Get the per-token log probabilities for the completions for the model and the reference model
            def get_per_token_logps(model, input_ids, attention_mask, num_logits_to_keep, mini_batch_size):
                mini_batch_size = input_ids.size(0) if mini_batch_size == 0 else mini_batch_size
                per_token_logps = []
                for i in range(0, input_ids.size(0), mini_batch_size):
                    mini_batch_input_ids = input_ids[i : i + mini_batch_size, :]  # (B_mini, P+C)
                    mini_batch_attention_mask = attention_mask[i : i + mini_batch_size, :]  # (B_mini, P+C)
                    logits = model(
                        input_ids=mini_batch_input_ids,
                        attention_mask=mini_batch_attention_mask,
                        num_logits_to_keep=num_logits_to_keep + 1,
                    ).logits[:, -num_logits_to_keep - 1 : -1]  # (B_mini, P+C, Vocab_size)

                    token_index = mini_batch_input_ids[:, -num_logits_to_keep:].unsqueeze(-1)  # (B_mini, P+C, 1)
                    token_logits = torch.gather(logits, dim=-1, index=token_index).squeeze(-1)
                    logsumexp_values = torch.stack([torch.logsumexp(l, dim=-1) for l in logits])
                    del logits
                    token_log_prob = token_logits - logsumexp_values
                    per_token_logps.append(token_log_prob)
                return torch.cat(per_token_logps, dim=0)

            num_logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
            # print(f"prompt_completion_ids{prompt_completion_ids}")
            # print(f"attention_mask{attention_mask}")
            # print(f"num_logits_to_keep{num_logits_to_keep}")
            # print(f"mini_batch_size{mini_batch_size}")
            per_token_logps = get_per_token_logps(
                model=model,
                input_ids=prompt_completion_ids,
                attention_mask=attention_mask,
                num_logits_to_keep=num_logits_to_keep,
                mini_batch_size=mini_batch_size,
            )
            # print(f"prompt_completion_ids{prompt_completion_ids.size()}")
            # print(f"per_token_logps{per_token_logps.size()}")

            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps = get_per_token_logps(
                        model=self.ref_model,
                        input_ids=prompt_completion_ids,
                        attention_mask=attention_mask,
                        num_logits_to_keep=num_logits_to_keep,
                        mini_batch_size=mini_batch_size,
                    )
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = get_per_token_logps(
                            model=model,
                            input_ids=prompt_completion_ids,
                            attention_mask=attention_mask,
                            num_logits_to_keep=num_logits_to_keep,
                            mini_batch_size=mini_batch_size,
                        )
        end_time = time.perf_counter()
        # if self.accelerator.is_main_process and self.args.enable_profiling:
        print(f"Logits computation took {end_time - start_time:0.4f} seconds")

        # Compute the KL divergence between the model and the reference model
        # print(f"ref_per_token_logps{ref_per_token_logps}")
        # print(f"per_token_logps{per_token_logps}")
        # per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        current_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
           
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        start_time = time.perf_counter()
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        
#========================================================================================================================
        rewards_current = rewards_per_func.sum(dim=1)  # 当前生成的 rewards

        # ======== 如果存在上一次保存的最佳8个回答，则与当前轮数据合并进行归一化 ========
        # if hasattr(self, "good_logit_cache") and self.good_logit_cache is not None:
        #     # 从缓存中取出上一次的 raw rewards
        #     stored_rewards = self.good_logit_cache["reward"]  # shape: (N_cache,)
        #     print(f"rewards_current{rewards_current}")
        #     merged_rewards = torch.cat([stored_rewards, rewards_current], dim=0)
        #     global_mean = merged_rewards.mean()
        #     global_std = merged_rewards.std() + 1e-4
        #     advantages_current = (rewards_current - global_mean) / global_std
        #     advantages_stored = (stored_rewards - global_mean) / global_std
            
            
        #     print(f"advantages_current{advantages_current.mean()}")
        #     print(f"advantages_stored{advantages_stored.mean()}")
        # else:
        #     global_mean = rewards_current.mean()
        #     global_std = rewards_current.std() + 1e-4
        #     advantages_current = (rewards_current - global_mean) / global_std
        
        mean_grouped_rewards = rewards_current.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards_current.view(-1, self.num_generations).std(dim=1)
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages_current = (rewards_current - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    
        # ========= 8. 计算当前响应的 loss =========
        # print(f"per_token_logps{per_token_logps.size()}")
        loss_current = torch.exp(per_token_logps - per_token_logps.detach()) * advantages_current.unsqueeze(1)
        loss_current = -(loss_current - self.beta * current_kl)
        loss_current = ((loss_current * completion_mask).sum(dim=1) / completion_mask.sum(dim=1))#.mean()
        
        # print(f"completion_mask{completion_mask}")
        # print(completion_mask.size())
        # ========= 9. 若存在 good_logit_cache，则计算缓存部分的 loss =========
        # if self.state.global_step>=20:
        if hasattr(self, "good_logit_cache") and self.good_logit_cache is not None:
            cache = self.good_logit_cache
            # 从缓存中取出必要数据
            stored_seq = cache["seq"]                  # (N_cache, seq_len)
            stored_attn = cache["attn"]                # (N_cache, seq_len)
            stored_token_logps = cache["stored_token_logps"]  # (N_cache, comp_len)  —— 缓存时存储的是上一次计算的 logits
            stored_ref_logps = cache["stored_ref_per_token_logps"]  # (N_cache, comp_len)
            stored_completion_mask = cache["completion_mask"]  # (N_cache, comp_len)
            num_logits_list = cache["num_logits"]
            advantages_stored = cache["advatages"] 

            
            # 注意：我们不存储优势，而是每次根据 raw reward 重新计算；此处已在上面得到 advantages_stored
            
            

            def get_per_token_logps_list(model, input_ids_list, attention_mask_list, num_logits_to_keep_list, mini_batch_size):
                logps_list = []  # 用于存储每个 `prompt_completion_ids` 的 log 可能性
                for input_ids, attention_mask, num_logits_to_keep in zip(input_ids_list, attention_mask_list, num_logits_to_keep_list):
                    # **1. 在前面添加 batch 维度**
                    input_ids = input_ids.unsqueeze(0)  # (1, seq_len)
                    attention_mask = attention_mask.unsqueeze(0)  # (1, seq_len)
            
                    # **2. 计算 per_token_logps**
                    per_token_logps = get_per_token_logps(
                        model=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        num_logits_to_keep=num_logits_to_keep,
                        mini_batch_size=mini_batch_size
                    )  # 输出: (num_logits_to_keep,)
            
                    # **3. 存入列表**
                    # print(per_token_logps.size())
                    logps_list.append(per_token_logps.squeeze(0))
            
                return logps_list  # list of tensor，每个 tensor 形状: (num_logits_to_keep,)

            
            stored_per_token_logps_current = get_per_token_logps_list(
                    model=model,
                    input_ids_list=stored_seq,
                    attention_mask_list=stored_attn,
                    num_logits_to_keep_list=num_logits_list,
                    mini_batch_size=self.args.logit_computation_mini_batch_size,
                )
 
            from torch.nn.utils.rnn import pad_sequence
            stored_per_token_logps_current = pad_sequence(stored_per_token_logps_current, batch_first=True, padding_value=0)
            
            
            if stored_ref_logps.size(1) > stored_per_token_logps_current.size(1):
                stored_ref_logps = stored_ref_logps[:, :stored_per_token_logps_current.size(1)]
                stored_token_logps = stored_token_logps[:, :stored_per_token_logps_current.size(1)]
                stored_completion_mask = stored_completion_mask[:, :stored_per_token_logps_current.size(1)]
                
            print(stored_per_token_logps_current.size())
            print(stored_ref_logps.size())
            
            stored_kl = torch.exp(stored_ref_logps - stored_per_token_logps_current) - (stored_ref_logps - stored_per_token_logps_current) - 1
            # print(f"stored_token_logps{stored_token_logps}") 
            # print(f"stored_per_token_logps_current{stored_per_token_logps_current}") 
            
            loss_stored = torch.exp(stored_per_token_logps_current - stored_token_logps) * advantages_stored.unsqueeze(1)
            # print(f"loss_stored{loss_stored}")
            loss_stored = -(loss_stored - self.beta * stored_kl)
            loss_stored = ((loss_stored * stored_completion_mask).sum(dim=1) / stored_completion_mask.sum(dim=1))#.mean()
            final_loss = torch.cat((loss_current, 0.5*loss_stored)).mean()
            # final_loss = loss_stored.mean()
            # print(f"stored_kl{stored_kl}")
            print(f"loss_current: {loss_current.mean()}")
            print(f"loss_stored: {loss_stored.mean()}")
            print(f"final_loss: {final_loss}")
        else:
            final_loss = loss_current.mean()
            

        
        # ========= 10. 更新 good_logit_cache：将当前响应与缓存数据合并后，重新归一化 raw rewards，并选出优势最高的 topk =========
        topk = 64 #32
        curr_seq = list(torch.unbind(prompt_completion_ids, dim=0))            # (B*G, seq_len)
        curr_attn =list(torch.unbind(attention_mask, dim=0))                   
        curr_token_logps = per_token_logps            # (B*G, comp_len)
        curr_ref_logps = ref_per_token_logps          # (B*G, comp_len)
        curr_completion_mask = completion_mask          # (B*G, comp_len)
        curr_reward = rewards_current                   # (B*G,)
        curr_num_logits_to_keep =torch.tensor([num_logits_to_keep]).repeat(rewards_current.size(0)).tolist() 
        
        # 合并当前响应与缓存（若存在）
        if hasattr(self, "all_logit_cache") and self.all_logit_cache is not None:
            cache = self.all_logit_cache
            def cat_cur_old(tensor1, tensor2):
                max_dim1 = max(tensor1.shape[1], tensor2.shape[1])
                tensor1_padded = F.pad(tensor1, (0, max_dim1 - tensor1.shape[1]))  # (left, right) padding
                tensor2_padded = F.pad(tensor2, (0, max_dim1 - tensor2.shape[1]))
                result = torch.cat([tensor1_padded, tensor2_padded], dim=0)
                return result
            
            merged_seq =cache["seq"]+[x.detach() for x in curr_seq]
            merged_attn = cache["attn"]+[x.detach() for x in curr_attn]
            merged_token_logps = cat_cur_old(cache["stored_token_logps"], curr_token_logps.detach())
            merged_ref_logps = cat_cur_old(cache["stored_ref_per_token_logps"], curr_ref_logps.detach())
            merged_completion_mask = cat_cur_old(cache["completion_mask"], curr_completion_mask.detach())
            merged_num_logits_to_keep =cache["num_logits"] + curr_num_logits_to_keep
            merged_advatages =torch.cat([cache["advatages"], advantages_current.detach()],dim=0)
            merged_std =torch.cat([cache["std"], std_grouped_rewards.detach()],dim=0)
        else:
            merged_seq = [x.detach() for x in curr_seq]
            merged_attn = [x.detach() for x in curr_attn]
            merged_token_logps = curr_token_logps.detach()
            merged_ref_logps =curr_ref_logps.detach()
            merged_completion_mask = curr_completion_mask.detach()
            merged_num_logits_to_keep = curr_num_logits_to_keep
            merged_advatages = advantages_current.detach()
            merged_std = std_grouped_rewards.detach()
    
        self.all_logit_cache = {
            "seq": merged_seq,  # (topk, seq_len)
            "attn":merged_attn,             # (topk, seq_len)
            "stored_token_logps": merged_token_logps,  # (topk, comp_len)
            "stored_ref_per_token_logps": merged_ref_logps,  # (topk, comp_len)
            "completion_mask": merged_completion_mask,  # (topk, comp_len)
            "advatages": merged_advatages,# 存储 raw reward，用于下轮归一化
            "std": merged_std,
            "num_logits": merged_num_logits_to_keep,
        }

        print(f"merged_std{merged_std.size(0)}")
        if merged_std.size(0) >= topk:
            group_std = merged_std.view(-1, self.num_generations).mean(dim=1)
            probs = group_std / (group_std.sum() + 1e-8)
            group_best_indices = torch.multinomial(probs, int(topk/self.num_generations), replacement=False)
            best_indices = [int(x) * self.num_generations + i for x in group_best_indices for i in range(self.num_generations)]
            print(8888)
            print(best_indices)
        else:
            best_indices = torch.arange(merged_std.size(0), device=device)
        
        self.good_logit_cache = {
            "seq": [merged_seq[i].detach() for i in best_indices],  # (topk, seq_len)
            "attn":[merged_attn[i].detach() for i in best_indices],             # (topk, seq_len)
            "stored_token_logps": merged_token_logps[best_indices].detach(),  # (topk, comp_len)
            "stored_ref_per_token_logps": merged_ref_logps[best_indices].detach(),  # (topk, comp_len)
            "completion_mask": merged_completion_mask[best_indices].detach(),  # (topk, comp_len)
            "advatages": merged_advatages[best_indices].detach(),# 存储 raw reward，用于下轮归一化
            "std": merged_std[best_indices].detach(),
            "num_logits": [merged_num_logits_to_keep[i] for i in best_indices],
        }
    
        # …（后续日志记录等代码保持不变）…
        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards_current).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(rewards_current.std()).mean().item())
        mean_kl = ((current_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())


        torch.cuda.empty_cache()
        del per_token_logps, ref_per_token_logps  # 清理中间变量
        return final_loss









#=========================================================================================================================
        
        
        
        
        
        # # Sum the rewards from all reward functions
        # rewards = rewards_per_func.sum(dim=1)

        # # Compute grouped-wise rewards
        # mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        # std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # # Normalize the rewards to compute the advantages
        # mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        # std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        # advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        
        # end_time = time.perf_counter()
        # # if self.accelerator.is_main_process and self.args.enable_profiling:
        # # print(f"Reward computation took {end_time - start_time:0.4f} seconds")
        
        
        # # x - x.detach() allows for preserving gradients from x
        # # print(f"per_token_logps{per_token_logps}")
        # # print(f"per_token_kl{per_token_kl}")
        # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
                             
        # # Log the metrics
        # completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        # self._metrics["completion_length"].append(completion_length)

        # reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        # for i, reward_func in enumerate(self.reward_funcs):
        #     if isinstance(reward_func, PreTrainedModel):
        #         reward_func_name = reward_func.config._name_or_path.split("/")[-1]
        #     else:
        #         reward_func_name = reward_func.__name__
        #     self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        # self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        # self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        # mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # torch.cuda.empty_cache()
        # del mean_kl, prompt_inputs, per_token_loss, per_token_logps, ref_per_token_logps
        # return loss






    

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
