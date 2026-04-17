"""Hugging Face causal language model scorer.

The scorer computes a sentence score from next-token log probabilities. For a
tokenized sentence ``t_1 ... t_n``, a causal language model predicts each token
from the tokens to its left. The score is the sum or mean of:

``log P(t_i | t_1 ... t_{i-1})`` for each predictable token.

Mean log probability is the default because BLiMP sentence pairs can differ
slightly in tokenization length. Higher scores are better.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_HF_MODEL = "sshleifer/tiny-gpt2"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HuggingFaceScoringConfig:
    """Configuration for causal-LM sentence scoring."""

    model_name: str = DEFAULT_HF_MODEL
    device: str | None = None
    normalize_by_token_count: bool = True
    cache_dir: str | None = None
    torch_dtype: str | None = None
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = False
    tokenizer_use_fast: bool | None = None
    device_map: str | dict[str, Any] | None = None
    offload_folder: str | None = None
    max_memory: dict[str, str] | None = None


class HuggingFaceCausalLMScorer:
    """Score sentences with a Hugging Face causal language model."""

    def __init__(
        self,
        model_name: str = DEFAULT_HF_MODEL,
        device: str | None = None,
        normalize_by_token_count: bool = True,
        cache_dir: str | None = None,
        torch_dtype: str | None = None,
        trust_remote_code: bool = False,
        low_cpu_mem_usage: bool = False,
        tokenizer_use_fast: bool | None = None,
        device_map: str | dict[str, Any] | None = None,
        offload_folder: str | None = None,
        max_memory: dict[str, str] | None = None,
    ) -> None:
        """Load a causal LM and tokenizer from Hugging Face.

        Dependencies are imported lazily so the dry-run baseline remains usable
        without installing ``torch`` and ``transformers``.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "Hugging Face scoring requires optional dependencies. "
                "Install them with: py -m pip install -r requirements-hf.txt"
            ) from error

        self.model_name = model_name
        self.normalize_by_token_count = normalize_by_token_count
        self._torch = torch
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)

        tokenizer_kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "trust_remote_code": trust_remote_code,
        }
        if tokenizer_use_fast is not None:
            tokenizer_kwargs["use_fast"] = tokenizer_use_fast
        model_kwargs: dict[str, Any] = {
            "cache_dir": cache_dir,
            "trust_remote_code": trust_remote_code,
        }
        resolved_torch_dtype = _resolve_torch_dtype(torch, torch_dtype)
        if resolved_torch_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_torch_dtype
        if low_cpu_mem_usage:
            model_kwargs["low_cpu_mem_usage"] = True
        if _has_config_value(device_map):
            model_kwargs["device_map"] = device_map
        if offload_folder:
            offload_path = Path(offload_folder)
            offload_path.mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = str(offload_path)
        if max_memory:
            model_kwargs["max_memory"] = max_memory

        start = time.perf_counter()
        LOGGER.info("Loading Hugging Face tokenizer for %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        LOGGER.info(
            "Loaded Hugging Face tokenizer for %s in %.2fs",
            model_name,
            time.perf_counter() - start,
        )

        start = time.perf_counter()
        LOGGER.info("Loading Hugging Face causal LM for %s", model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if _has_config_value(device_map):
            self.device = _primary_device_from_device_map(self.model, torch, device_map)
            LOGGER.info(
                "Using device_map=%s for Hugging Face model %s; primary input device=%s",
                device_map,
                model_name,
                self.device,
            )
        else:
            requested_device = None if device in {None, "auto"} else device
            self.device = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
            LOGGER.info("Using device %s for Hugging Face model %s", self.device, model_name)
            self.model.to(self.device)
        self.model.eval()
        LOGGER.info(
            "Loaded Hugging Face causal LM for %s in %.2fs",
            model_name,
            time.perf_counter() - start,
        )

    @property
    def name(self) -> str:
        """Return a human-readable scorer name."""
        return f"huggingface:{self.model_name}"

    def score(self, sentence: str) -> float:
        """Return a causal-LM sequence score for one sentence."""
        if not sentence.strip():
            return float("-inf")

        encoded = self.tokenizer(sentence, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_ids = encoded["input_ids"]
        if input_ids.shape[1] < 2:
            return 0.0

        with self._torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits

        shifted_logits = logits[:, :-1, :]
        shifted_labels = input_ids[:, 1:]
        token_log_probs = self._torch.log_softmax(shifted_logits, dim=-1)
        selected_log_probs = token_log_probs.gather(
            dim=-1,
            index=shifted_labels.unsqueeze(-1),
        ).squeeze(-1)

        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            token_mask = attention_mask[:, 1:].to(selected_log_probs.dtype)
            selected_log_probs = selected_log_probs * token_mask
            token_count = token_mask.sum().item()
        else:
            token_count = selected_log_probs.numel()

        total_log_prob = selected_log_probs.sum().item()
        return self._aggregate_log_prob(
            total_log_prob=total_log_prob,
            token_count=int(token_count),
            normalize_by_token_count=self.normalize_by_token_count,
        )

    @staticmethod
    def _aggregate_log_prob(
        total_log_prob: float,
        token_count: int,
        normalize_by_token_count: bool,
    ) -> float:
        """Aggregate token log probabilities into a sentence score."""
        if token_count <= 0:
            return 0.0
        if normalize_by_token_count:
            return total_log_prob / token_count
        return total_log_prob


HuggingFaceModelScorer = HuggingFaceCausalLMScorer


def _resolve_torch_dtype(torch_module: Any, torch_dtype: str | None) -> Any:
    """Convert a YAML dtype string into a torch dtype accepted by Transformers."""
    if torch_dtype in {None, ""}:
        return None
    normalized = str(torch_dtype).lower()
    if normalized == "auto":
        return "auto"
    dtype_map = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "half": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if normalized not in dtype_map:
        raise ValueError(
            f"Unsupported torch_dtype '{torch_dtype}'. "
            "Use one of: auto, float16, bfloat16, float32."
        )
    return dtype_map[normalized]


def _primary_device_from_device_map(model: Any, torch_module: Any, device_map: str | dict[str, Any]) -> str:
    """Choose the input tensor device for Accelerate-loaded models."""
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return str(parameter.device)
    if isinstance(device_map, str) and device_map == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if isinstance(device_map, dict):
        for device in device_map.values():
            if device not in {"disk", "meta"}:
                return str(device)
    return "cpu"


def _has_config_value(value: Any) -> bool:
    """Return whether an optional config value was provided."""
    return value is not None and value != ""
