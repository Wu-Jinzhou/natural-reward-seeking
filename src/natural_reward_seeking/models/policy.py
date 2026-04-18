from __future__ import annotations

import gc
import inspect
from pathlib import Path
import re
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from natural_reward_seeking.prompting.conditions import build_messages


TERMINAL_MARKER_SUFFIX_RE = re.compile(r"(?:\s*(?:<\|im_end\|>|<\|im_start\|>|<\|endoftext\|>))+\s*$")
EMPTY_THINK_RE = re.compile(r"<think>\s*</think>\s*", flags=re.DOTALL)


def strip_empty_think_blocks(text: str) -> str:
    return EMPTY_THINK_RE.sub("", text)


def _supports_enable_thinking(tokenizer) -> bool:
    try:
        params = inspect.signature(tokenizer.apply_chat_template).parameters
    except Exception:
        return False
    supports_extra_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
    return "enable_thinking" in params or supports_extra_kwargs


def _looks_like_olmo_chat(tokenizer) -> bool:
    special_tokens = set(getattr(tokenizer, "all_special_tokens", []) or [])
    return "<|im_start|>" in special_tokens and "<|im_end|>" in special_tokens


def render_messages_to_text(tokenizer, messages: list[dict[str, str]], *, enable_thinking: bool, add_generation_prompt: bool) -> str:
    if _looks_like_olmo_chat(tokenizer):
        parts = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "").rstrip()
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return strip_empty_think_blocks("\n".join(parts))

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            extra_kwargs = {"enable_thinking": True} if enable_thinking and _supports_enable_thinking(tokenizer) else {}
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **extra_kwargs,
            )
            return strip_empty_think_blocks(rendered)
        except Exception:
            pass

    parts = []
    for message in messages:
        role = message.get("role", "user").upper()
        parts.append(f"{role}: {message.get('content', '')}")
    if add_generation_prompt:
        parts.append("ASSISTANT:")
    return strip_empty_think_blocks("\n".join(parts))


def get_stop_token_ids(tokenizer) -> list[int]:
    ids: list[int] = []
    for attr in ("eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int) and value >= 0 and value not in ids:
            ids.append(value)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for token in ("<|im_end|>", "<|im_start|>", "<|endoftext|>"):
        token_id = None
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            token_id = None
        if not (isinstance(token_id, int) and token_id >= 0 and token_id != unk_id):
            try:
                encoded = tokenizer.encode(token, add_special_tokens=False)
            except Exception:
                encoded = None
            if encoded and len(encoded) == 1:
                token_id = int(encoded[0])
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk_id and token_id not in ids:
            ids.append(token_id)
    return ids


def trim_trailing_stop_ids(token_ids: list[int], stop_ids: Sequence[int]) -> tuple[list[int], int]:
    stop_id_set = set(stop_ids)
    trimmed = list(token_ids)
    removed = 0
    while trimmed and trimmed[-1] in stop_id_set:
        trimmed.pop()
        removed += 1
    return trimmed, removed


def strip_terminal_markers(text: str) -> str:
    return TERMINAL_MARKER_SUFFIX_RE.sub("", text or "").strip()


def split_think_tags(text: str) -> tuple[str, str]:
    open_tag = "<think>"
    close_tag = "</think>"
    open_idx = text.find(open_tag)
    close_idx = text.find(close_tag)
    if open_idx != -1 and close_idx != -1 and open_idx < close_idx:
        reasoning = text[open_idx + len(open_tag) : close_idx].strip()
        answer = (text[:open_idx] + text[close_idx + len(close_tag) :]).strip()
        return reasoning, answer
    return "", text.strip()


def parse_generated_output(clean_output: str) -> dict[str, Any]:
    text = strip_terminal_markers(clean_output)
    reasoning_trace, answer = split_think_tags(text)
    return {
        "raw_output": text,
        "reasoning_trace": reasoning_trace,
        "answer": answer.strip() if answer else text,
        "has_reasoning_trace": bool(reasoning_trace),
    }


class PolicyGenerator:
    def __init__(
        self,
        *,
        model_id: str,
        batch_size: int,
        max_new_tokens: int,
        enable_thinking: bool,
        attn_implementation: str,
        trust_remote_code: bool,
    ) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.attn_implementation = attn_implementation
        self.trust_remote_code = trust_remote_code
        self.tokenizer = None
        self.model = None

    def load(self):
        if self.tokenizer is not None and self.model is not None:
            return self.tokenizer, self.model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.tokenizer.padding_side = "left"

        kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self.model_id,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            if self.attn_implementation != "auto":
                kwargs["attn_implementation"] = self.attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(**kwargs)
        self.model.eval()
        return self.tokenizer, self.model

    def release(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate_case_rows(self, case_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tokenizer, model = self.load()
        stop_ids = get_stop_token_ids(tokenizer)
        model_device = next(model.parameters()).device
        results: list[dict[str, Any]] = []

        for start in range(0, len(case_rows), self.batch_size):
            batch_rows = case_rows[start : start + self.batch_size]
            prompts = []
            for row in batch_rows:
                messages = build_messages(row["prompt_text"], row["system_prompt"])
                prompts.append(
                    render_messages_to_text(
                        tokenizer,
                        messages,
                        enable_thinking=self.enable_thinking,
                        add_generation_prompt=True,
                    )
                )
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            encoded = {key: value.to(model_device) for key, value in encoded.items()}
            prompt_width = int(encoded["input_ids"].shape[1])
            outputs = model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                eos_token_id=(stop_ids or None),
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            for row, output_ids in zip(batch_rows, outputs, strict=True):
                generated_ids = output_ids[prompt_width:].tolist()
                trimmed_ids, removed_terminal_tokens = trim_trailing_stop_ids(generated_ids, stop_ids)
                decoded = tokenizer.decode(trimmed_ids, skip_special_tokens=False)
                parsed = parse_generated_output(decoded)
                enriched = dict(row)
                enriched.update(parsed)
                enriched.update(
                    {
                        "has_reasoning_trace": parsed["has_reasoning_trace"],
                        "reasoning_char_count": len(parsed["reasoning_trace"]),
                        "answer_char_count": len(parsed["answer"]),
                        "num_tokens_generated": len(trimmed_ids),
                        "removed_terminal_tokens": removed_terminal_tokens,
                    }
                )
                results.append(enriched)
        return results

