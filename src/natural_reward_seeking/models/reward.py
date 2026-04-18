from __future__ import annotations

from typing import Any, Sequence

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SkyworkRewardScorer:
    def __init__(
        self,
        *,
        model_id: str,
        trust_remote_code: bool = False,
        attn_implementation: str = "auto",
        max_length: int = 4096,
    ) -> None:
        self.model_id = model_id
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation
        self.max_length = max_length
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
        self.tokenizer.padding_side = "right"
        kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self.model_id,
            "trust_remote_code": self.trust_remote_code,
            "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            if self.attn_implementation != "auto":
                kwargs["attn_implementation"] = self.attn_implementation
        self.model = AutoModelForSequenceClassification.from_pretrained(**kwargs)
        self.model.eval()
        return self.tokenizer, self.model

    def _render_text(self, messages: list[dict[str, str]]) -> str:
        tokenizer, _ = self.load()
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                pass
        parts = []
        for message in messages:
            parts.append(f"{message.get('role', 'user').upper()}: {message.get('content', '')}")
        return "\n".join(parts)

    @torch.inference_mode()
    def score_messages(
        self,
        message_batches: Sequence[list[dict[str, str]]],
        batch_size: int = 8,
        *,
        progress_desc: str | None = None,
        show_progress: bool = True,
    ) -> list[float]:
        tokenizer, model = self.load()
        texts = [self._render_text(messages) for messages in message_batches]
        scores: list[float] = []
        model_device = next(model.parameters()).device
        batch_starts = range(0, len(texts), batch_size)
        if show_progress:
            batch_starts = tqdm(
                batch_starts,
                total=(len(texts) + batch_size - 1) // batch_size,
                desc=progress_desc or "Scoring rewards",
                unit="batch",
            )
        for start in batch_starts:
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(model_device) for key, value in encoded.items()}
            outputs = model(**encoded)
            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise RuntimeError("Reward model did not return logits.")
            scores.extend(float(score) for score in logits[:, 0].detach().float().cpu().tolist())
        return scores
