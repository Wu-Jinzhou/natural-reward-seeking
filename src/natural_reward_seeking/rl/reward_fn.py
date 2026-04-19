from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import aiohttp
from transformers import PreTrainedTokenizer

from natural_reward_seeking.prompting.conditions import build_messages


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


async def _post_request(router_address: str, payload: dict[str, Any], endpoint: str, max_retries: int = 8) -> dict[str, Any]:
    url = f"http://{router_address}/{endpoint}"
    last_exception: Exception | None = None
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=None)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    response.raise_for_status()
                    return await response.json()
        except Exception as exc:  # pragma: no cover - network/retry surface
            last_exception = exc
            if attempt < max_retries - 1:
                await asyncio.sleep(min(2**attempt, 30))
            continue
    if last_exception is not None:
        raise last_exception
    raise RuntimeError(f"Request to {url} failed without raising an exception.")


def _render_reward_prompt(
    reward_model_tokenizer: PreTrainedTokenizer,
    *,
    prompt_text: str,
    system_prompt: str | None,
    response_text: str,
) -> str:
    messages = build_messages(prompt_text, system_prompt)
    messages.append({"role": "assistant", "content": response_text})
    prompt = reward_model_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    bos_token = getattr(reward_model_tokenizer, "bos_token", None)
    if bos_token and prompt.startswith(bos_token):
        prompt = prompt[len(bos_token) :]
    return prompt


async def compute_skywork_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer,
    model_name: str,
) -> dict[str, Any]:
    del data_source, ground_truth
    prompt_text = str(extra_info.get("prompt_text", "")).strip()
    system_prompt = extra_info.get("system_prompt")
    if not prompt_text:
        raise ValueError("extra_info.prompt_text is required for Skywork reward scoring.")
    reward_prompt = _render_reward_prompt(
        reward_model_tokenizer,
        prompt_text=prompt_text,
        system_prompt=system_prompt if isinstance(system_prompt, str) and system_prompt else None,
        response_text=solution_str,
    )
    payload = {
        "model": model_name,
        "input": reward_prompt,
        "use_activation": False,
    }
    output = await _post_request(reward_router_address, payload, "classify")
    score = float(output["data"][-1]["probs"][-1])
    return {"score": score}
