"""Repeatable, single-diff generation; keep raw output and reject format violations."""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

MAX_DIFF_TOKENS = 4096
MAX_NEW_TOKENS = 64
MAX_SUBJECT_CHARACTERS = 72
GENERATION_SETTINGS = {
    "do_sample": False,
    "num_beams": 1,
    "num_return_sequences": 1,
    "max_new_tokens": MAX_NEW_TOKENS,
    "min_new_tokens": 0,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "repetition_penalty": 1.0,
    "use_cache": True,
}
SYSTEM_PROMPT = (
    Path(__file__).with_name("commit_subject_prompt.md").read_text(encoding="utf-8").rstrip("\n")
)


def prepare_messages(diff: str, tokenizer) -> tuple[list[dict[str, str]], dict]:
    """Truncate only the diff, leaving instructions and chat framing intact."""
    if not isinstance(diff, str) or not diff.strip():
        raise ValueError("Provide a nonempty diff string.")
    tokens = tokenizer.encode(diff, add_special_tokens=False)
    tokenizer.truncation_side = "right"
    kept = tokenizer.encode(
        diff, add_special_tokens=False, truncation=True, max_length=MAX_DIFF_TOKENS
    )
    content = tokenizer.decode(kept, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    return messages, {
        "diff_tokens": len(tokens),
        "diff_tokens_used": len(kept),
        "diff_truncated": len(tokens) > MAX_DIFF_TOKENS,
    }


def output_errors(raw: str, *, hit_token_limit: bool = False) -> list[str]:
    """Check mechanical formatting and common preambles, without rewriting text."""
    subject = raw.strip(" \t")
    errors = []
    if not subject.strip():
        errors.append("empty")
    if any(character in subject for character in "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"):
        errors.append("multiple_lines")
    if len(subject) > MAX_SUBJECT_CHARACTERS:
        errors.append("too_long")
    if any(
        subject.startswith(left) and subject.endswith(right)
        for left, right in (('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u2018", "\u2019"))
    ):
        errors.append("surrounding_quotes")
    if (
        "`" in subject
        or re.search(r"^(?:#{1,6}\s|>\s|[-+*]\s|\d+[.)]\s)", subject)
        or re.search(r"\*[^*]+\*|__.+__|\[[^]]+\]\([^)]+\)", subject)
    ):
        errors.append("markdown")
    if re.match(
        r"(?i)^(?:(?:commit(?: message| subject)?|subject)\s*:|here(?:'s| is)\b|sure[,!:])", subject
    ):
        errors.append("preamble")
    if hit_token_limit:
        errors.append("token_limit")
    return errors


def generation_config(model, tokenizer):
    from transformers import GenerationConfig

    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("The model or tokenizer must define an end-of-sequence token.")
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos[0] if isinstance(eos, list) else eos
    # Construct a fresh config so Qwen's sampling defaults cannot leak into the baseline.
    return GenerationConfig(
        **GENERATION_SETTINGS,
        eos_token_id=eos,
        pad_token_id=pad,
    )


def generate_subject(diff: str, *, model, tokenizer) -> dict:
    """Generate once from a diff; reference subjects are never accepted as inputs."""
    import torch

    messages, result = prepare_messages(diff, tokenizer)
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    prompt_tokens = inputs["input_ids"].shape[-1]
    if prompt_tokens + MAX_NEW_TOKENS > model.config.max_position_embeddings:
        raise ValueError("The formatted prompt and output allowance exceed the model context.")
    config = generation_config(model, tokenizer)
    model.eval()
    on_cuda = model.device.type == "cuda"
    if on_cuda:
        torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        sequence = model.generate(**inputs, generation_config=config)
    if on_cuda:
        torch.cuda.synchronize(model.device)
    seconds = time.perf_counter() - started
    token_ids = sequence[0, prompt_tokens:].tolist()
    eos_ids = (
        config.eos_token_id if isinstance(config.eos_token_id, list) else [config.eos_token_id]
    )
    finished = bool(token_ids) and token_ids[-1] in eos_ids
    raw = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    errors = output_errors(raw, hit_token_limit=not finished)
    result.update(
        raw_output=raw,
        subject=raw.strip(" \t") if not errors else None,
        valid=not errors,
        errors=errors,
        finish_reason="eos" if finished else "token_limit",
        generated_token_ids=token_ids,
        generated_tokens=len(token_ids),
        prompt_tokens=prompt_tokens,
        generation_seconds=seconds,
        system_prompt=SYSTEM_PROMPT,
        system_prompt_sha256=hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        max_diff_tokens=MAX_DIFF_TOKENS,
        truncation_side="right",
        max_subject_characters=MAX_SUBJECT_CHARACTERS,
        generation_config=config.to_dict(),
    )
    return result
