"""Load the pinned Qwen tokenizer and full BF16 model on CUDA, without inference."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from model_spec import MODEL_ID, MODEL_REVISION


def load_report() -> dict:
    import torch
    import transformers
    from smoke_test import _cuda_diagnostic
    from transformers import AutoModelForCausalLM, AutoTokenizer

    diagnostic = _cuda_diagnostic()
    device = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    print(f"Loading {MODEL_ID} at revision {MODEL_REVISION}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False
    )
    tokens = tokenizer.encode("Load the model.", add_special_tokens=False)
    if not tokens:
        raise RuntimeError("The tokenizer did not produce any tokens.")
    model, loading = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        use_safetensors=True,
        trust_remote_code=False,
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise RuntimeError(
            "The checkpoint did not load cleanly; some model weights differ or are missing."
        )
    model.to(f"cuda:{device}")
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    cuda_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.device.type == "cuda"
    )
    if parameter_count == 0 or cuda_parameter_count != parameter_count:
        raise RuntimeError("The full model must be resident on CUDA.")
    if any(parameter.dtype != torch.bfloat16 for parameter in model.parameters()):
        raise RuntimeError("All model parameters must use BF16.")
    torch.cuda.synchronize(device)
    diagnostic.update(
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        model_class=type(model).__name__,
        parameter_count=parameter_count,
        cuda_parameter_count=cuda_parameter_count,
        parameter_dtype="bfloat16",
        tokenizer_vocab_size=len(tokenizer),
        tokenizer_probe_tokens=len(tokens),
        transformers_version=transformers.__version__,
        load_seconds=time.monotonic() - started,
        cuda_memory_allocated_bytes=torch.cuda.memory_allocated(device),
        cuda_peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
        ran_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    return diagnostic


def main(report: Path = Path("/artifacts/model-load.json")) -> None:
    report.unlink(missing_ok=True)
    result = load_report()
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
