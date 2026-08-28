"""Static checks for the isolated CUDA experiment environment."""

from __future__ import annotations

from pathlib import Path

CUDA_DIRECTORY = Path(__file__).parents[1] / "experiments" / "cuda"


def test_compose_keeps_data_read_only_artifacts_writable_and_logging_local() -> None:
    dockerfile = (CUDA_DIRECTORY / "Dockerfile").read_text(encoding="utf-8")
    compose = (CUDA_DIRECTORY / "compose.yaml").read_text(encoding="utf-8")

    assert "nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04" in dockerfile
    assert "https://download.pytorch.org/whl/cu132 torch==2.12.1" in dockerfile
    assert "VIRTUAL_ENV=/opt/venv" in dockerfile
    assert 'ENV PATH="$VIRTUAL_ENV/bin:$PATH"' in dockerfile
    assert 'python3.12 -m venv "$VIRTUAL_ENV"' in dockerfile
    assert "cuda-experiment:" in compose
    assert "gpus: all" in compose
    assert "network_mode: none" in compose
    assert "target: /datasets" in compose
    assert "read_only: true" in compose
    assert "target: /artifacts" in compose
    assert "HF_HUB_DISABLE_TELEMETRY" in compose
    assert "WANDB_DISABLED" in compose


def test_smoke_test_reads_one_batch_checks_cuda_and_writes_a_private_report() -> None:
    smoke_test = (CUDA_DIRECTORY / "smoke_test.py").read_text(encoding="utf-8")

    assert 'Path("/datasets/train.parquet")' in smoke_test
    assert "iter_batches(batch_size=1)" in smoke_test
    assert "torch.cuda.is_available()" in smoke_test
    assert 'Path("/artifacts/cuda-smoke.json")' in smoke_test
