"""Exercise model-load evidence and failure handling without installing GPU libraries."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

CUDA = Path(__file__).parents[1] / "experiments/cuda"
with pytest.MonkeyPatch.context() as imports:
    imports.syspath_prepend(str(CUDA))
    spec = importlib.util.spec_from_file_location("model_load", CUDA / "model_load.py")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)


@pytest.fixture
def model_dependencies(monkeypatch):
    state = {"failure": None, "downloaded": False, "evaluating": False, "inputs": []}
    parameters = []
    # Two different tensor sizes detect a report that counts tensors instead of parameters.
    for size in (100, 200):
        parameters.append(
            SimpleNamespace(
                device=SimpleNamespace(type="cpu"), dtype="bf16", numel=lambda size=size: size
            )
        )

    class Model:
        def to(self, device):
            assert device == "cuda:0"
            for parameter in parameters:
                parameter.device.type = "cuda"
                if state["failure"] == "dtype":
                    parameter.dtype = "fp32"
            if state["failure"] == "offload":
                parameters[-1].device.type = "cpu"
            return self

        def eval(self):
            state["evaluating"] = True

        def parameters(self):
            return iter([] if state["failure"] == "empty-model" else parameters)

    class Tokenizer:
        def encode(self, text, **options):
            return [] if state["failure"] == "tokenizer" else [1, 2, 3, 4]

        def __len__(self):
            return 500

    def tokenizer_from_pretrained(model_id, **options):
        state["inputs"].append((model_id, options))
        return Tokenizer()

    def model_from_pretrained(model_id, **options):
        state["inputs"].append((model_id, options))
        state["downloaded"] = True
        if state["failure"] == "download":
            raise OSError("Download failed")
        loading = {"missing_keys": ["weight"]} if state["failure"] == "checkpoint" else {}
        return Model(), loading

    def diagnostic():
        if state["failure"] == "cuda":
            raise RuntimeError("CUDA is unavailable")
        return {"cuda_device_name": "fixture GPU", "cuda_runtime_version": "13.2"}

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            bfloat16="bf16",
            cuda=SimpleNamespace(
                current_device=lambda: 0,
                reset_peak_memory_stats=lambda device: None,
                synchronize=lambda device: None,
                memory_allocated=lambda device: 600,
                max_memory_allocated=lambda device: 800,
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            __version__="fixture-version",
            AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_from_pretrained),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_from_pretrained),
        ),
    )
    monkeypatch.setitem(sys.modules, "smoke_test", SimpleNamespace(_cuda_diagnostic=diagnostic))
    return state


def test_load_report_records_full_cuda_parameter_placement_and_tokenizer(
    tmp_path, model_dependencies
):
    path = tmp_path / "model-load.json"
    loader.main(path)
    report = json.loads(path.read_text())
    assert report["parameter_count"] == report["cuda_parameter_count"] == 300
    assert report["parameter_dtype"] == "bfloat16"
    assert report["tokenizer_vocab_size"] == 500
    assert report["tokenizer_probe_tokens"] == 4
    assert report["cuda_memory_allocated_bytes"] == 600
    assert report["cuda_peak_memory_allocated_bytes"] == 800
    assert report["cuda_device_name"] == "fixture GPU"
    assert model_dependencies["evaluating"]
    assert len(model_dependencies["inputs"]) == 2
    for model_id, options in model_dependencies["inputs"]:
        assert model_id == report["model_id"] == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        assert (
            options["revision"]
            == report["model_revision"]
            == "2e1fd397ee46e1388853d2af2c993145b0f1098a"
        )
        assert options["trust_remote_code"] is False
    assert model_dependencies["inputs"][1][1]["use_safetensors"] is True


@pytest.mark.parametrize(
    "failure, error, message",
    [
        ("cuda", RuntimeError, "CUDA is unavailable"),
        ("tokenizer", RuntimeError, "tokenizer"),
        ("download", OSError, "Download failed"),
        ("checkpoint", RuntimeError, "checkpoint"),
        ("offload", RuntimeError, "resident on CUDA"),
        ("empty-model", RuntimeError, "resident on CUDA"),
        ("dtype", RuntimeError, "BF16"),
    ],
)
def test_load_failure_removes_stale_report_and_does_not_write_success(
    tmp_path, model_dependencies, failure, error, message
):
    path = tmp_path / "model-load.json"
    path.write_text('{"previous": "success"}')
    model_dependencies["failure"] = failure
    with pytest.raises(error, match=message):
        loader.main(path)
    assert not path.exists()
    if failure in {"cuda", "tokenizer"}:
        assert not model_dependencies["downloaded"]
