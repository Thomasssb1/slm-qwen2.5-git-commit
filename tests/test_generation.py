"""Check prompt boundaries, raw decoding, and acceptance without GPU dependencies."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

with pytest.MonkeyPatch.context() as imports:
    imports.syspath_prepend(str(Path(__file__).parents[1] / "experiments/cuda"))
    import generation
    import inference_smoke


class TokenIds:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def __getitem__(self, index):
        _batch, section = index
        return TokenIds(self.values[section])

    def tolist(self):
        return self.values


class Inputs(dict):
    def to(self, device):
        return self


class Tokenizer:
    eos_token_id = 0
    pad_token_id = None

    def encode(self, text, **options):
        tokens = [ord(character) for character in text]
        return tokens[: options["max_length"]] if options.get("truncation") else tokens

    def decode(self, tokens, **options):
        return "".join(chr(token) for token in tokens if token != 0)

    def apply_chat_template(self, messages, **options):
        assert options == {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        self.messages = messages
        text = (
            "<system>" + messages[0]["content"] + "<user>" + messages[1]["content"] + "<assistant>"
        )
        return Inputs(input_ids=TokenIds(self.encode(text)))


@pytest.mark.parametrize("length, truncated", [(4095, False), (4096, False), (4097, True)])
def test_only_diff_is_truncated_and_exact_limit_is_not_marked_truncated(length, truncated):
    diff = "+" * length
    messages, stats = generation.prepare_messages(diff, Tokenizer())
    assert messages == [
        {
            "role": "system",
            "content": "Write a concise Git commit subject describing the main change in the supplied diff.\nReturn exactly one line, at most 72 characters.\nReturn only the subject, without explanations, Markdown, or surrounding quotes.",
        },
        {"role": "user", "content": diff[:4096]},
    ]
    assert stats == {
        "diff_tokens": length,
        "diff_tokens_used": min(length, 4096),
        "diff_truncated": truncated,
    }


@pytest.mark.parametrize("diff", ["", " \n", None, {"diff": "patch", "subject": "reference"}])
def test_prompt_rejects_empty_input_and_dataset_records(diff):
    with pytest.raises(ValueError, match="nonempty diff string"):
        generation.prepare_messages(diff, Tokenizer())


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Add input validation", []),
        ("é" * 72, []),
        ("x" * 73, ["too_long"]),
        ("", ["empty"]),
        ("Add validation\nExplain the change", ["multiple_lines"]),
        ("Add validation\u2028Explain", ["multiple_lines"]),
        ('"Add validation"', ["surrounding_quotes"]),
        ("\u2018Add validation\u2019", ["surrounding_quotes"]),
        ("- Add validation", ["markdown"]),
        ("**Add validation**", ["markdown"]),
        ("Update `validate`", ["markdown"]),
        ("[Add validation](example.com)", ["markdown"]),
        ("Commit subject: Add validation", ["preamble"]),
        ("Here is the subject: Add validation", ["preamble"]),
        ("fix: handle empty input", []),
    ],
)
def test_format_checks_preserve_valid_subjects_and_identify_violations(raw, expected):
    assert generation.output_errors(raw) == expected


@pytest.fixture
def inference(monkeypatch):
    class Config:
        def __init__(self, **values):
            self.__dict__.update(values)

        def to_dict(self):
            return vars(self).copy()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(GenerationConfig=Config))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    tokenizer = Tokenizer()

    class Model:
        device = SimpleNamespace(type="cpu")
        config = SimpleNamespace(max_position_embeddings=8192)
        generation_config = SimpleNamespace(eos_token_id=[0, 1], do_sample=True, top_p=0.8)
        response = "Add input validation"
        finished = True

        def eval(self):
            self.evaluating = True

        def generate(self, input_ids, generation_config):
            assert self.evaluating
            self.used_config = generation_config
            suffix = tokenizer.encode(self.response) + ([1] if self.finished else [])
            return TokenIds(input_ids.tolist() + suffix)

    # Both configured EOS IDs must be removed on decoding.
    tokenizer.decode = lambda tokens, **options: "".join(
        chr(token) for token in tokens if token not in {0, 1}
    )
    return Model(), tokenizer


@pytest.mark.parametrize(
    "raw, finished, errors",
    [
        ("Add input validation", True, []),
        ("  Add input validation  ", True, []),
        ("x" * 73, True, ["too_long"]),
        ("Add validation\nExtra explanation", True, ["multiple_lines"]),
        ("Add", False, ["token_limit"]),
    ],
)
def test_generation_decodes_only_completion_retains_raw_and_rejects_invalid_subjects(
    inference, raw, finished, errors
):
    model, tokenizer = inference
    model.response, model.finished = raw, finished
    result = generation.generate_subject("+validate(input)", model=model, tokenizer=tokenizer)
    assert result["raw_output"] == raw
    assert result["errors"] == errors
    assert result["subject"] == (None if errors else raw.strip())
    assert result["valid"] == (not errors)
    assert result["finish_reason"] == ("eos" if finished else "token_limit")
    assert result["generated_tokens"] == len(raw) + int(finished)
    assert result["generation_config"]["max_new_tokens"] == 64
    assert result["generation_config"]["do_sample"] is False
    assert result["generation_config"]["num_beams"] == 1
    assert result["generation_config"]["top_p"] == 1.0
    assert result["generation_config"]["eos_token_id"] == [0, 1]
    assert result["generation_config"]["pad_token_id"] == 0


def test_context_check_includes_chat_overhead_and_output_allowance(inference):
    model, tokenizer = inference
    model.config.max_position_embeddings = 300
    with pytest.raises(ValueError, match="exceed the model context"):
        generation.generate_subject("+x" * 20, model=model, tokenizer=tokenizer)
    assert not hasattr(model, "used_config")


def test_generation_uses_tokenizer_eos_and_existing_pad_when_model_has_no_eos(inference):
    model, tokenizer = inference
    model.generation_config.eos_token_id = None
    tokenizer.pad_token_id = 10
    config = generation.generation_config(model, tokenizer)
    assert config.eos_token_id == 0
    assert config.pad_token_id == 10
    tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="end-of-sequence"):
        generation.generation_config(model, tokenizer)


def test_cuda_generation_synchronizes_and_keeps_the_same_decoding_contract(inference, monkeypatch):
    model, tokenizer = inference
    model.device = SimpleNamespace(type="cuda")
    synchronizations = []
    monkeypatch.setattr(
        sys.modules["torch"],
        "cuda",
        SimpleNamespace(synchronize=synchronizations.append),
        raising=False,
    )
    result = generation.generate_subject("+validate(input)", model=model, tokenizer=tokenizer)
    assert result["subject"] == "Add input validation"
    assert result["errors"] == []
    assert result["generation_seconds"] >= 0
    assert synchronizations == [model.device, model.device]


def test_inference_smoke_writes_all_cases_truncation_and_repeat_evidence(
    inference, tmp_path, monkeypatch
):
    import json

    model, tokenizer = inference
    monkeypatch.setattr(
        inference_smoke, "load_model", lambda: (model, tokenizer, {"model_id": "fixture-model"})
    )
    path = tmp_path / "report.json"
    path.write_text('{"stale": true}')
    inference_smoke.main(path)
    report = json.loads(path.read_text())
    assert report["model_id"] == "fixture-model"
    assert set(report["examples"]) == {"single_file", "multiple_files", "truncated"}
    assert report["examples"]["truncated"]["diff_truncated"] is True
    assert report["examples"]["truncated"]["diff_tokens_used"] == 4096
    assert report["examples"]["single_file"]["subject"] == "Add input validation"
    assert report["repeat_matches"] is True
    assert report["repeat_token_ids"] == report["examples"]["single_file"]["generated_token_ids"]
    assert "stale" not in report
