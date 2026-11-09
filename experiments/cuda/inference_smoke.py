"""Exercise generation on synthetic diffs without reading a prepared dataset."""

from __future__ import annotations

import json
from pathlib import Path

from generation import generate_subject
from model_load import load_model

SINGLE_FILE_DIFF = """diff --git a/greeting.py b/greeting.py
--- a/greeting.py
+++ b/greeting.py
@@ -1,2 +1,2 @@
 def greet(name):
-    return "Hello " + name
+    return "Hello " + name.strip()
"""
MULTI_FILE_DIFF = (
    SINGLE_FILE_DIFF
    + """diff --git a/test_greeting.py b/test_greeting.py
--- a/test_greeting.py
+++ b/test_greeting.py
@@ -1,2 +1,5 @@
 from greeting import greet
+
+def test_greet_strips_whitespace():
+    assert greet(" Ada ") == "Hello Ada"
"""
)
SMOKE_DIFFS = {
    "single_file": SINGLE_FILE_DIFF,
    "multiple_files": MULTI_FILE_DIFF,
    "truncated": SINGLE_FILE_DIFF + "+# Additional example context for a long diff\n" * 800,
}


def main(report: Path = Path("/artifacts/inference-smoke.json")) -> None:
    report.unlink(missing_ok=True)
    model, tokenizer, diagnostic = load_model()
    results = {
        name: generate_subject(diff, model=model, tokenizer=tokenizer)
        for name, diff in SMOKE_DIFFS.items()
    }
    repeat = generate_subject(SINGLE_FILE_DIFF, model=model, tokenizer=tokenizer)
    diagnostic.update(
        examples=results,
        repeat_token_ids=repeat["generated_token_ids"],
        repeat_matches=repeat["generated_token_ids"]
        == results["single_file"]["generated_token_ids"],
    )
    report.write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
    print(f"Saved inference results to {report}", flush=True)


if __name__ == "__main__":
    main()
