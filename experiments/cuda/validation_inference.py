"""Generate once for every validation diff, without receiving reference subjects."""

from __future__ import annotations

import json
from pathlib import Path

from generation import generate_subject
from model_load import load_model
from startup import CHECKS
from validation_data import file_sha256, read_rows


def main(
    dataset: Path = Path("/datasets") / CHECKS["validation"]["dataset"],
    report: Path = Path("/artifacts") / CHECKS["validation"]["report"],
) -> None:
    report.unlink(missing_ok=True)
    rows = read_rows(dataset, ["id", "diff"], exact=True)
    model, tokenizer, diagnostic = load_model()
    predictions = []
    for row in rows:
        result = generate_subject(row["diff"], model=model, tokenizer=tokenizer)
        predictions.append({"id": row["id"], **result})
        if len(predictions) % 10 == 0 or len(predictions) == len(rows):
            print(f"Generated {len(predictions)}/{len(rows)} validation predictions", flush=True)
    diagnostic.update(
        input_sha256=file_sha256(dataset), row_count=len(rows), predictions=predictions
    )
    report.write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
    print(f"Saved validation results to {report}", flush=True)


if __name__ == "__main__":
    main()
