"""Container entrypoint for a single smoke test using scoped, expiring transfer URLs."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def execute(
    job: dict,
    *,
    sample: Path = Path("/datasets/train.parquet"),
    report: Path | None = None,
    command: list[str] | None = None,
) -> dict:
    result = {"run_id": job["run_id"], "outcome": "failed", "log": ""}
    try:
        check = job.get("check", "cuda")
        if check not in {"cuda", "model"}:
            raise ValueError("Unknown check")
        if check == "cuda":
            with urllib.request.urlopen(job["sample_url"], timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != job["sample_sha256"]:
                raise ValueError("Sample checksum mismatch")
            sample.write_bytes(data)
        if report is None:
            report = Path(
                "/artifacts/model-load.json" if check == "model" else "/artifacts/cuda-smoke.json"
            )
        report.unlink(missing_ok=True)
        script = "model_load.py" if check == "model" else "smoke_test.py"
        process = subprocess.run(
            command or [sys.executable, f"/opt/qwen-commit/{script}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800 if check == "model" else 300,
            check=False,
        )
        result["log"] = process.stdout[-65536:]
        if process.returncode == 0:
            result["report"] = json.loads(report.read_text(encoding="utf-8"))
            result["outcome"] = "passed"
    except Exception as error:
        # Exceptions from HTTP clients can include the signed URL; never print it.
        result["log"] += f"\nSmoke job failed: {type(error).__name__}"
    return result


def main(job: dict) -> int:
    deadline = time.monotonic() + job["hold_seconds"]
    result = execute(job)
    request = urllib.request.Request(
        job["result_url"],
        data=json.dumps(result).encode(),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    uploaded = False
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30):
                uploaded = True
                break
        except OSError as error:
            print(f"Result upload failed: {type(error).__name__}", flush=True)
            if attempt < 2:
                time.sleep(5)
    print(f"Smoke test {result['outcome']}; waiting for runner teardown.", flush=True)
    # Vast's provider waits for 'running'. A fast exit races that check.
    # Normally Terraform destroys us immediately after collecting the result.
    while time.monotonic() < deadline:
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    return 0 if uploaded and result["outcome"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main(json.loads(sys.argv[1])))
