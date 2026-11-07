"""Build, rent, smoke-test, collect results, and destroy a Vast GPU VM."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from vast_runner import RunError
from vast_runner.artifacts import PROJECT_DIRECTORY
from vast_runner.terraform import recover
from vast_runner.workflow import run_smoke

DEFAULT_GPU_NAMES = ["RTX 3090", "RTX PRO 4000"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_DIRECTORY / "notebooks/prepared/train.parquet"
    )
    parser.add_argument(
        "--gpu",
        action="append",
        dest="gpu_names",
        metavar="NAME",
        help="Allowed GPU model; repeat for multiple models. Replaces the default allowlist.",
    )
    parser.add_argument("--max-hourly-price", type=float, default=0.50)
    parser.add_argument("--timeout-minutes", type=float, default=60)
    parser.add_argument(
        "--cleanup", type=Path, help="Retry teardown for an existing run directory."
    )
    arguments = parser.parse_args(argv)
    gpu_names = arguments.gpu_names if arguments.gpu_names is not None else DEFAULT_GPU_NAMES
    if any(not name.strip() for name in gpu_names):
        parser.error("Provide at least one nonblank GPU model name.")
    gpu_names = list(dict.fromkeys(name.strip() for name in gpu_names))
    if not 0 < arguments.max_hourly_price <= 10 or not 0 < arguments.timeout_minutes <= 120:
        parser.error("Use a positive hourly limit <= $10 and a time limit <= 120 minutes.")
    api_key = os.environ.get("VAST_API_KEY") or os.environ.get("VASTAI_API_KEY")
    if not api_key:
        parser.error("Set VAST_API_KEY before running.")
    try:
        if arguments.cleanup:
            recover(arguments.cleanup)
        else:
            report = run_smoke(
                arguments.dataset.resolve(),
                gpu_names=gpu_names,
                max_hourly_price=arguments.max_hourly_price,
                timeout_minutes=arguments.timeout_minutes,
            )
            print(f"CUDA smoke test passed. Report: {report}")
    except KeyboardInterrupt:
        print("Interrupted. Check run.json for cleanup status.", file=sys.stderr)
        return 130
    except (RunError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"CUDA run failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
