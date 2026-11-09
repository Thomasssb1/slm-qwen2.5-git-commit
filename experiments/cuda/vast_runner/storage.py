"""Private sample and result transfer through the run's temporary S3 bucket."""

import hashlib
import json
import time
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from . import RunError
from .artifacts import validate_report, write_json


class Transfer:
    def __init__(self, region: str | None):
        session = boto3.Session(region_name=region)
        if session.get_credentials() is None:
            raise RunError("Configure AWS credentials before running.")
        self.region = session.region_name or "eu-west-2"
        self.client = session.client(
            "s3",
            region_name=self.region,
            config=Config(
                signature_version="s3v4",
                connect_timeout=10,
                read_timeout=10,
                retries={"mode": "standard", "total_max_attempts": 3},
                s3={"addressing_style": "virtual", "us_east_1_regional_endpoint": "regional"},
            ),
        )

    def prepare(self, bucket: str, sample: Path | None, run_id: str, seconds: int) -> dict:
        job = {
            "run_id": run_id,
            "result_url": self.client.generate_presigned_url(
                "put_object",
                Params={"Bucket": bucket, "Key": "result.json", "ContentType": "application/json"},
                ExpiresIn=seconds,
            ),
            "hold_seconds": seconds,
        }
        if sample is not None:
            data = sample.read_bytes()
            self.client.put_object(Bucket=bucket, Key="train.parquet", Body=data)
            job["sample_sha256"] = hashlib.sha256(data).hexdigest()
            job["sample_url"] = self.client.generate_presigned_url(
                "get_object", Params={"Bucket": bucket, "Key": "train.parquet"}, ExpiresIn=seconds
            )
        return job

    def collect(
        self, bucket: str, directory: Path, run_id: str, deadline: float, *, check: str = "cuda"
    ) -> Path:
        while time.monotonic() < deadline:
            try:
                response = self.client.get_object(Bucket=bucket, Key="result.json")
            except ClientError as error:
                if error.response["Error"]["Code"] not in {"NoSuchKey", "404"}:
                    raise
                time.sleep(min(5, max(0, deadline - time.monotonic())))
                continue
            with response["Body"] as body:
                payload = body.read(2 * 1024 * 1024 + 1)
            if len(payload) > 2 * 1024 * 1024:
                raise RunError("The container returned an oversized result.")
            result = json.loads(payload)
            if not isinstance(result, dict) or result.get("run_id") != run_id:
                raise RunError("The container result does not belong to this run.")
            write_json(directory / "result.json", result)
            (directory / "smoke.log").write_text(str(result.get("log", "")), encoding="utf-8")
            if result.get("outcome") != "passed":
                raise RunError(f"The container smoke test failed; see {directory / 'smoke.log'}")
            report = directory / ("model-load.json" if check == "model" else "cuda-smoke.json")
            write_json(report, result.get("report"))
            validate_report(report, check=check)
            return report
        raise RunError("The container did not return a result within the rental time limit.")


STORAGE_ERRORS = (BotoCoreError, ClientError)
