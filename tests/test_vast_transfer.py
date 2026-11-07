"""Exercise private transfer URLs and the container wrapper without AWS or a GPU."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, urlparse

import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber

CUDA = Path(__file__).parents[1] / "experiments/cuda"
with pytest.MonkeyPatch.context() as imports:
    imports.syspath_prepend(str(CUDA))
    from vast_runner import RunError, storage

spec = importlib.util.spec_from_file_location("cuda_remote", CUDA / "run_remote.py")
remote = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remote)


class Clock:
    def __init__(self):
        self.now = 0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def transfer(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "local-fixture")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "local-fixture-secret")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    return storage.Transfer("eu-west-2")


def test_prepare_uploads_sample_and_signs_only_its_get_and_result_put(tmp_path, transfer):
    sample = tmp_path / "train.parquet"
    sample.write_bytes(b"one private row")
    with Stubber(transfer.client) as stub:
        stub.add_response(
            "put_object",
            {},
            {
                "Bucket": "test-bucket",
                "Key": "train.parquet",
                "Body": sample.read_bytes(),
            },
        )
        job = transfer.prepare("test-bucket", sample, "run-id", 3900)
        stub.assert_no_pending_responses()
    assert job["sample_sha256"] == hashlib.sha256(sample.read_bytes()).hexdigest()
    for key, path in (("sample_url", "/train.parquet"), ("result_url", "/result.json")):
        url = urlparse(job[key])
        query = parse_qs(url.query)
        assert url.scheme == "https"
        assert url.hostname == "test-bucket.s3.eu-west-2.amazonaws.com"
        assert url.path == path
        assert query["X-Amz-Expires"] == ["3900"]
        assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
        assert len(query["X-Amz-Signature"][0]) == 64
    assert parse_qs(urlparse(job["result_url"]).query)["X-Amz-SignedHeaders"] == [
        "content-type;host"
    ]
    assert "local-fixture-secret" not in json.dumps(job)


def test_missing_aws_credentials_fail_before_provisioning(monkeypatch):
    class Session:
        def get_credentials(self):
            return None

    monkeypatch.setattr(storage.boto3, "Session", lambda **kwargs: Session())
    with pytest.raises(RunError, match="Configure AWS credentials"):
        storage.Transfer(None)


def test_missing_result_polls_until_deadline(tmp_path, transfer, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(storage, "time", clock)
    with Stubber(transfer.client) as stub:
        for _ in range(2):
            stub.add_client_error("get_object", "NoSuchKey", http_status_code=404)
        with pytest.raises(RunError, match="within the rental time limit"):
            transfer.collect("test-bucket", tmp_path, "run-id", 10)
        stub.assert_no_pending_responses()
    assert clock.now == 10
    assert not (tmp_path / "cuda-smoke.json").exists()


def test_access_denied_does_not_wait_until_timeout(tmp_path, transfer):
    with Stubber(transfer.client) as stub:
        stub.add_client_error("get_object", "AccessDenied", http_status_code=403)
        with pytest.raises(ClientError, match="AccessDenied"):
            transfer.collect("test-bucket", tmp_path, "run-id", float("inf"))
        stub.assert_no_pending_responses()


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"x" * (2 * 1024 * 1024 + 1), "oversized"),
        (b"[]", "does not belong"),
        (b'{"run_id":"different"}', "does not belong"),
        (b'{"run_id":"run-id","outcome":"passed"}', "does not prove"),
    ],
    ids=["oversized", "wrong-shape", "wrong-run", "missing-report"],
)
def test_invalid_remote_results_cannot_pass(tmp_path, transfer, payload, message):
    with Stubber(transfer.client) as stub:
        stub.add_response("get_object", {"Body": io.BytesIO(payload)})
        with pytest.raises(RunError, match=message):
            transfer.collect("test-bucket", tmp_path, "run-id", float("inf"))


@pytest.fixture
def http_transfer():
    state = {"sample": b"private sample", "result": None, "put_failures": 0, "get_status": 200}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(state["get_status"])
            self.end_headers()
            self.wfile.write(state["sample"])

        def do_PUT(self):
            data = self.rfile.read(int(self.headers["Content-Length"]))
            if state["put_failures"]:
                state["put_failures"] -= 1
                self.send_response(503)
            else:
                state["result"] = json.loads(data)
                state["content_type"] = self.headers["Content-Type"]
                self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "failure", [None, "cuda", "checksum", "download", "missing-report", "upload"]
)
def test_container_downloads_runs_and_uploads_outcome_then_exits_after_hold(
    tmp_path, http_transfer, monkeypatch, failure
):
    sample = tmp_path / "train.parquet"
    report = tmp_path / "report.json"
    marker = tmp_path / "executed"
    script = tmp_path / "smoke.py"
    script.write_text(
        "from pathlib import Path\n"
        f"assert Path({str(sample)!r}).read_bytes() == b'private sample'\n"
        f"Path({str(marker)!r}).touch()\n"
        "print('smoke diagnostics')\n"
        + (
            f"Path({str(report)!r}).write_text('{{\"result\": 123}}')\n"
            if failure != "missing-report"
            else ""
        )
        + ("raise SystemExit(1)\n" if failure == "cuda" else "")
    )
    # A stale report must not turn a run that writes no report into a success.
    report.write_text('{"result": "stale"}')
    if failure == "download":
        http_transfer["get_status"] = 403
    http_transfer["put_failures"] = 3 if failure == "upload" else 1
    job = {
        "run_id": "fixture-run",
        "sample_url": http_transfer["url"] + "/sample?secret=do-not-log",
        "sample_sha256": "wrong"
        if failure == "checksum"
        else hashlib.sha256(b"private sample").hexdigest(),
        "result_url": http_transfer["url"] + "/result?secret=do-not-log",
        "hold_seconds": 65,
    }
    clock = Clock()
    monkeypatch.setattr(remote, "time", clock)
    monkeypatch.setattr(
        remote,
        "execute",
        partial(
            remote.execute, sample=sample, report=report, command=[sys.executable, str(script)]
        ),
    )
    assert remote.main(job) == (1 if failure else 0)
    assert clock.now == 65
    assert marker.exists() == (failure not in {"checksum", "download"})
    result = http_transfer["result"]
    if failure == "upload":
        assert result is None
        assert http_transfer["put_failures"] == 0
        return
    assert result["run_id"] == "fixture-run"
    assert result["outcome"] == ("failed" if failure else "passed")
    assert http_transfer["content_type"] == "application/json"
    assert "do-not-log" not in json.dumps(result)
    if failure is None:
        assert result["report"] == {"result": 123}
        assert "smoke diagnostics" in result["log"]


def test_container_reports_subprocess_timeout(http_transfer, tmp_path, monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("smoke", 300)

    monkeypatch.setattr(remote.subprocess, "run", timeout)
    result = remote.execute(
        {
            "run_id": "fixture-run",
            "sample_url": http_transfer["url"],
            "sample_sha256": hashlib.sha256(http_transfer["sample"]).hexdigest(),
        },
        sample=tmp_path / "sample",
        report=tmp_path / "report",
    )
    assert result["outcome"] == "failed"
    assert "TimeoutExpired" in result["log"]
