"""Recover status reads without retrying rental creation or hiding other failures."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from test_vast_runner import (
    RUNNER_PATH,
    FakeTools,
    artifacts,
    command_tools,
    fake_transfer,
    runner,
    saved_run,
    terraform_tools,
    valid_model_report,
)

TIMEOUT = (
    'showing instance 123: Get "https://console.vast.ai/api/v0/instances/123": '
    "context deadline exceeded (Client.Timeout exceeded while awaiting headers)"
)
READY = "Error waiting for instance to become ready"
READ = "Error reading instance"
GONE = "Error waiting for instance to be destroyed"


def diagnostic(summary=READY, detail=TIMEOUT):
    # Terraform's JSON UI envelope, captured from a real provider failure.
    return json.dumps(
        {
            "@level": "error",
            "type": "diagnostic",
            "diagnostic": {"severity": "error", "summary": summary, "detail": detail},
        }
    )


class PollingTools(FakeTools):
    def run(self, arguments, **options):
        operation = arguments[2] if arguments[0] == "terraform" and len(arguments) > 2 else None
        if operation == "apply" and "-refresh-only" in arguments:
            self.cloud["refreshes"] += 1
            if self.cloud["refresh_errors"]:
                raise command_tools.CommandError(
                    "refresh failed", self.cloud["refresh_errors"].pop(0)
                )
            return subprocess.CompletedProcess(arguments, 0, "", "")
        if operation == "destroy":
            self.cloud["destroys"] += 1
            if self.cloud["destroy_errors"]:
                # Deletion succeeded remotely, but its confirmation read failed.
                self.cloud["resources"].discard("container")
                raise command_tools.CommandError(
                    "destroy failed", self.cloud["destroy_errors"].pop(0)
                )
        result = super().run(arguments, **options)
        if operation == "apply" and self.cloud["rentals"] and self.cloud["create_error"]:
            error = self.cloud.pop("create_error")
            raise command_tools.CommandError("readiness failed", error)
        return result


@pytest.mark.parametrize("failure", [None, "price", "smoke"])
def test_readiness_recovery_keeps_one_rental_and_price_and_report_checks(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(terraform_tools.time, "sleep", lambda seconds: None)
    cloud = {
        "resources": set(),
        "tracked": set(),
        "rentals": 0,
        "uploaded": [],
        "refreshes": 0,
        "destroys": 0,
        "create_error": diagnostic(),
        "refresh_errors": [
            diagnostic(READ, "showing instance 123: vast.ai: unavailable (status 503)")
        ],
        "destroy_errors": [diagnostic(GONE)],
    }

    def run():
        return runner.run_smoke(
            tmp_path / "no-dataset",
            gpu_names=["RTX 3090"],
            max_hourly_price=0.50,
            timeout_minutes=60,
            runs_directory=tmp_path / "runs",
            check="model",
            commands_type=lambda directory: PollingTools(directory, cloud, failure),
            transfer_type=lambda region: fake_transfer(cloud),
        )

    if failure:
        message = "hourly price limit" if failure == "price" else "container smoke test failed"
        with pytest.raises(runner.RunError, match=message):
            run()
    else:
        report = run()
        assert json.loads(report.read_text()) == valid_model_report()
    assert cloud["rentals"] == 1
    assert cloud["refreshes"] == 2
    assert cloud["destroys"] == 2
    assert cloud["resources"] == cloud["tracked"] == set()
    state = saved_run(tmp_path)
    assert state["outcome"] == ("failed" if failure else "passed")
    assert state["apply_complete"] is True
    assert state["cleanup"] == "complete"


@pytest.mark.parametrize(
    "detail",
    [
        TIMEOUT,
        TIMEOUT.split(": context")[0] + ": EOF",
        TIMEOUT.split(": context")[0] + ": unexpected EOF",
        TIMEOUT.split(": context")[0] + ": read tcp: connection reset by peer",
        TIMEOUT.split(": context")[0]
        + ": read tcp: wsarecv: An existing connection was forcibly closed by the remote host.",
        TIMEOUT.split(": context")[0] + ": i/o timeout",
        TIMEOUT.split(": context")[0] + ": net/http: TLS handshake timeout",
        *[
            f"showing instance 123: vast.ai: unavailable (status {code})"
            for code in (429, 500, 502, 503, 504)
        ],
    ],
)
def test_transient_status_diagnostics_are_recoverable(detail):
    assert terraform_tools._transient_poll_failure(diagnostic(detail=detail), {READY})


@pytest.mark.parametrize(
    "output",
    [
        "",
        "not json",
        "[]",
        '{"@level":"error","diagnostic":null}',
        '{"@level":"error"}',
        '{"@level":"info"}',
        diagnostic(detail=None),
        diagnostic(detail=5),
        diagnostic(summary=[]),
        diagnostic("Error creating instance", TIMEOUT),
        diagnostic(detail="creating instance from offer 5: " + TIMEOUT),
        diagnostic(detail=TIMEOUT.replace("/instances/123", "/instances/456")),
        diagnostic(detail=TIMEOUT.split(": context")[0] + ": context canceled"),
        diagnostic(detail=TIMEOUT.split(": context")[0] + ": certificate is not trusted"),
        diagnostic(detail="instance 123 entered actual status offline and will not reach running"),
        *[
            diagnostic(detail=f"showing instance 123: vast.ai: failure (status {code})")
            for code in (400, 401, 403, 404)
        ],
        diagnostic() + "\n" + diagnostic("S3 bucket creation failed", "network timeout"),
    ],
)
def test_unknown_permanent_and_mixed_failures_are_not_recoverable(output):
    assert not terraform_tools._transient_poll_failure(output, {READY})


@pytest.mark.parametrize("operation,summary", [("apply", READY), ("destroy", GONE)])
def test_persistent_polling_failure_stops_after_three_recovery_attempts(
    tmp_path, monkeypatch, operation, summary
):
    delays = []
    monkeypatch.setattr(terraform_tools.time, "sleep", delays.append)

    class FailingCommands:
        directory = tmp_path
        calls = 0

        def run(self, arguments, **options):
            self.calls += 1
            current_summary = READ if "-refresh-only" in arguments else summary
            raise command_tools.CommandError("still unavailable", diagnostic(current_summary))

    commands = FailingCommands()
    with pytest.raises(command_tools.CommandError, match="still unavailable"):
        terraform_tools.apply_or_destroy(commands, operation, timeout=900)
    assert commands.calls == 4
    assert delays == [10, 20, 30]


def test_polling_recovery_shares_the_original_deadline(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=0)
    timeouts = []
    monkeypatch.setattr(
        terraform_tools,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.now,
            sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
        ),
    )

    class SlowCommands:
        directory = tmp_path

        def run(self, arguments, *, timeout, **options):
            timeouts.append(timeout)
            clock.now += 20
            raise command_tools.CommandError(
                "timed out", diagnostic(READ if "-refresh-only" in arguments else READY)
            )

    with pytest.raises(command_tools.CommandError, match="timed out"):
        terraform_tools.apply_or_destroy(SlowCommands(), "apply", timeout=55)
    assert timeouts == [55, 25]
    assert clock.now == 50  # No further retry or sleep can fit the deadline.


@pytest.mark.parametrize("error", [diagnostic("Error creating instance"), "provider crashed"])
def test_ambiguous_creation_failures_are_propagated_without_another_apply(tmp_path, error):
    class FailedCreate:
        directory = tmp_path
        calls = 0

        def run(self, arguments, **options):
            self.calls += 1
            raise command_tools.CommandError("creation failed", error)

    commands = FailedCreate()
    with pytest.raises(command_tools.CommandError, match="creation failed") as raised:
        terraform_tools.apply_or_destroy(commands, "apply", timeout=900)
    assert raised.value.output == error
    assert commands.calls == 1


@pytest.mark.skipif(
    not shutil.which("terraform") or not os.environ.get("VASTAI_TEST_PLUGIN_DIR"),
    reason="Set VASTAI_TEST_PLUGIN_DIR to a local Terraform provider mirror to run offline integration",
)
def test_real_provider_recovers_failed_create_and_delete_polls_without_rerenting(
    tmp_path, monkeypatch
):
    """Use the pinned provider and a local HTTP API; no cloud credentials or resources."""
    cloud = {"exists": False, "creates": 0, "deletes": 0, "fail_read": False}

    class API(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_PUT(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            cloud.update(exists=True, creates=cloud["creates"] + 1, fail_read=True)
            self.respond(200, {"success": True, "new_contract": 123})

        def do_GET(self):
            if cloud["fail_read"]:
                cloud["fail_read"] = False
                self.respond(503, {"success": False, "msg": "temporarily unavailable"})
                return
            instance = {
                "id": 123,
                "actual_status": "running",
                "intended_status": "running",
                "image_uuid": "test-image",
                "image_runtype": "args",
                "dph_total": 0.25,
                "gpu_name": "RTX 3090",
            }
            self.respond(200, {"instances": instance if cloud["exists"] else None})

        def do_DELETE(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            cloud.update(exists=False, deletes=cloud["deletes"] + 1, fail_read=True)
            self.respond(200, {"success": True})

    server = ThreadingHTTPServer(("127.0.0.1", 0), API)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    configuration = tmp_path / "config"
    configuration.mkdir()
    artifacts.write_json(
        configuration / "main.tf.json",
        {
            "terraform": {
                "backend": {"local": {}},
                "required_providers": {"vastai": {"source": "berops/vastai"}},
            },
            "provider": {"vastai": {"api_url": f"http://127.0.0.1:{server.server_port}"}},
            "resource": {
                "vastai_instance": {"smoke": {"count": 1, "image": "test-image", "offer_id": 1}}
            },
            "output": {
                "instance": {
                    "value": {
                        "id": "${vastai_instance.smoke[0].id}",
                        "hourly_price": "${vastai_instance.smoke[0].dph_total}",
                    }
                }
            },
        },
    )
    shutil.copyfile(
        RUNNER_PATH.parent / "vast/.terraform.lock.hcl", configuration / ".terraform.lock.hcl"
    )
    monkeypatch.setattr(terraform_tools, "TERRAFORM_DIRECTORY", configuration)
    monkeypatch.setattr(terraform_tools.time, "sleep", lambda seconds: None)
    monkeypatch.setenv("VAST_API_KEY", "local-test-only")
    directory = tmp_path / "run"
    (directory / "terraform").mkdir(parents=True)
    artifacts.write_json(directory / "terraform/run.auto.tfvars.json", {})
    commands = command_tools.Commands(directory)
    try:
        terraform_tools.terraform(
            commands,
            "init",
            "-input=false",
            "-no-color",
            "-plugin-dir=" + os.environ["VASTAI_TEST_PLUGIN_DIR"],
        )
        terraform_tools.apply_or_destroy(commands, "apply", timeout=60)
        instance = json.loads(
            terraform_tools.terraform(commands, "output", "-json", "instance").stdout
        )
        assert instance == {"id": 123, "hourly_price": 0.25}
        assert cloud["exists"] is True
        assert cloud["creates"] == 1
        state = {"apply_complete": True}
        terraform_tools.cleanup(commands, state)
        assert state["cleanup"] == "complete"
        assert not cloud["exists"]
        assert cloud["creates"] == cloud["deletes"] == 1
        assert terraform_tools.terraform(commands, "state", "list").stdout.strip() == ""
        logs = "\n".join(path.read_text() for path in directory.glob("*-terraform.log"))
        assert READY in logs and GONE in logs
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
