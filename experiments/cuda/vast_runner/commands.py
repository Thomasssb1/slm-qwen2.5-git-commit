"""Bounded subprocess execution and per-operation logs."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from . import RunError


class Commands:
    """Run tools with bounded waits and private, per-operation logs."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.sequence = max(
            (int(path.name.split("-", 1)[0]) for path in directory.glob("[0-9]*-*.log")),
            default=0,
        )
        self.environment = os.environ.copy()
        self.api_key = os.environ.get("VAST_API_KEY") or os.environ.get("VASTAI_API_KEY")
        for name in ("VAST_API_KEY", "VASTAI_API_KEY"):
            self.environment.pop(name, None)

    def run(
        self,
        arguments: list[str],
        *,
        timeout: float = 300,
        stdin: str | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        self.sequence += 1
        log_path = self.directory / f"{self.sequence:02d}-{Path(arguments[0]).stem}.log"
        if timeout <= 0:
            raise RunError("The rental time limit expired.")
        environment = self.environment.copy()
        if Path(arguments[0]).stem == "terraform":
            environment.update(TF_IN_AUTOMATION="1", TF_INPUT="0")
            if self.api_key:
                environment["VASTAI_API_KEY"] = self.api_key
        with log_path.open("w", encoding="utf-8") as log:
            # Give Terraform a chance to flush state before attempting cleanup.
            process = subprocess.Popen(
                arguments,
                cwd=self.directory,
                env=environment,
                stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                start_new_session=os.name != "nt",
            )
            try:
                process.communicate(input=stdin, timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                try:
                    if os.name == "nt":
                        process.send_signal(signal.CTRL_BREAK_EVENT)
                    else:
                        os.killpg(process.pid, signal.SIGINT)
                except OSError:
                    # Windows jobs without a console cannot receive console signals.
                    process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        output = log_path.read_text(encoding="utf-8", errors="replace")
        if check and process.returncode:
            raise RunError(f"{Path(arguments[0]).name} failed; see {log_path}")
        return subprocess.CompletedProcess(arguments, process.returncode, output, "")
