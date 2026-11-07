"""SSH connections and the remote Docker smoke-test script."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from . import RunError


def connection(instance: dict) -> tuple[str, int]:
    """Reject unusable connection details before passing them to OpenSSH."""
    host = instance.get("ssh_host")
    port = instance.get("ssh_port")
    if (
        not isinstance(host, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", host)
        or not isinstance(port, int)
        or isinstance(port, bool)
        or not 1 <= port <= 65535
    ):
        raise RunError("Vast returned invalid SSH connection details.")
    return host, port


def ssh_arguments(directory: Path, host: str, port: int, *, copy: bool = False) -> list[str]:
    return [
        "scp" if copy else "ssh",
        "-P" if copy else "-p",
        str(port),
        "-i",
        str(directory / "id_ed25519"),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={directory / 'known_hosts'}",
        *([] if copy else [f"root@{host}"]),
    ]


def remote_script(remote_directory: str, image_tag: str, image_id: str) -> str:
    """Run the existing Compose service with host permissions matching its UID."""
    return f"""set -eu
cd {shlex.quote(remote_directory)}
test "$(id -u)" -eq 0
nvidia-smi
docker compose version
docker load --input image.tar
test "$(docker image inspect --format '{{{{.Id}}}}' {shlex.quote(image_tag)})" = {shlex.quote(image_id)}
docker tag {shlex.quote(image_tag)} qwen-commit-cuda:local
mkdir -p datasets artifacts
mv train.parquet datasets/train.parquet
chmod 755 . datasets
chmod 444 datasets/train.parquet
chown 10001:10001 artifacts
chmod 700 artifacts
export QWEN_COMMIT_DATASET_DIR="$PWD/datasets"
export QWEN_COMMIT_ARTIFACT_DIR="$PWD/artifacts"
timeout --signal=TERM 300 docker compose --file compose.yaml run --rm --no-build cuda-experiment
"""
