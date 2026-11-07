"""Publish the CUDA image and resolve its immutable registry digest."""

import json
import re

from . import RunError
from .artifacts import CUDA_DIRECTORY
from .commands import Commands


def publish(commands: Commands, repository: str, run_id: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9.:-]*/[a-z0-9/._-]+", repository):
        raise RunError("Use an image repository with a registry hostname and no tag or digest.")
    tag = f"{repository}:{run_id.lower()}"
    print(f"Building and pushing {tag}...", flush=True)
    commands.run(
        ["docker", "build", "--platform", "linux/amd64", "-t", tag, str(CUDA_DIRECTORY)],
        timeout=3600,
    )
    commands.run(["docker", "push", tag], timeout=1800)
    digests = json.loads(
        commands.run(
            ["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", tag]
        ).stdout
    )
    for digest in digests or []:
        if re.fullmatch(re.escape(repository) + r"@sha256:[a-f0-9]{64}", digest):
            return digest
    raise RunError("Docker did not return the pushed image's registry digest.")
