"""Normalization and safety filters for historical Git commits."""

from __future__ import annotations

import unicodedata
from pathlib import PurePosixPath

_BOT_NAMES = frozenset(
    {
        "dependabot",
        "github-actions",
        "greenkeeper",
        "renovate",
        "semantic-release-bot",
    }
)
_GENERATED_PATH_PATTERNS = (
    "*.designer.cs",
    "*.generated.*",
    "*.g.cs",
    "*.min.css",
    "*.min.js",
    "*.pb.go",
    "*.snap",
    "*.svgz",
    "*.map",
    "build/*",
    "dist/*",
    "node_modules/*",
    "vendor/*",
    "*/__pycache__/*",
    "*/build/*",
    "*/dist/*",
    "*/node_modules/*",
    "*/vendor/*",
)


def normalise_subject(subject: str) -> str:
    """Produce a stable, single-line subject without altering its wording."""
    return " ".join(unicodedata.normalize("NFC", subject).split())


def normalise_patch_text(patch: str) -> str:
    """Normalize Unicode and line endings in Git patch text."""
    return unicodedata.normalize("NFC", patch).replace("\r\n", "\n").replace("\r", "\n")


def is_bot(author_name: str, author_email: str) -> bool:
    """Recognise clear automated commit identities without guessing about people."""
    name = author_name.strip().casefold()
    email = author_email.strip().casefold()
    return "[bot]" in name or name in _BOT_NAMES or email.startswith("dependabot[")


def is_fixup(subject: str) -> bool:
    """Return whether a subject is an autosquash helper commit."""
    return subject.casefold().startswith(("fixup!", "squash!", "amend!"))


def is_generated_only(paths: tuple[str, ...]) -> bool:
    """Return whether every changed path is a known generated artifact."""
    return bool(paths) and all(
        any(PurePosixPath(path).match(pattern) for pattern in _GENERATED_PATH_PATTERNS)
        for path in paths
    )
