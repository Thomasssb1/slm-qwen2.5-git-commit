"""Normalization and safety filters for historical Git commits."""

from __future__ import annotations

import unicodedata
from pathlib import PurePosixPath

_GENERATED_FILE_PATTERNS = (
    "*.designer.cs",
    "*.generated.*",
    "*.g.cs",
    "*.min.css",
    "*.min.js",
    "*.pb.go",
    "*.snap",
    "*.svgz",
    "*.map",
)
_GENERATED_DIRECTORY_NAMES = frozenset({"__pycache__", "build", "dist", "node_modules", "vendor"})


def normalise_subject(subject: str) -> str:
    """Produce a stable, single-line subject without altering its wording."""
    return " ".join(unicodedata.normalize("NFC", subject).split())


def normalise_patch_text(patch: str) -> str:
    """Normalize Unicode and line endings in Git patch text."""
    return unicodedata.normalize("NFC", patch).replace("\r\n", "\n").replace("\r", "\n")


def is_bot(author_name: str, bot_names: tuple[str, ...]) -> bool:
    """Recognise clear automated commit identities without guessing about people."""
    name = author_name.strip().casefold()
    configured_names = frozenset(bot_name.casefold() for bot_name in bot_names)
    return name in configured_names


def is_fixup(subject: str) -> bool:
    """Return whether a subject is an autosquash helper commit."""
    return subject.casefold().startswith(("fixup!", "squash!", "amend!"))


def is_generated_only(paths: tuple[str, ...]) -> bool:
    """Return whether every changed path is a known generated artifact."""
    return bool(paths) and all(_is_generated_path(path) for path in paths)


def _is_generated_path(path: str) -> bool:
    parsed_path = PurePosixPath(path)
    return bool(
        _GENERATED_DIRECTORY_NAMES.intersection(parsed_path.parts)
        or any(parsed_path.match(pattern) for pattern in _GENERATED_FILE_PATTERNS)
    )
