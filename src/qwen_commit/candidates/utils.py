"""Small reusable helpers for candidate extraction."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime


def opaque_id(kind: str, value: str) -> str:
    """Create a stable identifier without exposing the source value."""
    return f"{kind}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]}"


def as_utc(value: str) -> str:
    """Convert an ISO-8601 timestamp to a UTC timestamp string."""
    timestamp = datetime.fromisoformat(value.strip()).astimezone(UTC)
    return timestamp.isoformat(timespec="seconds").replace("+00:00", "Z")
