"""Minimal ``.env`` loader — no external dependency.

Reads ``KEY=VALUE`` pairs from a ``.env`` file at the repo root and sets
them in ``os.environ``. Existing shell env vars win (we never override).

Used by the experiment runners so API keys can live in a gitignored
``.env`` file instead of the user's shell, which keeps them out of
process-inherited env dumps and out of this tool's Bash history.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | str, override: bool = False) -> dict[str, str]:
    """Read ``KEY=VALUE`` pairs from ``path`` into ``os.environ``.

    Returns a dict of keys that were actually set (skipped-for-existing
    are excluded). If ``path`` does not exist, returns an empty dict —
    this is expected for fresh clones; the script fails later with a
    clearer "missing API key" error.
    """
    p = Path(path)
    if not p.exists():
        return {}

    loaded: dict[str, str] = {}
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Drop surrounding quotes, if present.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key:
            continue
        # Skip empty values: a blank `KEY=` line is a placeholder, not a
        # real assignment, so it should not block a later non-empty
        # entry for the same key (e.g., when the .env has both a
        # template stub `ANTHROPIC_API_KEY=` and a real value below).
        if not value:
            continue
        if key in os.environ and os.environ[key] and not override:
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded
