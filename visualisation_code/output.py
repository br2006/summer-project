"""Utilities for managing visualisation output directories."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

BASE_OUTPUT_DIR = Path("outputs") / "figures"


def get_output_dir(scope: str, subdir: Optional[str] = None, create: bool = True) -> Path:
    """Return the canonical output directory for generated figures.

    Parameters
    ----------
    scope:
        Top-level workflow identifier (e.g. "training", "evaluation", "demo").
    subdir:
        Optional nested folder for finer-grained grouping (e.g. run id).
    create:
        When True (default), ensure the directory exists before returning it.
    """
    if not scope:
        raise ValueError("scope must be a non-empty string")

    path = BASE_OUTPUT_DIR / scope
    if subdir:
        path = path / subdir

    if create:
        path.mkdir(parents=True, exist_ok=True)

    return path
