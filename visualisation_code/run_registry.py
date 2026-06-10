"""Run-archive utilities for NN-only training/evaluation workflows."""

from __future__ import annotations

import csv
import json
import math
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

NN_ONLY_RUNS_DIR = Path("outputs") / "runs" / "nn_only"


def _angle_tag(angle_deg: float) -> str:
    """Format an angle for run-id path segments (filesystem-friendly)."""
    rounded = round(float(angle_deg), 1)
    return str(rounded).replace("-", "m").replace(".", "p")


def create_nn_only_run_dir(
    population_size: int,
    generations: int,
    initial_angle_rad: float,
) -> tuple[str, str, Path]:
    """Create and return a unique NN-only run directory and identifiers.

    Returns
    -------
    run_id:
        Folder-style run identifier.
    timestamp:
        UTC-local timestamp string used in run_id.
    run_dir:
        Created output path under outputs/runs/nn_only.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    angle_deg = math.degrees(float(initial_angle_rad))
    run_id = f"{timestamp}__pop-{int(population_size)}__gen-{int(generations)}__angle-{_angle_tag(angle_deg)}"
    run_dir = NN_ONLY_RUNS_DIR / run_id

    (run_dir / "figures" / "training").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures" / "evaluation").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata").mkdir(parents=True, exist_ok=True)
    return run_id, timestamp, run_dir


def snapshot_config(config_path: Path | None, run_dir: Path) -> Path:
    """Persist a config snapshot into run metadata."""
    dest = run_dir / "metadata" / "config_snapshot.yaml"
    if config_path is not None and config_path.exists():
        shutil.copy2(config_path, dest)
    else:
        dest.write_text("# Config source file unavailable for this run.\n", encoding="utf-8")
    return dest


def _run_metadata_paths(run_dir: Path) -> tuple[Path, Path]:
    metadata_dir = run_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return metadata_dir / "run_metadata.yaml", metadata_dir / "run_metadata.json"


def write_run_metadata(run_dir: Path, metadata: Dict[str, Any]) -> None:
    """Write metadata as both YAML and JSON."""
    yaml_path, json_path = _run_metadata_paths(run_dir)
    yaml_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def read_run_metadata(run_dir: Path) -> Dict[str, Any]:
    """Load metadata (prefers JSON, falls back to YAML)."""
    yaml_path, json_path = _run_metadata_paths(run_dir)
    if json_path.exists():
        return json.loads(json_path.read_text(encoding="utf-8"))
    if yaml_path.exists():
        return yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    return {}


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def update_run_metadata(run_dir: Path, patch: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge metadata patch and persist YAML+JSON."""
    current = read_run_metadata(run_dir)
    merged = _deep_merge(current, patch)
    write_run_metadata(run_dir, merged)
    return merged


def append_nn_only_index(row: Dict[str, Any]) -> Path:
    """Append one row to outputs/runs/nn_only/index.csv."""
    index_path = NN_ONLY_RUNS_DIR / "index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "run_id",
        "timestamp",
        "pop",
        "generations",
        "initial_angle_deg",
        "target_band",
        "noise_band",
        "best_fitness",
        "run_dir",
    ]

    write_header = not index_path.exists()
    with open(index_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in columns})
    return index_path
