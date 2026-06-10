"""Regenerate rollout animation GIFs for archived NN-only runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualisation_code.animation import run_demo


def _resolve_run_dirs(runs_root: Path, run_id: str | None) -> list[Path]:
    if run_id:
        run_dir = runs_root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return [run_dir]

    return sorted([p for p in runs_root.iterdir() if p.is_dir()])


def regenerate_archived_animations(
    runs_root: Path,
    run_id: str | None,
    frame_stride: int,
) -> list[Path]:
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root does not exist: {runs_root}")

    regenerated: list[Path] = []
    for run_dir in _resolve_run_dirs(runs_root, run_id):
        genome_path = run_dir / "artifacts" / "best_genome_nn_only.pkl"
        config_path = run_dir / "metadata" / "config_snapshot.yaml"
        animation_path = run_dir / "figures" / "evaluation" / "rollout_animation.gif"

        if not genome_path.exists():
            print(f"[skip] Missing genome: {genome_path}")
            continue
        if not config_path.exists():
            print(f"[skip] Missing config snapshot: {config_path}")
            continue

        animation_path.parent.mkdir(parents=True, exist_ok=True)
        run_demo(
            genome_path=genome_path,
            frame_stride=frame_stride,
            show=False,
            config_path=config_path,
            nn_only=True,
            save_gif_path=animation_path,
        )
        regenerated.append(animation_path)
        print(f"[ok] Regenerated: {animation_path}")

    return regenerated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate rollout_animation.gif for archived NN-only runs"
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT / "outputs" / "runs" / "nn_only",
        help="Root directory containing archived NN-only runs",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Specific run folder name to regenerate (omit to process all runs)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=5,
        help="Render every N simulation ticks (default: 5)",
    )
    args = parser.parse_args()

    regenerated = regenerate_archived_animations(
        runs_root=args.runs_root,
        run_id=args.run_id,
        frame_stride=max(1, int(args.frame_stride)),
    )

    print(f"Done. Regenerated {len(regenerated)} animation file(s).")


if __name__ == "__main__":
    main()
