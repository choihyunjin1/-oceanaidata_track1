"""Run the fixed P2 method screen and write aggregate-only evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.features import build_training_features
from p2_restore.research import (
    append_public_dynamics,
    diagnostic_summary,
    paired_day_bootstrap,
    run_method_screen,
    run_seasonal_stability_screen,
    select_lean_m2_dynamics,
    stability_diagnostics,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/p2_method_scout/result.json")
    )
    parser.add_argument(
        "--status-file", type=Path, default=Path("artifacts/progress/p2_method_scout.json")
    )
    args = parser.parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.status_file.parent.mkdir(parents=True, exist_ok=True)

    def status(percent: int, phase: str) -> None:
        args.status_file.write_text(
            json.dumps(
                {
                    "task": "P2 method reconnaissance",
                    "state": "running" if percent < 100 else "complete",
                    "percent": percent,
                    "phase": phase,
                    "updated_at": datetime.now().astimezone().isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    status(5, "loading and auditing source data")
    data = load_p2_data(data_dir)
    status(20, "building public-only base features")
    base = build_training_features(data.observations)
    status(35, "building public-layer dynamics")
    dynamic = append_public_dynamics(base, data.observations)
    lean = select_lean_m2_dynamics(base, dynamic)
    status(40, "running fixed blocked method screen")
    blocks, outputs = run_method_screen(base, dynamic, data.observations)
    status(65, "running 11-block lean-feature stability screen")
    stability, stability_oof = run_seasonal_stability_screen(base, lean)
    status(90, "building aggregate diagnostics and bootstrap")
    artifact = {
        "created_at": datetime.now().astimezone().isoformat(),
        "research_only": True,
        "external_observations_used": False,
        "method_contract": {
            "validation_blocks": list(blocks),
            "methods": list(next(iter(blocks.values()))["methods"]),
            "dynamic_steps": [6, 36, 75, 144],
            "eof_rank": 3,
            "ridge_alpha": 10.0,
            "physics_blend_contrast_c": [0.2, 2.0],
        },
        "source_hashes": {
            name: _sha256(data_dir / name)
            for name in ("observations.csv", "test_index.csv", "baseline_interp.csv")
        },
        "blocks": blocks,
        "diagnostics": diagnostic_summary(outputs),
        "stability_blocks": stability,
        "stability_bootstrap": {
            candidate: paired_day_bootstrap(stability_oof, candidate=candidate)
            for candidate in ("lean_m2", "blend50")
        },
        "stability_diagnostics": stability_diagnostics(stability_oof),
    }
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    (args.output.with_suffix(args.output.suffix + ".sha256")).write_text(
        _sha256(args.output) + "\n", encoding="ascii"
    )
    status(100, "complete")
    print(json.dumps({"output": args.output.as_posix(), "sha256": _sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
