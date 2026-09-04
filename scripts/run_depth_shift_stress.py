"""Run the frozen-XGBoost deployment-depth counterfactual stress test."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from p1_qc.config import load_config  # noqa: E402
from p1_qc.data import load_dataset  # noqa: E402
from p1_qc.depth_shift_stress import run_depth_shift_stress  # noqa: E402
from p1_qc.experiment import environment_summary, sha256_file, write_json  # noqa: E402
from p1_qc.metrics import group_row_shares  # noqa: E402
from p1_qc.pipeline import load_or_build_features, resolve_data_dir  # noqa: E402


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/p1.toml"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--reference-run",
        type=Path,
        default=Path("artifacts/runs/20260813T153038+0900_cv_378a4e89"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/depth_shift_stress_20260813/result.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    reference_run = args.reference_run.resolve()
    output_path = args.output.resolve()
    config = load_config(config_path)
    data_dir = resolve_data_dir(config, args.data_dir)

    reference_paths = {
        "oof": reference_run / "oof.parquet",
        "metrics": reference_run / "metrics.json",
        "selection": reference_run / "selection.json",
    }
    missing = [name for name, path in reference_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"reference run is missing artifacts: {missing}")

    print("[1/4] loading audited train data", flush=True)
    train_path = data_dir / "train.csv"
    train = load_dataset(train_path, kind="train", audit=True, strict=True)
    test_path = data_dir / "test.csv"
    test = load_dataset(test_path, kind="test", audit=True, strict=True)
    print("[2/4] loading cached offline features", flush=True)
    bundle = load_or_build_features(train, config, kind="train", use_cache=True)

    print("[3/4] reproducing frozen outer folds and running counterfactuals", flush=True)
    import pandas as pd

    reference_oof = pd.read_parquet(reference_paths["oof"])
    result = run_depth_shift_stress(
        train,
        bundle,
        config,
        reference_oof,
        _json(reference_paths["metrics"]),
        _json(reference_paths["selection"]),
        target_group_weights=group_row_shares(test),
    )
    payload = {
        "experiment": {
            "name": "frozen_xgboost_depth_shift_stress",
            "created_at_kst": datetime.now().astimezone().isoformat(),
            "config_path": str(config_path.relative_to(PROJECT_ROOT)),
            "reference_run": str(reference_run.relative_to(PROJECT_ROOT)),
            "output_is_aggregate_only": True,
            "environment": environment_summary(),
        },
        "inputs": {
            "train": {
                "rows": len(train),
                "sha256": sha256_file(train_path),
            },
            "test_group_weights": {
                "rows": len(test),
                "sha256": sha256_file(test_path),
                "use": "station-layer row shares only; no labels exist or are inferred",
            },
            **{
                name: {
                    "path": str(path.relative_to(PROJECT_ROOT)),
                    "sha256": sha256_file(path),
                }
                for name, path in reference_paths.items()
            },
        },
        **result,
    }
    print("[4/4] writing aggregate-only result", flush=True)
    write_json(output_path, payload)

    gors = payload["scenarios"]["gors_depth_100pct_missing"]["aggregate"]
    sors = payload["scenarios"]["sors_layer5_unseen_depth_regime"]["aggregate"]
    print(f"result={output_path}")
    print(
        "G-ORS affected: "
        f"delta_f1={gors['affected']['delta_f1']:+.9f}, "
        f"delta_fpr={gors['affected']['delta_fpr']:+.9f}, "
        f"flip_rate={gors['affected']['flip_rate']:.9f}"
    )
    print(
        "S-ORS L5 affected: "
        f"delta_f1={sors['affected']['delta_f1']:+.9f}, "
        f"delta_fpr={sors['affected']['delta_fpr']:+.9f}, "
        f"flip_rate={sors['affected']['flip_rate']:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
