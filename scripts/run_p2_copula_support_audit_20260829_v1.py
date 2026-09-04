from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_copula_support_audit_20260829_v1"
CONFIG_PATH = ROOT / "configs/experiments/p2_copula_support_audit_20260829_v1.json"
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID


class AuditError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank_gaussian(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, method="average")
    probability = (ranks - 0.5) / len(values)
    return norm.ppf(probability)


def audit(data_dir: Path) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise AuditError("config experiment id changed")
    root = data_dir.resolve(strict=True)
    source = (root / config["allowed_source_file"]).resolve(strict=True)
    if source.parent != root or source.name != "observations.csv":
        raise AuditError("observations source escaped P2_DATA_DIR")
    frame = pd.read_csv(
        source,
        usecols=["station", "layer", "time", "temp", "psal", "depth", "nominal_depth"],
    )
    required = {"station", "layer", "time", "temp", "psal", "depth", "nominal_depth"}
    if set(frame.columns) != required:
        raise AuditError("observations schema changed")
    timestamp = pd.to_datetime(frame["time"], errors="raise")
    frame["month"] = timestamp.dt.month
    month_to_season = {
        month: season
        for season, months in config["season_definition"].items()
        for month in months
    }
    frame["season"] = frame["month"].map(month_to_season)
    if frame["season"].isna().any():
        raise AuditError("season mapping is incomplete")
    duplicate_count = int(frame.duplicated(["station", "layer", "time"]).sum())
    complete_per_time = frame.assign(complete=frame[["temp", "psal"]].notna().all(axis=1)).groupby(
        ["station", "time"], observed=True
    )["complete"].all()
    complete_timestamp_count = int(complete_per_time.sum())

    coordinate_receipts: list[dict[str, Any]] = []
    nonfinite = 0
    minimum_unique = None
    for (season, layer), cell in frame.groupby(["season", "layer"], sort=True, observed=True):
        for coordinate in ("temp", "psal"):
            values = cell[coordinate].dropna().to_numpy(dtype=np.float64)
            unique = int(np.unique(values).size)
            minimum_unique = unique if minimum_unique is None else min(minimum_unique, unique)
            transformed = _rank_gaussian(values) if len(values) else np.asarray([], dtype=float)
            current_nonfinite = int((~np.isfinite(transformed)).sum())
            nonfinite += current_nonfinite
            coordinate_receipts.append(
                {
                    "season": str(season),
                    "layer": int(layer),
                    "coordinate": coordinate,
                    "observed_rows": int(len(values)),
                    "unique_values": unique,
                    "nonfinite_rank_gaussian": current_nonfinite,
                    "minimum": float(np.min(values)) if len(values) else None,
                    "maximum": float(np.max(values)) if len(values) else None,
                }
            )
    gates = config["gates"]
    checks = {
        "complete_historical_timestamps_gte_1000": complete_timestamp_count
        >= int(gates["minimum_complete_historical_timestamps"]),
        "minimum_unique_values_per_layer_coordinate_gte_200": int(minimum_unique or 0)
        >= int(gates["minimum_unique_values_per_layer_coordinate"]),
        "duplicate_station_layer_time_rows_eq_0": duplicate_count
        == int(gates["duplicate_station_layer_time_rows"]),
        "nonfinite_rank_gaussian_values_eq_0": nonfinite
        == int(gates["nonfinite_rank_gaussian_values"]),
    }
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": (
            "TRAIN_ONLY_SUPPORT_PASS_QUERY_AUDIT_NOT_AUTHORIZED"
            if all(checks.values())
            else "NO_GO_CLOSE_COPULA_AXIS_TRAIN_SUPPORT"
        ),
        "config_sha256": _sha256(CONFIG_PATH),
        "source": {
            "basename": source.name,
            "bytes": source.stat().st_size,
            "sha256": _sha256(source),
            "rows": int(len(frame)),
        },
        "complete_historical_timestamps": complete_timestamp_count,
        "duplicate_station_layer_time_rows": duplicate_count,
        "minimum_unique_values_per_season_layer_coordinate": int(minimum_unique or 0),
        "nonfinite_rank_gaussian_values": nonfinite,
        "checks": checks,
        "coordinate_receipts": coordinate_receipts,
        "query_dependent_checks": config["query_dependent_checks"],
        "query_dependent_checks_executed": False,
        "model_fit_count": 0,
        "official_input_rows_read": 0,
        "csv_output_count": 0,
        "upload_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    result = audit(args.data_dir)
    if args.write_report:
        REPORT_DIR.mkdir(parents=True, exist_ok=False)
        (REPORT_DIR / "train-only-audit.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
