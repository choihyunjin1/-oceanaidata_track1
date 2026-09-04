"""Independent aggregate/hash QA for the finalized P3 v5 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def audit(*, root: Path, data_dir: Path) -> dict[str, Any]:
    artifact = root / "artifacts/p3_corrected_fixed_long_shrink_v5_full_refit"
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((artifact / "metrics.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    for relative, expected in manifest["output_files"].items():
        path = artifact / relative
        if not path.is_file() or _sha(path) != expected["sha256"]:
            failures.append(f"output_sha:{relative}")
        elif path.stat().st_size != expected["bytes"]:
            failures.append(f"output_bytes:{relative}")
    candidate_path = artifact / "candidate/submission.csv"
    reproduced_path = artifact / "candidate/reproduced_submission.csv"
    candidate = pd.read_csv(candidate_path)
    test_index = pd.read_csv(data_dir / "test_index.csv")
    keys = ["case_id", "station", "lead_h"]
    if candidate_path.read_bytes() != reproduced_path.read_bytes():
        failures.append("candidate_reproduction_bytes")
    if not candidate[keys].equals(test_index[keys]):
        failures.append("candidate_key_order")
    if len(candidate) != 1_200 or candidate["case_id"].nunique() != 200:
        failures.append("candidate_grain")
    prediction = candidate["hs_pred"].to_numpy(float)
    if not np.isfinite(prediction).all() or not np.all((prediction >= 0.0) & (prediction <= 30.0)):
        failures.append("candidate_finite_range")
    oof = pd.read_parquet(
        root / "artifacts/p3_corrected_repeated_forward_catboost_v2/oof.parquet"
    ).reset_index(drop=True)
    truth = oof["target_hs"].to_numpy(float)
    incumbent = oof["final_prediction"].to_numpy(float)
    proposed = oof["routed_prediction"].to_numpy(float).copy()
    active = oof["lead_h"].isin([12, 18, 24]).to_numpy()
    proposed[active] = 0.75 * proposed[active] + 0.25 * oof.loc[active, "persistence"].to_numpy(
        float
    )
    incumbent_rmse = _rmse(truth, incumbent)
    proposed_rmse = _rmse(truth, proposed)
    reported = metrics["evaluation"]
    if abs(incumbent_rmse - reported["incumbent"]["rmse_m"]) > 1e-15:
        failures.append("incumbent_rmse")
    if abs(proposed_rmse - reported["candidate"]["rmse_m"]) > 1e-15:
        failures.append("candidate_rmse")
    fold_deltas: dict[str, float] = {}
    for fold, group in oof.groupby("fold", sort=True, observed=True):
        index = group.index.to_numpy()
        fold_deltas[str(fold)] = _rmse(truth[index], proposed[index]) - _rmse(
            truth[index], incumbent[index]
        )
    if any(abs(fold_deltas[name] - reported["fold_delta_m"][name]) > 1e-15 for name in fold_deltas):
        failures.append("fold_deltas")
    station_deltas: dict[str, float] = {}
    for station, group in oof.groupby("station", sort=True, observed=True):
        index = group.index.to_numpy()
        station_deltas[str(station)] = _rmse(truth[index], proposed[index]) - _rmse(
            truth[index], incumbent[index]
        )
    if any(
        abs(station_deltas[name] - reported["station_delta_m"][name]) > 1e-15
        for name in station_deltas
    ):
        failures.append("station_deltas")
    current = root / "output/2026-08-20/ready/P3_submission.csv"
    current_sha = _sha(current)
    if current_sha != "d89e69b940c90ea1fbecf1e882bee69136255fffb12601d2fc853d032900e5b7":
        failures.append("frozen_current_sha")
    if (
        manifest["candidate_uploaded"]
        or metrics["access_counters_total_attempt"]["upload_attempts"]
    ):
        failures.append("upload_state")
    if metrics["access_counters_total_attempt"]["test_target_or_hidden_label_reads"]:
        failures.append("hidden_label_access")
    return {
        "audited_at": datetime.now().astimezone().isoformat(),
        "status": "PASS_P0_0_P1_0" if not failures else "FAIL",
        "p0_count": len(failures),
        "p1_count": 0,
        "failures": failures,
        "checks": {
            "manifest_output_hashes_and_bytes": not any(
                item.startswith("output_") for item in failures
            ),
            "candidate_and_reproduction_byte_identical": candidate_path.read_bytes()
            == reproduced_path.read_bytes(),
            "candidate_rows": len(candidate),
            "candidate_cases": int(candidate["case_id"].nunique()),
            "candidate_key_order_exact": candidate[keys].equals(test_index[keys]),
            "candidate_finite_range": bool(
                np.isfinite(prediction).all() and np.all((prediction >= 0.0) & (prediction <= 30.0))
            ),
            "corrected_oof_rows": len(oof),
            "corrected_oof_cases": int(oof["anchor_id"].nunique()),
            "incumbent_rmse_m": incumbent_rmse,
            "candidate_rmse_m": proposed_rmse,
            "delta_m": proposed_rmse - incumbent_rmse,
            "strictly_improved_folds": int(sum(value < 0.0 for value in fold_deltas.values())),
            "station_deltas_m": station_deltas,
            "candidate_sha256": _sha(candidate_path),
            "reproduced_candidate_sha256": _sha(reproduced_path),
            "metrics_sha256": _sha(artifact / "metrics.json"),
            "manifest_sha256": _sha(artifact / "manifest.json"),
            "completion_sha256": _sha(artifact / "resume_completion_status.json"),
            "frozen_current_sha256": current_sha,
            "hidden_label_reads": metrics["access_counters_total_attempt"][
                "test_target_or_hidden_label_reads"
            ],
            "upload_attempts": metrics["access_counters_total_attempt"]["upload_attempts"],
        },
        "caveat": (
            "The paired case-bootstrap CI90 includes zero; this is a smaller descriptive "
            "corrected-OOF improvement, not an official hidden-score claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    result = audit(root=root, data_dir=Path(args.data_dir).resolve(strict=True))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS_P0_0_P1_0" else 1


if __name__ == "__main__":
    raise SystemExit(main())
