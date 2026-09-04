"""Technical contract repair for the zero-fit P2 v9 experiment.

Scientific candidates and every hyperparameter remain unchanged.  Only four
public layer-8 columns that are all-missing in the frozen training window are
deterministically represented as zero after their presence indicators.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_domain_invariant_vertical_curvature_20260901_v9 as engine  # noqa: E402

EXPERIMENT_ID = "p2_domain_invariant_vertical_curvature_20260901_v9r1"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)
EXPECTED_ALL_MISSING = {
    "temp_offset_l8",
    "psal_anomaly_l8",
    "depth_offset_l8",
    "nominal_offset_l8",
}

engine.EXPERIMENT_ID = EXPERIMENT_ID
engine.CONFIG = CONFIG
engine.ARTIFACT = ARTIFACT
engine.REPORT = REPORT
engine.RUNNER = RUNNER
engine.SCHEMA_VERSION = "p2.domain_invariant_vertical_curvature.result.20260901.v9r1"
_ORIGINAL_WRITE_REPORT = engine.write_report


def deterministic_finite_column_median(frame: pd.DataFrame) -> pd.Series:
    """Use zero only for the four sealed all-missing layer-8 columns."""

    cleaned = frame.replace([float("inf"), float("-inf")], pd.NA)
    median = cleaned.median(axis=0, numeric_only=True)
    all_missing = set(median.index[median.isna()])
    if not all_missing.issubset(EXPECTED_ALL_MISSING):
        raise engine.ContractError(
            f"unexpected all-missing feature columns: {sorted(all_missing)}"
        )
    return median.fillna(0.0)


def write_report(result: dict[str, Any]) -> None:
    _ORIGINAL_WRITE_REPORT(result)
    path = REPORT / "report-source.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "exploratory cycle 20260901 v9",
        "exploratory cycle 20260901 v9r1",
        1,
    )
    repair = (
        "\n## 기술 contract repair\n\n"
        "v9은 첫 model.fit 전 0-fit으로 종료됐다. v9r1은 presence indicator가 이미 0인 "
        "training-window all-missing layer-8 feature 네 열만 deterministic zero로 바꿨다. "
        "후보, split, alpha, 0.8/0.2 blend, gate, 4-MAD winsor는 변경하지 않았다.\n"
    )
    path.write_text(text + repair, encoding="utf-8")


engine._finite_column_median = deterministic_finite_column_median
engine.write_report = write_report


def preflight() -> dict[str, Any]:
    payload = engine.preflight()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    repair = config["contract_repair"]
    if set(repair["all_missing_columns"]) != EXPECTED_ALL_MISSING:
        raise engine.ContractError("all-missing column allow-list drift")
    if repair["candidate_feature_split_alpha_blend_gate_winsor_changed"]:
        raise engine.ContractError("scientific contract changed")
    payload["contract_repair"] = {
        "predecessor": repair["predecessor"],
        "all_missing_columns": sorted(EXPECTED_ALL_MISSING),
        "deterministic_fill": 0.0,
        "scientific_contract_changed": False,
    }
    payload["preflight_sha256"] = engine.sha256_json(
        {key: value for key, value in payload.items() if key != "preflight_sha256"}
    )
    return payload


def run() -> dict[str, Any]:
    result = engine.run()
    atomic_payload = {
        "predecessor": "p2_domain_invariant_vertical_curvature_20260901_v9",
        "predecessor_model_fits": 0,
        "predecessor_predictions": 0,
        "repair": "four sealed training-window all-missing layer-8 feature columns filled with deterministic zero after presence indicators",
        "all_missing_columns": sorted(EXPECTED_ALL_MISSING),
        "candidate_feature_split_alpha_blend_gate_winsor_changed": False,
        "official_rows_read": 0,
    }
    engine.atomic_json(ARTIFACT / "technical_recovery.json", atomic_payload)
    engine.atomic_json(REPORT / "technical_recovery.json", atomic_payload)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.execute:
        raise SystemExit("Choose exactly one of --preflight or --execute")
    payload = preflight() if args.preflight else run()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
