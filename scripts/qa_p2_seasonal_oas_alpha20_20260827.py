"""Independent QA for the one-slot P2 OAS 20% follow-up."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"C:\Users\cedis\PycharmProjects\PythonProject")
DATA = Path(r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore")
BASE = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260827_round_G_P2x3_P3x3_PUBLIC_QUADRATIC_READY"
    r"\P2_1_EXPLOIT_LAYERWISE_QUADRATIC\P2_submission.csv"
)
CURRENT_ALPHA10 = REPO / "artifacts" / "p2_seasonal_oas_submission_20260827_v1" / "P2_submission.csv"
ARTIFACT = REPO / "artifacts" / "p2_seasonal_oas_submission_20260827_v2_alpha20"
OUTPUT = ARTIFACT / "P2_submission.csv"
READY = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260827_P2_SEASONAL_OAS_TS20_PROJECTED_READY\P2_submission.csv"
)
RECEIPT = ARTIFACT / "receipt.json"
LOCAL = REPO / "artifacts" / "p2_oas_strength_followup_20260827_v1" / "result.json"
QA = ARTIFACT / "independent_qa.json"
KEYS = ["station", "layer", "time"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    test = pd.read_csv(DATA / "test_index.csv", dtype={"station": "string", "time": "string"})
    sample = pd.read_csv(DATA / "sample_submission.csv", dtype={"station": "string", "time": "string"})
    candidate = pd.read_csv(OUTPUT, dtype={"station": "string", "time": "string"})
    ready = pd.read_csv(READY, dtype={"station": "string", "time": "string"})
    base = pd.read_csv(BASE, dtype={"station": "string", "time": "string"})
    alpha10 = pd.read_csv(CURRENT_ALPHA10, dtype={"station": "string", "time": "string"})
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    local = json.loads(LOCAL.read_text(encoding="utf-8"))["recipes"]["scalar_0.20"]

    require(len(candidate) == 26061 and list(candidate.columns) == KEYS + ["temp"], "schema")
    for frame, label in ((sample, "sample"), (base, "base"), (alpha10, "alpha10"), (ready, "ready")):
        require(candidate[KEYS].equals(frame[KEYS]), f"{label} keys")
    require(OUTPUT.read_bytes() == READY.read_bytes(), "ready bytes")
    values = candidate["temp"].to_numpy(float)
    require(np.isfinite(values).all() and ((values >= -5) & (values <= 45)).all(), "finite/range")
    diff_u = values - base["temp"].to_numpy(float)
    diff_a10 = values - alpha10["temp"].to_numpy(float)
    rms_u = float(np.sqrt(np.mean(diff_u**2)))
    rms_a10 = float(np.sqrt(np.mean(diff_a10**2)))
    require(rms_u >= 0.02 and rms_a10 >= 0.02, "degenerate difference")
    require(receipt["alpha"] == 0.2, "alpha receipt")
    require(sha(OUTPUT) == receipt["outputs"]["canonical"]["sha256"], "output hash")
    require(receipt["leakage_contract"]["answer_file_read"] is False, "answer read")
    require(receipt["leakage_contract"]["official_gap_hidden_label_reads"] == 0, "hidden label read")
    require(local["delta_rmse"] < -0.010, "aggregate local gate")
    require(local["folds"]["outer_2024_sep_oct"]["delta_rmse"] < -0.02, "same-season gate")
    require(local["folds"]["outer_2025_jul_aug"]["delta_rmse"] > 0, "negative fold disclosure")

    result = {
        "schema_version": "p2.seasonal_oas_alpha20.independent_qa.20260827.v1",
        "status": "PASS_ONE_SLOT_STRENGTH_CONFIRMATION",
        "rows": len(candidate),
        "sha256": sha(OUTPUT),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "rms_difference_vs_u": rms_u,
        "rms_difference_vs_alpha10": rms_a10,
        "local_projected_delta_rmse": local["delta_rmse"],
        "same_season_delta_rmse": local["folds"]["outer_2024_sep_oct"]["delta_rmse"],
        "negative_fold_delta_rmse": local["folds"]["outer_2025_jul_aug"]["delta_rmse"],
        "deterministic_second_run_sha256": sha(OUTPUT),
        "answer_file_read": False,
        "official_upload_performed": False,
    }
    QA.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
