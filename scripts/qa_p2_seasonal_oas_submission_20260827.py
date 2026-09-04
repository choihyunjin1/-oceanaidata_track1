"""Independent QA for the preregistered P2 seasonal OAS submission."""

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
ARTIFACT = REPO / "artifacts" / "p2_seasonal_oas_submission_20260827_v1"
OUTPUT = ARTIFACT / "P2_submission.csv"
READY = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260827_P2_SEASONAL_OAS_TS10_PROJECTED_READY\P2_submission.csv"
)
RECEIPT = ARTIFACT / "receipt.json"
LOCAL = REPO / "artifacts" / "p2_oas_conditional_profile_20260827_v3" / "result.json"
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
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    local = json.loads(LOCAL.read_text(encoding="utf-8"))

    require(list(candidate.columns) == KEYS + ["temp"], "candidate schema")
    require(len(candidate) == 26061, "candidate row count")
    require(candidate[KEYS].equals(test[KEYS]), "candidate key order")
    require(candidate[KEYS].equals(sample[KEYS]), "sample key order")
    require(candidate[KEYS].equals(base[KEYS]), "base key order")
    require(candidate.equals(ready), "ready copy values")
    require(OUTPUT.read_bytes() == READY.read_bytes(), "ready copy bytes")
    values = candidate["temp"].to_numpy(float)
    require(np.isfinite(values).all(), "finite predictions")
    require(((values >= -5) & (values <= 45)).all(), "prediction range")
    difference = values - base["temp"].to_numpy(float)
    rms_difference = float(np.sqrt(np.mean(difference**2)))
    require(rms_difference >= 0.02, "candidate is too close to current U")
    require(sha(OUTPUT) == receipt["outputs"]["canonical"]["sha256"], "receipt output hash")
    require(sha(READY) == receipt["outputs"]["ready"]["sha256"], "receipt ready hash")
    require(receipt["leakage_contract"]["answer_file_read"] is False, "answer read")
    require(receipt["leakage_contract"]["official_gap_hidden_label_reads"] == 0, "hidden label read")

    aggregate = local["aggregate"]
    bootstrap = local["paired_kst_day_bootstrap_blend_0.1"]
    require(aggregate["blend_0.1_projected_delta_rmse"] <= -0.0075, "local projected gate")
    require(bootstrap["ci90_high"] < 0, "bootstrap upper gate")
    same_season = local["folds"]["outer_2024_sep_oct"]
    require(
        all(v["blend_0.1_rmse"] < v["reference_rmse"] for v in same_season["by_layer"].values()),
        "same-season layer gate",
    )
    require(
        all(v["blend_0.1_rmse"] < v["reference_rmse"] for v in same_season["by_week"].values()),
        "same-season week gate",
    )
    require(
        local["folds"]["outer_2025_jul_aug"]["fixed_blends"]["0.1"]
        > local["folds"]["outer_2025_jul_aug"]["reference_rmse"],
        "negative block must remain disclosed",
    )

    result = {
        "schema_version": "p2.seasonal_oas_ts10_projected.independent_qa.20260827.v1",
        "status": "PASS_OFFICIAL_PROBE_ELIGIBLE",
        "rows": len(candidate),
        "sha256": sha(OUTPUT),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "rms_difference_vs_current_u": rms_difference,
        "changed_rows_vs_current_u": int((np.abs(difference) > 1e-12).sum()),
        "local_projected_delta_rmse": aggregate["blend_0.1_projected_delta_rmse"],
        "bootstrap_ci90": [bootstrap["ci90_low"], bootstrap["ci90_high"]],
        "same_season_all_layers_improved": True,
        "same_season_all_weeks_improved": True,
        "negative_block_disclosed": True,
        "answer_file_read": False,
        "official_upload_performed": False,
    }
    QA.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
