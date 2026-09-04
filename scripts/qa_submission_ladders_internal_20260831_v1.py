"""Independent aggregate QA for the 2026-08-31 internal submission scores."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in os.sys.path:
        os.sys.path.insert(0, str(directory))

import run_p1_mstcn_segment_precision_router_retroaudit_20260829_v1 as p1_e150  # noqa: E402

from p2_restore.profile_projection import (  # noqa: E402
    project_profiles_vectorized,
    public_endpoint_frame,
)
from p3_wave.kma_alpha_surface import prepare_oof_frame  # noqa: E402

REPORT_DIR = ROOT / "reports/submission_ladders_internal_validation_20260831_v1"
RESULT_PATH = REPORT_DIR / "result.json"
STRUCTURAL_QA = ROOT / "reports/submission_ladders_20260831_v1/independent-qa.json"
P1_KEYS = ["station", "year", "layer", "time", "fold"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(prediction, dtype=np.int8)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    return float(2 * tp / (2 * tp + fp + fn))


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(prediction) - np.asarray(truth)))))


def independently_recompute_p1() -> dict[str, float]:
    peer_path = (
        ROOT
        / "artifacts/runs/20260813T205237+0900_strat_gate_fixed24h_59f6d5c6/oof.parquet"
    )
    truth_path = ROOT / "artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet"
    peer = pd.read_parquet(
        peer_path, columns=[*P1_KEYS, "probability", "deployment_prediction"]
    )
    truth = pd.read_parquet(truth_path, columns=[*P1_KEYS, "label"])
    e150_rows = []
    for bundle in p1_e150.load_bundles().values():
        part = bundle.frame[P1_KEYS].copy()
        part["reference"] = bundle.raw_candidate
        e150_rows.append(part)
    e150 = pd.concat(e150_rows, ignore_index=True)
    frame = truth.merge(peer, on=P1_KEYS, validate="one_to_one").merge(
        e150, on=P1_KEYS, validate="one_to_one"
    )
    y = frame["label"].to_numpy(np.int8)
    reference = frame["reference"].to_numpy(np.int8)
    peer_label = frame["deployment_prediction"].to_numpy(np.int8)
    probability = frame["probability"].to_numpy(float)
    candidates = {
        "P1_1_PEER_HIGHCONF_UNION": reference | (peer_label & (probability >= 0.5)),
        "P1_2_PEER_FULL_UNION": reference | peer_label,
        "P1_3_PEER_STANDALONE": peer_label,
    }
    baseline = f1(y, reference)
    return {name: f1(y, prediction) - baseline for name, prediction in candidates.items()}


def independently_recompute_p2() -> dict[str, float]:
    prediction_path = (
        ROOT
        / "artifacts/p2_alpha50_supervised_rank1_threeway_crossfit_regime_veto_20260828_v2/scored_predictions_no_truth.parquet"
    )
    observations_path = Path(
        r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore\observations.csv"
    )
    prediction = pd.read_parquet(prediction_path)
    prediction["time"] = pd.to_datetime(prediction["time"], utc=True)
    observations = pd.read_csv(
        observations_path,
        usecols=["station", "time", "layer", "temp"],
        dtype={"station": "string", "time": "string"},
    )
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    truth = observations.loc[
        observations["layer"].isin([2, 3, 4]), ["station", "time", "layer", "temp"]
    ]
    frame = prediction.merge(truth, on=["time", "layer"], validate="one_to_one")
    local = frame["time"].dt.tz_convert("Asia/Seoul")
    active = (((local.dt.dayofyear - 1) // 14).astype(int) == 17).to_numpy()
    anchor = frame["reference"].to_numpy(float)
    correction = frame["candidate"].to_numpy(float) - anchor
    full = anchor + np.where(active, correction, 0.0)
    endpoints = public_endpoint_frame(observations)
    projection_frame = frame[["station", "time", "layer"]]
    reference = project_profiles_vectorized(projection_frame, full, endpoints).prediction
    y = frame["temp"].to_numpy(float)
    baseline = rmse(y, reference)
    result = {}
    for layer in (2, 3, 4):
        disabled = active & frame["layer"].eq(layer).to_numpy()
        values = np.where(disabled, anchor, full)
        candidate = project_profiles_vectorized(
            projection_frame, values, endpoints
        ).prediction
        result[f"P2_{layer - 1}_BIN17_DROP_LAYER{layer}"] = rmse(y, candidate) - baseline
    return result


def independently_recompute_p3() -> dict[str, float]:
    blind = pd.read_parquet(
        ROOT
        / "artifacts/p3_kma_calibrated_longlead_blend_v2/one_shot/blind_predictions.parquet"
    )
    evaluated = pd.read_parquet(ROOT / "artifacts/p3/long_persistence_shrink/oof.parquet")
    frame = prepare_oof_frame(blind, evaluated)
    y = frame["target_hs"].to_numpy(float)
    base = frame["base"].to_numpy(float)
    axis = frame["delta"].to_numpy(float)
    lead = frame["lead_h"].to_numpy(int)

    def make(a18: float, a24: float) -> np.ndarray:
        alpha = np.zeros(len(frame), dtype=float)
        alpha[lead == 18] = a18
        alpha[lead == 24] = a24
        return np.clip(base + alpha * axis, 0.0, 30.0)

    reference = make(0.425, 0.425)
    baseline = rmse(y, reference)
    specs = {
        "P3_1_KMA_A18_0425_A24_0600": (0.425, 0.600),
        "P3_2_KMA_A18_0200_A24_0425": (0.200, 0.425),
        "P3_3_KMA_A18_0200_A24_0600": (0.200, 0.600),
    }
    return {name: rmse(y, make(*alphas)) - baseline for name, alphas in specs.items()}


def main() -> int:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    structural = json.loads(STRUCTURAL_QA.read_text(encoding="utf-8"))
    recomputed = {
        "P1": independently_recompute_p1(),
        "P2": independently_recompute_p2(),
        "P3": independently_recompute_p3(),
    }
    checks: dict[str, bool] = {}
    for problem, records in recomputed.items():
        metric = "delta_f1" if problem == "P1" else "delta_rmse"
        for name, value in records.items():
            expected = float(result["problems"][problem]["candidates"][name][metric])
            checks[f"{problem}:{name}:pooled_delta_exact"] = bool(
                np.isclose(value, expected, rtol=0.0, atol=1e-12)
            )
    expected_decisions = {
        "P1_1_PEER_HIGHCONF_UNION": "INTERNAL_NO_GO",
        "P1_2_PEER_FULL_UNION": "INTERNAL_NO_GO",
        "P1_3_PEER_STANDALONE": "INTERNAL_NO_GO",
        "P2_1_BIN17_DROP_LAYER2": "INTERNAL_NO_GO",
        "P2_2_BIN17_DROP_LAYER3": "INTERNAL_NO_GO",
        "P2_3_BIN17_DROP_LAYER4": "INTERNAL_PASS_STRICT",
        "P3_1_KMA_A18_0425_A24_0600": "INTERNAL_NO_GO",
        "P3_2_KMA_A18_0200_A24_0425": "INTERNAL_SIGNAL_ONLY_UNSTABLE",
        "P3_3_KMA_A18_0200_A24_0600": "INTERNAL_SIGNAL_ONLY_UNSTABLE",
    }
    for problem, payload in result["problems"].items():
        for name, record in payload["candidates"].items():
            checks[f"{problem}:{name}:decision"] = record["decision"] == expected_decisions[name]
    files = [item for values in structural["results"].values() for item in values]
    checks["nine_structural_candidates"] = len(files) == 9
    checks["all_submission_hashes_unchanged"] = all(
        Path(item["path"]).is_file() and sha256(Path(item["path"])) == item["sha256"]
        for item in files
    )
    checks["hidden_truth_zero"] = (
        result["operation_counters"]["official_hidden_truth_rows_read"] == 0
    )
    checks["upload_zero"] = result["operation_counters"]["uploads"] == 0
    qa = {
        "schema_version": "submission_ladders.internal_validation.independent_qa.20260831.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "recomputed_pooled_delta": recomputed,
        "source_hashes": {
            "result": sha256(RESULT_PATH),
            "structural_qa": sha256(STRUCTURAL_QA),
        },
        "submission_eligibility": {
            "official_probe_eligible": ["P2_3_BIN17_DROP_LAYER4"],
            "not_eligible": [
                name for name in expected_decisions if name != "P2_3_BIN17_DROP_LAYER4"
            ],
        },
        "official_hidden_truth_rows_read": 0,
        "uploads": 0,
    }
    (REPORT_DIR / "independent-qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    return 0 if qa["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
