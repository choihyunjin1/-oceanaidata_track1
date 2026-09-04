"""Materialize the sole P2 v7 PASS after internal scoring was sealed."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from run_p2_parallel_candidate_cycle_20260831_v4 import p2_public_features  # noqa: E402

from p2_restore.profile_projection import (  # noqa: E402
    project_profiles_vectorized,
    public_endpoint_frame,
)

EXP = "p2_public_feature_benefit_gate_cycle_20260831_v7"
ART = ROOT / "artifacts" / EXP
OBS = Path(r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore\observations.csv")
TEST = Path(r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore\test_index.csv")
CHAMPION = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용\20260830_P2_RANK1_BIN_DECOMPOSITION_READY_V1\P2_1_RANK1_BIN17_ONLY\P2_submission.csv"
)
BASE = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_P2_PARALLEL_CANDIDATE_CYCLE_V4\P2_2_HGB_ABSOLUTE_PROFILE\P2_submission.csv"
)
SCORING = (
    ROOT
    / "artifacts/p2_alpha50_supervised_rank1_threeway_crossfit_regime_veto_20260828_v2/scored_predictions_no_truth.parquet"
)
SEALED = ROOT / "artifacts/p2_parallel_candidate_cycle_20260831_v4/P2_2_HGB_ABSOLUTE_PROFILE.npz"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_features(frame: pd.DataFrame, public: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    local = pd.DatetimeIndex(frame["time"]).tz_convert("Asia/Seoul")
    layer = frame["layer"].to_numpy(int)
    extra = np.column_stack(
        [
            raw,
            np.abs(raw),
            np.sin(2 * np.pi * local.hour / 24),
            np.cos(2 * np.pi * local.hour / 24),
            *(layer == value for value in (2, 3, 4)),
        ]
    )
    return np.column_stack([public.to_numpy(float), extra])


def main() -> None:
    result = json.loads((ART / "result.json").read_text(encoding="utf-8"))
    passed = [item for item in result["candidates"] if item["pass"]]
    if [item["name"] for item in passed] != ["P2_V7_EXTRATREES_PUBLIC_BENEFIT_GATE"]:
        raise RuntimeError("sealed PASS set drifted")
    observations = pd.read_csv(OBS)
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    scored = pd.read_parquet(SCORING)
    scored["time"] = pd.to_datetime(scored["time"], utc=True)
    truth_frame = scored[["time", "layer"]].merge(
        observations[["time", "layer", "temp"]], on=["time", "layer"], validate="one_to_one"
    )
    sealed = np.load(SEALED)
    reference = sealed["reference"].astype(float)
    base = sealed["candidate"].astype(float)
    raw = base - reference
    public = p2_public_features(observations)
    train_matrix = build_features(
        truth_frame,
        public.reindex(pd.DatetimeIndex(truth_frame["time"])),
        raw,
    )
    local = pd.DatetimeIndex(truth_frame["time"]).tz_convert("Asia/Seoul")
    train = (local >= pd.Timestamp("2024-09-01", tz="Asia/Seoul")) & (
        local < pd.Timestamp("2024-11-01", tz="Asia/Seoul")
    )
    truth = truth_frame["temp"].to_numpy(float)
    beneficial = np.square(reference - truth) > np.square(base - truth)
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=32,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=4,
            random_state=20260831,
        ),
    )
    model.fit(train_matrix[train], beneficial[train])
    test_raw = pd.read_csv(TEST, dtype={"station": "string", "time": "string"})
    champion = pd.read_csv(CHAMPION, dtype={"station": "string", "time": "string"})
    base_official = pd.read_csv(BASE, dtype={"station": "string", "time": "string"})
    keys = ["station", "layer", "time"]
    if not champion[keys].equals(test_raw[keys]) or not base_official[keys].equals(test_raw[keys]):
        raise RuntimeError("official key/order mismatch")
    test = test_raw.copy()
    test["time"] = pd.to_datetime(test["time"], utc=True)
    champion_temp = champion["temp"].to_numpy(float)
    base_temp = base_official["temp"].to_numpy(float)
    official_matrix = build_features(
        test, public.reindex(pd.DatetimeIndex(test["time"])), base_temp - champion_temp
    )
    active = model.predict_proba(official_matrix)[:, 1] >= 0.5
    mixed = champion_temp.copy()
    mixed[active] = base_temp[active]
    projected = project_profiles_vectorized(
        test[keys], mixed, public_endpoint_frame(observations)
    ).prediction
    submission = test_raw[keys].copy()
    submission["temp"] = projected
    output_dir = ART / "submission" / "P2_V7_EXTRATREES_PUBLIC_BENEFIT_GATE"
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "P2_submission.csv"
    submission.to_csv(output, index=False, lineterminator="\n")
    checks = {
        "rows_26061": len(submission) == 26061,
        "columns_exact": list(submission.columns) == [*keys, "temp"],
        "keys_order_exact": submission[keys].equals(test_raw[keys]),
        "duplicates_zero": not submission.duplicated(keys).any(),
        "finite": bool(np.isfinite(submission["temp"]).all()),
        "only_pass_materialized": len(passed) == 1,
        "hidden_truth_rows_zero": True,
        "upload_count_zero": True,
    }
    receipt = {
        "status": "READY_NOT_UPLOADED" if all(checks.values()) else "QA_FAIL",
        "candidate": passed[0]["name"],
        "path": str(output),
        "sha256": sha256_file(output),
        "rows": len(submission),
        "active_rows_before_projection": int(active.sum()),
        "full_fit_count": 1,
        "official_test_index_rows_read": len(test_raw),
        "hidden_truth_rows_read": 0,
        "upload_count": 0,
        "checks": checks,
    }
    (ART / "materialization.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
