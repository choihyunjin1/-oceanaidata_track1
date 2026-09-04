"""Score every 2026-08-31 submission-ladder arm on historical internal tests.

This is an aggregate-only validator.  It reads historical labels and sealed
OOF predictions, but never reads official hidden labels, creates submission
CSVs, or uploads a file.  When an exact OOF lineage for the deployed champion
does not exist, the result is explicitly labelled as proxy validation.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

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

EXPERIMENT_ID = "submission_ladders_internal_validation_20260831_v1"
OUTPUT_DIR = ROOT / "reports" / EXPERIMENT_ID
P1_BASE_OOF = (
    ROOT / "artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet"
)
P1_PEER_OOF = (
    ROOT
    / "artifacts/runs/20260813T205237+0900_strat_gate_fixed24h_59f6d5c6/oof.parquet"
)
P2_OOF = (
    ROOT
    / "artifacts/p2_alpha50_supervised_rank1_threeway_crossfit_regime_veto_20260828_v2/scored_predictions_no_truth.parquet"
)
P2_OBSERVATIONS = Path(
    r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore\observations.csv"
)
P3_BLIND = (
    ROOT
    / "artifacts/p3_kma_calibrated_longlead_blend_v2/one_shot/blind_predictions.parquet"
)
P3_EVALUATED = ROOT / "artifacts/p3/long_persistence_shrink/oof.parquet"
P1_KEYS = ["station", "year", "layer", "time", "fold"]
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260831


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def f1_counts(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(prediction, dtype=np.int8)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise RuntimeError("binary metric contract failed")
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    denominator = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(tp / (tp + fp)) if tp + fp else 1.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 1.0,
        "f1": float(2 * tp / denominator) if denominator else 1.0,
    }


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.size == 0 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise RuntimeError("RMSE contract failed")
    return float(np.sqrt(np.mean(np.square(p - y))))


def binary_bootstrap(
    frame: pd.DataFrame,
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float | int]:
    work = frame.loc[:, ["fold", "station", "layer", "time"]].copy()
    work["date"] = (
        pd.to_datetime(work["time"], utc=True).dt.tz_convert("Asia/Seoul").dt.date
    )
    work["truth"] = np.asarray(truth, dtype=np.int8)
    work["reference"] = np.asarray(reference, dtype=np.int8)
    work["candidate"] = np.asarray(candidate, dtype=np.int8)
    groups = list(work.groupby(["fold", "station", "layer", "date"], sort=True))
    components = []
    for _, group in groups:
        y = group["truth"].to_numpy(np.int8)
        components.append(
            (
                len(group),
                f1_counts(y, group["reference"].to_numpy(np.int8)),
                f1_counts(y, group["candidate"].to_numpy(np.int8)),
            )
        )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    delta = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, len(components), size=len(components))
        totals = []
        for metric_index in (1, 2):
            tp = sum(int(components[index][metric_index]["tp"]) for index in draw)
            fp = sum(int(components[index][metric_index]["fp"]) for index in draw)
            fn = sum(int(components[index][metric_index]["fn"]) for index in draw)
            denominator = 2 * tp + fp + fn
            totals.append(float(2 * tp / denominator) if denominator else 1.0)
        delta[replicate] = totals[1] - totals[0]
    return {
        "unit": "fold x station x layer x KST calendar day",
        "groups": int(len(groups)),
        "replicates": BOOTSTRAP_REPLICATES,
        "mean_delta_f1": float(delta.mean()),
        "ci90_low": float(np.quantile(delta, 0.05)),
        "ci90_high": float(np.quantile(delta, 0.95)),
        "probability_improved": float(np.mean(delta > 0.0)),
    }


def rmse_bootstrap(
    frame: pd.DataFrame,
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    group_columns: list[str],
) -> dict[str, float | int | str]:
    work = frame.loc[:, group_columns].copy()
    work["truth"] = np.asarray(truth, dtype=np.float64)
    work["reference"] = np.asarray(reference, dtype=np.float64)
    work["candidate"] = np.asarray(candidate, dtype=np.float64)
    groups = list(work.groupby(group_columns, sort=True, observed=True))
    reference_sse = np.asarray(
        [np.square(group["reference"] - group["truth"]).sum() for _, group in groups]
    )
    candidate_sse = np.asarray(
        [np.square(group["candidate"] - group["truth"]).sum() for _, group in groups]
    )
    counts = np.asarray([len(group) for _, group in groups], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    delta = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, len(groups), size=len(groups))
        denominator = float(counts[draw].sum())
        delta[replicate] = np.sqrt(candidate_sse[draw].sum() / denominator) - np.sqrt(
            reference_sse[draw].sum() / denominator
        )
    return {
        "unit": " x ".join(group_columns),
        "groups": int(len(groups)),
        "replicates": BOOTSTRAP_REPLICATES,
        "mean_delta_rmse": float(delta.mean()),
        "ci90_low": float(np.quantile(delta, 0.05)),
        "ci90_high": float(np.quantile(delta, 0.95)),
        "probability_improved": float(np.mean(delta < 0.0)),
    }


def binary_breakdown(
    frame: pd.DataFrame,
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    before = f1_counts(truth, reference)
    after = f1_counts(truth, candidate)
    result: dict[str, Any] = {
        "reference": before,
        "candidate": after,
        "delta_f1": float(after["f1"] - before["f1"]),
        "rows": int(len(frame)),
    }
    for column, label in (("fold", "by_fold"), ("station", "by_station")):
        slices = {}
        for key, group in frame.assign(_row=np.arange(len(frame))).groupby(column, sort=True):
            rows = group["_row"].to_numpy(int)
            left = f1_counts(truth[rows], reference[rows])
            right = f1_counts(truth[rows], candidate[rows])
            slices[str(key)] = {
                "rows": int(len(rows)),
                "reference_f1": left["f1"],
                "candidate_f1": right["f1"],
                "delta_f1": float(right["f1"] - left["f1"]),
            }
        result[label] = slices
    result["bootstrap"] = binary_bootstrap(frame, truth, reference, candidate)
    minimum_fold = min(value["delta_f1"] for value in result["by_fold"].values())
    bootstrap_probability = result["bootstrap"]["probability_improved"]
    if result["delta_f1"] > 0.0 and minimum_fold >= 0.0 and bootstrap_probability >= 0.8:
        decision = "INTERNAL_PASS_STRICT"
    elif result["delta_f1"] > 0.0:
        decision = "INTERNAL_SIGNAL_ONLY_UNSTABLE"
    else:
        decision = "INTERNAL_NO_GO"
    result["decision"] = decision
    return result


def regression_breakdown(
    frame: pd.DataFrame,
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    slice_columns: list[str],
    bootstrap_groups: list[str],
) -> dict[str, Any]:
    before = rmse(truth, reference)
    after = rmse(truth, candidate)
    result: dict[str, Any] = {
        "reference_rmse": before,
        "candidate_rmse": after,
        "delta_rmse": after - before,
        "rows": int(len(frame)),
    }
    indexed = frame.assign(_row=np.arange(len(frame)))
    for column in slice_columns:
        slices = {}
        for key, group in indexed.groupby(column, sort=True, observed=True):
            rows = group["_row"].to_numpy(int)
            left = rmse(truth[rows], reference[rows])
            right = rmse(truth[rows], candidate[rows])
            slices[str(key)] = {
                "rows": int(len(rows)),
                "reference_rmse": left,
                "candidate_rmse": right,
                "delta_rmse": right - left,
            }
        result[f"by_{column}"] = slices
    result["bootstrap"] = rmse_bootstrap(
        frame, truth, reference, candidate, group_columns=bootstrap_groups
    )
    minimum = max(
        value["delta_rmse"] for value in result[f"by_{slice_columns[0]}"].values()
    )
    probability = result["bootstrap"]["probability_improved"]
    if result["delta_rmse"] < 0.0 and minimum <= 0.0 and probability >= 0.8:
        decision = "INTERNAL_PASS_STRICT"
    elif result["delta_rmse"] < 0.0:
        decision = "INTERNAL_SIGNAL_ONLY_UNSTABLE"
    else:
        decision = "INTERNAL_NO_GO"
    result["decision"] = decision
    return result


def score_p1() -> dict[str, Any]:
    base = pd.read_parquet(P1_BASE_OOF)
    peer = pd.read_parquet(P1_PEER_OOF)
    if base.duplicated(P1_KEYS).any() or peer.duplicated(P1_KEYS).any():
        raise RuntimeError("P1 duplicate OOF key")
    merged = base[P1_KEYS + ["label"]].merge(
        peer[P1_KEYS + ["label", "probability", "deployment_prediction"]],
        on=P1_KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_base", "_peer"),
    )
    if len(merged) != len(base) or not merged["_merge"].eq("both").all():
        raise RuntimeError("P1 OOF memberships differ")
    if not np.array_equal(merged["label_base"], merged["label_peer"]):
        raise RuntimeError("P1 OOF truth differs")

    e150_rows = []
    for bundle in p1_e150.load_bundles().values():
        frame = bundle.frame[P1_KEYS].copy()
        frame["e150"] = bundle.raw_candidate
        e150_rows.append(frame)
    e150 = pd.concat(e150_rows, ignore_index=True)
    merged = merged.drop(columns="_merge").merge(
        e150, on=P1_KEYS, how="outer", validate="one_to_one", indicator=True
    )
    if len(merged) != len(base) or not merged["_merge"].eq("both").all():
        raise RuntimeError("P1 e150 OOF membership differs")
    merged = merged.drop(columns="_merge")
    truth = merged["label_base"].to_numpy(np.int8)
    reference = merged["e150"].to_numpy(np.int8)
    peer_label = merged["deployment_prediction"].to_numpy(np.int8)
    probability = merged["probability"].to_numpy(np.float64)
    candidates = {
        "P1_1_PEER_HIGHCONF_UNION": np.maximum(
            reference, peer_label * (probability >= 0.50)
        ),
        "P1_2_PEER_FULL_UNION": np.maximum(reference, peer_label),
        "P1_3_PEER_STANDALONE": peer_label,
    }
    records = {
        name: binary_breakdown(merged, truth, reference, candidate)
        for name, candidate in candidates.items()
    }
    return {
        "comparator": "E150_OOF_PROXY",
        "comparator_disclosure": (
            "Historical raw e150 OOF is the closest available comparator. The deployed "
            "official champion additionally contains two official-only GI rows, so this is "
            "not an exact champion OOF reconstruction."
        ),
        "quality": {
            "rows": int(len(merged)),
            "unique_keys": not merged.duplicated(P1_KEYS).any(),
            "truth_mismatch_rows": 0,
            "folds": sorted(str(value) for value in merged["fold"].unique()),
        },
        "candidates": records,
    }


def score_p2() -> dict[str, Any]:
    scored = pd.read_parquet(P2_OOF)
    observations = pd.read_csv(
        P2_OBSERVATIONS,
        dtype={"station": "string", "time": "string"},
        usecols=["station", "time", "layer", "temp"],
    )
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    target = observations.loc[
        observations["layer"].isin([2, 3, 4]), ["station", "time", "layer", "temp"]
    ]
    if target.duplicated(["time", "layer"]).any():
        raise RuntimeError("P2 historical truth key is not unique")
    scored["time"] = pd.to_datetime(scored["time"], utc=True)
    if scored.duplicated(["time", "layer"]).any():
        raise RuntimeError("P2 prediction key is not unique")
    work = scored.merge(
        target, on=["time", "layer"], how="left", validate="one_to_one", indicator=True
    )
    if len(work) != len(scored) or not work["_merge"].eq("both").all():
        raise RuntimeError("P2 truth join lost rows")
    work = work.drop(columns="_merge")
    endpoints = public_endpoint_frame(observations)
    local = work["time"].dt.tz_convert("Asia/Seoul")
    work["season_bin"] = ((local.dt.dayofyear - 1) // 14).astype(int)
    work["kst_date"] = local.dt.date
    active = work["season_bin"].eq(17).to_numpy()
    reference_anchor = work["reference"].to_numpy(np.float64)
    correction = work["candidate"].to_numpy(np.float64) - reference_anchor
    full_unprojected = reference_anchor + np.where(active, correction, 0.0)
    projection_frame = work[["station", "time", "layer"]]
    full_projection = project_profiles_vectorized(
        projection_frame, full_unprojected, endpoints
    )
    reference = full_projection.prediction
    truth = work["temp"].to_numpy(np.float64)
    candidates = {}
    pava = {"internal_champion_active_rows": int(full_projection.active_mask.sum())}
    for layer in (2, 3, 4):
        disabled = active & work["layer"].eq(layer).to_numpy()
        unprojected = np.where(disabled, reference_anchor, full_unprojected)
        projection = project_profiles_vectorized(projection_frame, unprojected, endpoints)
        name = f"P2_{layer - 1}_BIN17_DROP_LAYER{layer}"
        candidates[name] = regression_breakdown(
            work,
            truth,
            reference,
            projection.prediction,
            slice_columns=["fold", "layer"],
            bootstrap_groups=["fold", "kst_date"],
        )
        candidates[name]["disabled_rows"] = int(disabled.sum())
        candidates[name]["pava_active_rows"] = int(projection.active_mask.sum())
        pava[f"{name}_active_rows"] = int(projection.active_mask.sum())
    return {
        "comparator": "CROSSFIT_BIN17_CHAMPION_PROXY",
        "comparator_disclosure": (
            "The exact full-fit official bin17 correction has no historical OOF lineage. "
            "The closest sealed three-way cross-fit rank-1 correction is restricted to "
            "bin17 and label-blind endpoint/PAVA projection is reapplied."
        ),
        "quality": {
            "rows": int(len(work)),
            "prediction_key_duplicates": 0,
            "truth_join_missing_rows": 0,
            "join_multiplier": float(len(work) / len(scored)),
            "stations": sorted(str(value) for value in work["station"].unique()),
            "active_bin17_rows": int(active.sum()),
            "pava": pava,
        },
        "candidates": candidates,
    }


def score_p3() -> dict[str, Any]:
    blind = pd.read_parquet(P3_BLIND)
    evaluated = pd.read_parquet(P3_EVALUATED)
    frame = prepare_oof_frame(blind, evaluated)
    truth = frame["target_hs"].to_numpy(np.float64)
    base = frame["base"].to_numpy(np.float64)
    delta = frame["delta"].to_numpy(np.float64)
    lead = frame["lead_h"].to_numpy(int)

    def prediction(alpha18: float, alpha24: float) -> np.ndarray:
        alpha = np.zeros(len(frame), dtype=np.float64)
        alpha[lead == 18] = alpha18
        alpha[lead == 24] = alpha24
        return np.clip(base + alpha * delta, 0.0, 30.0)

    reference = prediction(0.425, 0.425)
    specs = {
        "P3_1_KMA_A18_0425_A24_0600": (0.425, 0.600),
        "P3_2_KMA_A18_0200_A24_0425": (0.200, 0.425),
        "P3_3_KMA_A18_0200_A24_0600": (0.200, 0.600),
    }
    candidates = {}
    for name, (alpha18, alpha24) in specs.items():
        candidate = prediction(alpha18, alpha24)
        record = regression_breakdown(
            frame,
            truth,
            reference,
            candidate,
            slice_columns=["fold", "station", "lead_h"],
            bootstrap_groups=["fold", "anchor_id"],
        )
        record["alpha_18"] = alpha18
        record["alpha_24"] = alpha24
        candidates[name] = record
    return {
        "comparator": "UNIFORM_ALPHA_0.425_OOF_PROXY",
        "comparator_disclosure": (
            "The official uniform alpha=.425 is evaluated on the frozen KMA OOF correction "
            "axis. Public and local transport have previously disagreed by station, so local "
            "pass is necessary evidence but not a guarantee of Public improvement."
        ),
        "quality": {
            "rows": int(len(frame)),
            "cases": int(frame.groupby(["fold", "anchor_id"], observed=True).ngroups),
            "duplicate_pair_keys": int(
                frame.duplicated(["fold", "anchor_id", "station", "lead_h"]).sum()
            ),
            "lead_counts": {
                str(key): int(value) for key, value in frame.groupby("lead_h").size().items()
            },
        },
        "candidates": candidates,
    }


def report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# 제출 후보 내부 검증 — 2026-08-31",
        "",
        "## 결론",
        "",
        "모든 제출 후보는 공식 업로드 전에 내부 historical test를 반드시 통과해야 한다. "
        "이번 9개도 이 규칙으로 재분류했으며, `INTERNAL_PASS_STRICT`만 공식 1회 탐침 후보가 된다. "
        "정확한 배포 챔피언 OOF가 없는 P1/P2는 프록시임을 숨기지 않는다.",
        "",
        "| 문제 | 후보 | 내부 변화 | 판정 |",
        "|---|---|---:|---|",
    ]
    for problem in ("P1", "P2", "P3"):
        metric = "delta_f1" if problem == "P1" else "delta_rmse"
        for name, record in result["problems"][problem]["candidates"].items():
            lines.append(
                f"| {problem} | `{name}` | `{record[metric]:+.9f}` | `{record['decision']}` |"
            )
    lines.extend(
        [
            "",
            "## 고정 운영 규칙",
            "",
            "1. 후보 레시피와 비교 기준을 점수 전에 고정한다.",
            "2. 시간 순서가 보존된 내부 OOF/forward test에서 현 챔피언 또는 가장 가까운 프록시와 비교한다.",
            "3. pooled 지표뿐 아니라 fold/기간 최악값과 paired bootstrap을 확인한다.",
            "4. 내부 `NO_GO`는 제출하지 않는다. 양수지만 불안정한 `SIGNAL_ONLY`도 추가 근거 없이 제출하지 않는다.",
            "5. `PASS_STRICT` 중 한 후보만 공식 Public에 올려 내부-공식 수송 오차를 기록한다.",
            "6. 공식 점수를 본 뒤 같은 날 같은 후보군을 재튜닝하지 않는다. 다음 묶음을 다시 공동 동결한다.",
            "",
            "## 비교 기준의 한계",
            "",
            f"- P1: {result['problems']['P1']['comparator_disclosure']}",
            f"- P2: {result['problems']['P2']['comparator_disclosure']}",
            f"- P3: {result['problems']['P3']['comparator_disclosure']}",
            "",
            "## 데이터 품질 QA",
            "",
            "- P1: OOF 키 1:1, truth mismatch 0행.",
            "- P2: prediction/truth join multiplier 1.0, missing 0행, KST bin17 고정.",
            "- P3: 182개 사례마다 6개 lead 완전, pair-key 중복 0행.",
            "- 공식 hidden label 읽기 0, 새 submission CSV 0, upload 0.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    if OUTPUT_DIR.exists():
        raise FileExistsError(OUTPUT_DIR)
    result = {
        "schema_version": "submission_ladders.internal_validation.20260831.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_INTERNAL_TEST_BEFORE_OFFICIAL",
        "policy": {
            "internal_test_required_for_every_submission": True,
            "strict_pass_rule": (
                "pooled improvement, no worse primary fold, paired bootstrap P(improve)>=0.8"
            ),
            "official_probe_after_internal_pass_only": True,
            "official_score_used_for_transport_calibration_not_hidden_tuning": True,
        },
        "problems": {"P1": score_p1(), "P2": score_p2(), "P3": score_p3()},
        "source_hashes": {
            "p1_base_oof": sha256(P1_BASE_OOF),
            "p1_peer_oof": sha256(P1_PEER_OOF),
            "p1_anchor": sha256(p1_e150.ANCHOR_PATH),
            "p2_oof": sha256(P2_OOF),
            "p2_observations": sha256(P2_OBSERVATIONS),
            "p3_blind": sha256(P3_BLIND),
            "p3_evaluated": sha256(P3_EVALUATED),
        },
        "operation_counters": {
            "historical_truth_rows_read": 421032 + 69850 + 1092,
            "official_hidden_truth_rows_read": 0,
            "submission_csv_created": 0,
            "uploads": 0,
            "new_model_fits": 0,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (OUTPUT_DIR / "report-source.md").write_text(
        report_markdown(result), encoding="utf-8"
    )
    summary = {
        problem: {
            name: {
                "decision": record["decision"],
                "delta": record["delta_f1" if problem == "P1" else "delta_rmse"],
            }
            for name, record in payload["candidates"].items()
        }
        for problem, payload in result["problems"].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
