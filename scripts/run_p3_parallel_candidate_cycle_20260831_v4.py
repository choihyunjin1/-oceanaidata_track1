"""Run the frozen P3 physical-axis candidate cycle and materialize PASS arms only.

The internal stage reads historical OOF predictions and historical anchor metadata only.
Official test inputs are opened only after at least one candidate satisfies every
pre-registered internal gate.  Hidden targets and uploads are never accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))

from p3_wave.kma_alpha_surface import prepare_oof_frame  # noqa: E402

EXPERIMENT_ID = "p3_parallel_candidate_cycle_20260831_v4"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
DELIVERY_DIR = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용\20260831_P3_PARALLEL_CANDIDATE_CYCLE_V4"
)
ATTEMPT_LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
TECHNICAL_REPAIR = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.technical_recovery.json"

P3_DATA = Path(r"C:\Users\cedis\Downloads\p3\데이터셋_P3\P3_wave_forecast")
BLIND_PATH = ROOT / "artifacts/p3_kma_calibrated_longlead_blend_v2/one_shot/blind_predictions.parquet"
EVALUATED_PATH = ROOT / "artifacts/p3/long_persistence_shrink/oof.parquet"
ANCHOR_PATH = ROOT / "artifacts/p3/features_all20_v1/train_anchors.parquet"
BASE_PATH = ROOT / "submissions/p3_frozen_catboost/submission.csv"
SOURCE_PATH = ROOT / "submissions/p3_kma_calibrated_longlead_secondary_v1/submission.csv"
CHAMPION_PATH = ROOT / "submissions/p3_20260830_uniform_kma_0425_v1/P3_submission.csv"

KEYS = ["case_id", "station", "lead_h"]
PAIR_KEYS = ["fold", "anchor_id", "station", "lead_h"]
ALL_LEADS = (3, 6, 9, 12, 18, 24)
ACTIVE_LEADS = (18, 24)
REFERENCE_ALPHA = 0.425
ALPHA_MIN = 0.0
ALPHA_MAX = 0.65
PURGE_HOURS = 78.0
BOOTSTRAP_REPLICATES = 5_000
BOOTSTRAP_SEED = 20260831


class ContractError(RuntimeError):
    """Raised when an immutable experiment contract is violated."""


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    trust: float
    regularization: float
    summary: str


SPECS = (
    CandidateSpec(
        name="P3_1_PHYSICAL_STATE_GBDT_GATE",
        family="hist_gbdt",
        trust=0.80,
        regularization=20.0,
        summary="Train-only nonlinear sea-state gate constrained to the KMA physical correction axis.",
    ),
    CandidateSpec(
        name="P3_2_PHYSICAL_STATE_HUBER_AXIS",
        family="huber",
        trust=0.70,
        regularization=0.20,
        summary="Robust Huber state/lead/station shrinkage constrained to the KMA physical axis.",
    ),
    CandidateSpec(
        name="P3_3_PHYSICAL_STATE_RIDGE_AXIS",
        family="ridge",
        trust=0.60,
        regularization=20.0,
        summary="Strongly regularized hierarchical state/lead/station KMA-axis shrinkage.",
    ),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(y, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if truth.shape != pred.shape or truth.size == 0:
        raise ContractError("RMSE inputs are empty or differ in shape")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ContractError("RMSE inputs contain non-finite values")
    return float(np.sqrt(np.mean(np.square(pred - truth))))


def _station_codes(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    station = frame["station"].astype(str)
    known = station.isin(["G-ORS", "I-ORS", "S-ORS"])
    if not known.all():
        raise ContractError("unknown station in P3 frame")
    return (
        station.eq("G-ORS").to_numpy(dtype=np.float64),
        station.eq("I-ORS").to_numpy(dtype=np.float64),
        station.eq("S-ORS").to_numpy(dtype=np.float64),
    )


def state_basis(frame: pd.DataFrame) -> np.ndarray:
    """Build an official-deployable basis without calendar or target information."""

    g, i, s = _station_codes(frame)
    lead24 = frame["lead_h"].eq(24).to_numpy(dtype=np.float64)
    current = frame["current_hs"].to_numpy(dtype=np.float64)
    base = frame["base"].to_numpy(dtype=np.float64)
    delta = frame["delta"].to_numpy(dtype=np.float64)
    source = base + delta
    current_c = np.clip((current - 2.0) / 1.5, -2.0, 2.0)
    tendency = np.clip((base - current) / 1.0, -2.0, 2.0)
    source_tendency = np.clip((source - current) / 1.0, -2.0, 2.0)
    abs_delta = np.clip(np.abs(delta) / 0.5, 0.0, 3.0)
    calm = (current < 1.75).astype(np.float64)
    energetic = (current >= 2.75).astype(np.float64)
    return np.column_stack(
        [
            np.ones(len(frame), dtype=np.float64),
            lead24,
            g,
            i,
            s,
            current_c,
            np.square(current_c),
            tendency,
            source_tendency,
            abs_delta,
            calm,
            energetic,
            lead24 * current_c,
            lead24 * tendency,
            g * lead24,
            i * lead24,
            s * lead24,
            calm * lead24,
            energetic * lead24,
        ]
    )


class PhysicalAxisModel:
    """Estimate a bounded KMA mixture weight using one frozen model family."""

    def __init__(self, spec: CandidateSpec, seed: int) -> None:
        self.spec = spec
        self.seed = int(seed)
        self.estimator: Any | None = None

    def fit(self, frame: pd.DataFrame) -> PhysicalAxisModel:
        active = frame["lead_h"].isin(ACTIVE_LEADS).to_numpy()
        train = frame.loc[active].reset_index(drop=True)
        if len(train) < 100:
            raise ContractError("insufficient active rows for physical-axis fit")
        basis = state_basis(train)
        delta = train["delta"].to_numpy(dtype=np.float64)
        target_correction = (
            train["target_hs"].to_numpy(dtype=np.float64)
            - train["base"].to_numpy(dtype=np.float64)
        )
        if self.spec.family == "hist_gbdt":
            safe = np.abs(delta) >= 0.02
            alpha_target = np.zeros(len(train), dtype=np.float64)
            alpha_target[safe] = target_correction[safe] / delta[safe]
            alpha_target = np.clip(alpha_target, -0.50, 1.25)
            sample_weight = np.clip(np.square(delta) / 0.04, 0.05, 4.0)
            estimator = HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.04,
                max_iter=180,
                max_leaf_nodes=7,
                min_samples_leaf=15,
                l2_regularization=self.spec.regularization,
                random_state=self.seed,
            )
            estimator.fit(basis, alpha_target, sample_weight=sample_weight)
        else:
            axis_design = delta[:, None] * basis
            if self.spec.family == "huber":
                estimator = make_pipeline(
                    StandardScaler(with_mean=False),
                    HuberRegressor(
                        epsilon=1.50,
                        alpha=self.spec.regularization,
                        fit_intercept=False,
                        max_iter=1_000,
                        tol=1e-7,
                    ),
                )
            elif self.spec.family == "ridge":
                estimator = make_pipeline(
                    StandardScaler(with_mean=False),
                    Ridge(alpha=self.spec.regularization, fit_intercept=False),
                )
            else:
                raise ContractError(f"unknown family: {self.spec.family}")
            estimator.fit(axis_design, target_correction)
        self.estimator = estimator
        return self

    def alpha(self, frame: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            raise ContractError("physical-axis model is not fitted")
        basis = state_basis(frame)
        delta = frame["delta"].to_numpy(dtype=np.float64)
        if self.spec.family == "hist_gbdt":
            raw = self.estimator.predict(basis)
        else:
            correction = self.estimator.predict(delta[:, None] * basis)
            raw = np.full(len(frame), REFERENCE_ALPHA, dtype=np.float64)
            stable = np.abs(delta) >= 0.02
            raw[stable] = correction[stable] / delta[stable]
        bounded = np.clip(raw, ALPHA_MIN, ALPHA_MAX)
        effective = REFERENCE_ALPHA + self.spec.trust * (bounded - REFERENCE_ALPHA)
        return np.clip(effective, ALPHA_MIN, ALPHA_MAX)

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        alpha = self.alpha(frame)
        prediction = frame["reference"].to_numpy(dtype=np.float64).copy()
        active = frame["lead_h"].isin(ACTIVE_LEADS).to_numpy()
        prediction[active] = (
            frame.loc[active, "base"].to_numpy(dtype=np.float64)
            + alpha[active] * frame.loc[active, "delta"].to_numpy(dtype=np.float64)
        )
        prediction = np.clip(prediction, 0.0, 30.0)
        if not np.array_equal(
            prediction[~active], frame.loc[~active, "reference"].to_numpy(dtype=np.float64)
        ):
            raise ContractError("short-lead prediction changed")
        if not np.isfinite(prediction).all():
            raise ContractError("prediction contains non-finite values")
        return prediction, alpha


def attach_historical_metadata(frame: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    required = {"anchor_id", "station", "anchor_time"}
    if not required.issubset(anchors.columns):
        raise ContractError("historical anchor metadata schema changed")
    lookup = anchors.loc[:, ["anchor_id", "station", "anchor_time"]].drop_duplicates()
    if lookup.duplicated(["anchor_id", "station"]).any():
        raise ContractError("historical anchor metadata key is duplicated")
    output = frame.merge(
        lookup,
        on=["anchor_id", "station"],
        how="left",
        validate="many_to_one",
    )
    output["anchor_time"] = pd.to_datetime(output["anchor_time"], utc=True, errors="raise")
    if output["anchor_time"].isna().any():
        raise ContractError("historical anchor-time join failed")
    output["block"] = output["anchor_time"].dt.month.map(
        {1: "01_02", 2: "01_02", 3: "03_04", 4: "03_04", 5: "05_06", 6: "05_06",
         7: "07_08", 8: "07_08", 9: "09_10", 10: "09_10", 11: "11_12", 12: "11_12"}
    )
    case = (
        output.loc[:, ["anchor_id", "station", "anchor_time"]]
        .drop_duplicates()
        .sort_values(["station", "anchor_time", "anchor_id"], kind="stable")
    )
    gap = case.groupby("station", observed=True)["anchor_time"].diff().dt.total_seconds().div(3600)
    case["episode_local"] = gap.gt(PURGE_HOURS).fillna(True).groupby(case["station"]).cumsum()
    case["episode_id"] = case["station"].astype(str) + "|" + case["episode_local"].astype(int).astype(str)
    output = output.merge(
        case[["anchor_id", "station", "episode_id"]],
        on=["anchor_id", "station"],
        how="left",
        validate="many_to_one",
    )
    return output.sort_values(PAIR_KEYS, kind="stable").reset_index(drop=True)


def purge_training_cases(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    case_train = train[["anchor_id", "station", "anchor_time"]].drop_duplicates()
    case_test = test[["station", "anchor_time"]].drop_duplicates()
    keep = np.ones(len(case_train), dtype=bool)
    for station, test_group in case_test.groupby("station", observed=True):
        positions = case_train["station"].eq(station).to_numpy()
        if not positions.any():
            continue
        train_ns = pd.DatetimeIndex(case_train.loc[positions, "anchor_time"]).as_unit("ns").asi8
        test_ns = pd.DatetimeIndex(test_group["anchor_time"]).as_unit("ns").asi8
        distance_h = np.min(np.abs(train_ns[:, None] - test_ns[None, :]), axis=1) / 3.6e12
        keep[np.flatnonzero(positions)] = distance_h > PURGE_HOURS
    keys = case_train.loc[keep, ["anchor_id", "station"]]
    purged = train.merge(keys, on=["anchor_id", "station"], how="inner", validate="many_to_one")
    if purged.empty:
        raise ContractError("78h purge removed the entire training surface")
    return purged


def episode_diagnostics(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    work = frame.copy()
    work["candidate"] = prediction
    deltas: list[float] = []
    records: dict[str, Any] = {}
    for episode, group in work.groupby("episode_id", observed=True, sort=True):
        before = rmse(group["target_hs"].to_numpy(float), group["reference"].to_numpy(float))
        after = rmse(group["target_hs"].to_numpy(float), group["candidate"].to_numpy(float))
        delta = after - before
        deltas.append(delta)
        records[str(episode)] = {"rows": int(len(group)), "delta_rmse": delta}
    array = np.asarray(deltas, dtype=np.float64)
    return {
        "episode_count": int(len(array)),
        "improved_episode_count": int(np.sum(array < 0.0)),
        "improved_episode_share": float(np.mean(array < 0.0)),
        "median_episode_delta_rmse": float(np.median(array)),
        "records": records,
    }


def episode_bootstrap(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    work = frame.copy()
    work["candidate"] = prediction
    grouped = list(work.groupby("episode_id", observed=True, sort=True))
    ref_sse = np.asarray(
        [np.square(group["reference"] - group["target_hs"]).sum() for _, group in grouped]
    )
    cand_sse = np.asarray(
        [np.square(group["candidate"] - group["target_hs"]).sum() for _, group in grouped]
    )
    counts = np.asarray([len(group) for _, group in grouped], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, len(grouped), size=len(grouped))
        denominator = float(counts[draw].sum())
        deltas[index] = np.sqrt(cand_sse[draw].sum() / denominator) - np.sqrt(
            ref_sse[draw].sum() / denominator
        )
    return {
        "unit": "station_contiguous_episode_gap_gt_78h",
        "episode_count": int(len(grouped)),
        "replicates": BOOTSTRAP_REPLICATES,
        "mean_delta_rmse": float(np.mean(deltas)),
        "ci90_low": float(np.quantile(deltas, 0.05)),
        "ci90_high": float(np.quantile(deltas, 0.95)),
        "probability_improved": float(np.mean(deltas < 0.0)),
    }


def evaluate_candidate(frame: pd.DataFrame, spec: CandidateSpec, spec_index: int) -> tuple[dict[str, Any], PhysicalAxisModel]:
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    alpha = np.full(len(frame), np.nan, dtype=np.float64)
    by_block: dict[str, Any] = {}
    blocks = sorted(frame["block"].astype(str).unique())
    for block_index, block in enumerate(blocks):
        test_mask = frame["block"].astype(str).eq(block).to_numpy()
        test = frame.loc[test_mask].copy()
        train = purge_training_cases(frame.loc[~test_mask].copy(), test)
        model = PhysicalAxisModel(spec, BOOTSTRAP_SEED + 100 * spec_index + block_index).fit(train)
        fold_prediction, fold_alpha = model.predict(test)
        prediction[test_mask] = fold_prediction
        alpha[test_mask] = fold_alpha
        before = rmse(test["target_hs"].to_numpy(float), test["reference"].to_numpy(float))
        after = rmse(test["target_hs"].to_numpy(float), fold_prediction)
        by_block[block] = {
            "rows": int(len(test)),
            "cases": int(test.groupby(["anchor_id", "station"], observed=True).ngroups),
            "train_rows_after_78h_purge": int(len(train)),
            "reference_rmse": before,
            "candidate_rmse": after,
            "delta_rmse": after - before,
        }
    if not np.isfinite(prediction).all() or not np.isfinite(alpha).all():
        raise ContractError("cross-fit prediction is incomplete")
    y = frame["target_hs"].to_numpy(dtype=np.float64)
    reference = frame["reference"].to_numpy(dtype=np.float64)
    reference_rmse = rmse(y, reference)
    candidate_rmse = rmse(y, prediction)
    episodes = episode_diagnostics(frame, prediction)
    bootstrap = episode_bootstrap(frame, prediction)
    improved_blocks = int(sum(record["delta_rmse"] < 0.0 for record in by_block.values()))
    worst_block_delta = float(max(record["delta_rmse"] for record in by_block.values()))
    gates = {
        "pooled_rmse_improves": candidate_rmse < reference_rmse,
        "strict_majority_episodes_improve": episodes["improved_episode_share"] > 0.50,
        "at_least_four_of_six_blocks_improve": improved_blocks >= 4,
        "bootstrap_probability_at_least_0p80": bootstrap["probability_improved"] >= 0.80,
        "bootstrap_ci90_upper_below_zero": bootstrap["ci90_high"] < 0.0,
        "catastrophic_block_guard_max_degradation_le_0p01m": worst_block_delta <= 0.01,
    }
    record = {
        "spec": asdict(spec),
        "fit_count": len(blocks),
        "reference_rmse": reference_rmse,
        "candidate_rmse": candidate_rmse,
        "delta_rmse": candidate_rmse - reference_rmse,
        "by_block": by_block,
        "improved_block_count": improved_blocks,
        "worst_block_delta_rmse": worst_block_delta,
        "episodes": episodes,
        "bootstrap": bootstrap,
        "alpha_distribution": {
            "min": float(alpha.min()),
            "q10": float(np.quantile(alpha, 0.10)),
            "median": float(np.median(alpha)),
            "q90": float(np.quantile(alpha, 0.90)),
            "max": float(alpha.max()),
        },
        "gates": gates,
        "strict_internal_pass": bool(all(gates.values())),
    }
    full = PhysicalAxisModel(spec, BOOTSTRAP_SEED + 1_000 + spec_index).fit(frame)
    return record, full


def load_historical() -> tuple[pd.DataFrame, dict[str, Any]]:
    blind = pd.read_parquet(BLIND_PATH)
    evaluated = pd.read_parquet(EVALUATED_PATH)
    anchors = pd.read_parquet(ANCHOR_PATH)
    frame = prepare_oof_frame(blind, evaluated)
    current = blind[PAIR_KEYS + ["current_hs"]].drop_duplicates()
    frame = frame.merge(current, on=PAIR_KEYS, how="left", validate="one_to_one")
    if frame["current_hs"].isna().any():
        raise ContractError("historical current_hs join failed")
    frame = attach_historical_metadata(frame, anchors)
    lead = frame["lead_h"].to_numpy(dtype=np.int64)
    reference_alpha = np.where(np.isin(lead, ACTIVE_LEADS), REFERENCE_ALPHA, 0.0)
    frame["reference"] = np.clip(
        frame["base"].to_numpy(float) + reference_alpha * frame["delta"].to_numpy(float),
        0.0,
        30.0,
    )
    case_leads = frame.groupby(["fold", "anchor_id", "station"], observed=True)["lead_h"].agg(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    if not case_leads.map(lambda values: values == ALL_LEADS).all():
        raise ContractError("historical case does not contain all six leads")
    if frame.duplicated(PAIR_KEYS).any() or not np.isfinite(
        frame[["target_hs", "base", "delta", "current_hs", "reference"]].to_numpy(float)
    ).all():
        raise ContractError("historical grain, uniqueness, or numeric validity failed")
    profile = {
        "rows": int(len(frame)),
        "cases": int(len(case_leads)),
        "episodes": int(frame["episode_id"].nunique()),
        "blocks": frame.groupby("block", observed=True)[["anchor_id", "station"]]
        .apply(lambda group: len(group.drop_duplicates()))
        .astype(int)
        .to_dict(),
        "duplicate_pair_keys": int(frame.duplicated(PAIR_KEYS).sum()),
        "nonfinite_values": 0,
        "target_rows_read": int(len(frame)),
        "official_rows_read_before_internal_gate": 0,
    }
    return frame, profile


def official_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    test_index = pd.read_csv(P3_DATA / "test_index.csv", dtype={"case_id": "string", "station": "string"})
    context = pd.read_parquet(P3_DATA / "test_context.parquet")
    base = pd.read_csv(BASE_PATH, dtype={"case_id": "string", "station": "string"})
    source = pd.read_csv(SOURCE_PATH, dtype={"case_id": "string", "station": "string"})
    champion = pd.read_csv(CHAMPION_PATH, dtype={"case_id": "string", "station": "string"})
    for role, candidate in (("base", base), ("source", source), ("champion", champion)):
        if list(candidate.columns) != KEYS + ["hs_pred"] or not candidate[KEYS].equals(test_index[KEYS]):
            raise ContractError(f"official {role} schema or key order changed")
    if len(test_index) != 1_200 or test_index.duplicated(KEYS).any():
        raise ContractError("official test-index grain changed")
    current = context.loc[context["step_minute"].eq(0), ["case_id", "station", "hs"]]
    if len(current) != 200 or current.duplicated(["case_id", "station"]).any():
        raise ContractError("official context current-state grain changed")
    frame = test_index.merge(current, on=["case_id", "station"], how="left", validate="many_to_one")
    frame["current_hs"] = frame["hs"].to_numpy(float)
    frame["base"] = base["hs_pred"].to_numpy(float)
    frame["delta"] = source["hs_pred"].to_numpy(float) - frame["base"].to_numpy(float)
    frame["reference"] = champion["hs_pred"].to_numpy(float)
    if not np.isfinite(frame[["current_hs", "base", "delta", "reference"]].to_numpy(float)).all():
        raise ContractError("official deployable feature frame contains non-finite values")
    return frame, champion


def materialize_passes(
    passing: list[tuple[dict[str, Any], PhysicalAxisModel]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not passing:
        return [], {
            "test_index_rows_read": 0,
            "test_context_rows_read": 0,
            "official_prediction_rows_read": 0,
            "hidden_truth_rows_read": 0,
            "uploads": 0,
        }
    frame, champion = official_frame()
    DELIVERY_DIR.mkdir(parents=True, exist_ok=False)
    outputs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record, model in passing[:3]:
        prediction, alpha = model.predict(frame)
        submission = frame[KEYS].copy()
        submission["hs_pred"] = prediction
        if list(submission.columns) != KEYS + ["hs_pred"] or len(submission) != 1_200:
            raise ContractError("submission schema or row count failed")
        if submission.duplicated(KEYS).any() or not np.isfinite(prediction).all():
            raise ContractError("submission key uniqueness or finite check failed")
        inactive = ~submission["lead_h"].isin(ACTIVE_LEADS).to_numpy()
        if not np.array_equal(
            prediction[inactive], champion.loc[inactive, "hs_pred"].to_numpy(dtype=np.float64)
        ):
            raise ContractError("submission short-lead exact no-op failed")
        directory = DELIVERY_DIR / record["spec"]["name"]
        csv_path = directory / "P3_submission.csv"
        payload = submission.to_csv(index=False, lineterminator="\n").encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()
        if digest in seen:
            raise ContractError("duplicate passing submission candidate")
        seen.add(digest)
        write_new(csv_path, payload)
        info = {
            "name": record["spec"]["name"],
            "path": str(csv_path),
            "rows": 1_200,
            "sha256": digest,
            "bytes": len(payload),
            "minimum_m": float(prediction.min()),
            "maximum_m": float(prediction.max()),
            "short_lead_exact_noop": True,
            "changed_rows_vs_champion": int(
                np.sum(np.abs(prediction - champion["hs_pred"].to_numpy(float)) > 1e-12)
            ),
            "rms_change_vs_champion_m": float(
                np.sqrt(np.mean(np.square(prediction - champion["hs_pred"].to_numpy(float))))
            ),
            "official_alpha_distribution": {
                "min": float(alpha.min()),
                "median": float(np.median(alpha)),
                "max": float(alpha.max()),
            },
            "internal": record,
        }
        write_new(
            directory / "submission-info.txt",
            (
                f"title: {record['spec']['name']}\n"
                f"summary: {record['spec']['summary']}\n"
                f"sha256: {digest}\n"
                "status: INTERNAL_PASS_MATERIALIZED_NOT_UPLOADED\n"
            ).encode(),
        )
        outputs.append(info)
    manifest = {
        "schema_version": "p3.parallel_candidate_cycle.delivery.20260831.v4",
        "experiment_id": EXPERIMENT_ID,
        "status": "INTERNAL_PASS_MATERIALIZED_NOT_UPLOADED",
        "outputs": outputs,
        "hidden_truth_rows_read": 0,
        "uploads": 0,
    }
    write_new(DELIVERY_DIR / "SET_MANIFEST.json", json_bytes(manifest))
    access = {
        "test_index_rows_read": 1_200,
        "test_context_rows_read": 57_800,
        "official_prediction_rows_read": 3_600,
        "hidden_truth_rows_read": 0,
        "uploads": 0,
    }
    return outputs, access


def report_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# P3 parallel physical-axis candidate cycle v4",
        "",
        "## 결론",
        "",
        f"- 엄격 내부 PASS: **{result['passing_candidate_count']}/{len(result['candidates'])}**",
        f"- 제출 CSV 생성: **{len(result['outputs'])}개** (업로드 0)",
        "- direct residual 및 단순 alpha sweep를 반복하지 않고, KMA 물리축 내부의 train-only 상태 게이트만 평가했다.",
        "",
        "## 후보 결과",
        "",
        "| candidate | delta RMSE(m) | improved blocks | improved episodes | P(improve) | CI90 high | worst block | PASS |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in result["candidates"]:
        lines.append(
            "| {name} | {delta:+.6f} | {blocks}/6 | {episodes:.3f} | {prob:.3f} | {ci:+.6f} | {worst:+.6f} | {passed} |".format(
                name=record["spec"]["name"],
                delta=record["delta_rmse"],
                blocks=record["improved_block_count"],
                episodes=record["episodes"]["improved_episode_share"],
                prob=record["bootstrap"]["probability_improved"],
                ci=record["bootstrap"]["ci90_high"],
                worst=record["worst_block_delta_rmse"],
                passed="PASS" if record["strict_internal_pass"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            "## 검증 계약",
            "",
            "- 6개 bimonth holdout, 동일 station ±78시간 purge",
            "- 독립 단위: station별 78시간 초과 gap으로 분리한 historical episode",
            "- PASS: pooled 개선, episode 과반 개선, 4/6 block 개선, bootstrap P>=0.8, CI90 upper<0, worst block degradation<=0.01m",
            "- short leads 3/6/9/12h exact no-op, long leads는 alpha [0,0.65] 물리축 제약",
            "- hidden truth 0행, upload 0회",
            "",
            "## 데이터 품질",
            "",
            f"- historical rows/cases/episodes: {result['data_profile']['rows']}/{result['data_profile']['cases']}/{result['data_profile']['episodes']}",
            "- duplicate pair keys 0, non-finite 0, every case six leads intact",
            "- official inputs were not opened until after all internal gates were finalized.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-after-time-unit-repair", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("--execute required")
    if ARTIFACT_DIR.exists() or REPORT_DIR.exists() or DELIVERY_DIR.exists():
        raise FileExistsError("v4 output already exists")
    for path in (BLIND_PATH, EVALUATED_PATH, ANCHOR_PATH, BASE_PATH, SOURCE_PATH, CHAMPION_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    started = time.perf_counter()
    runner_hash = sha256(Path(__file__))
    if ATTEMPT_LOCK.exists():
        if not args.resume_after_time_unit_repair or TECHNICAL_REPAIR.exists():
            raise FileExistsError("v4 attempt lock already exists")
        original_lock = json.loads(ATTEMPT_LOCK.read_text(encoding="utf-8"))
        if original_lock.get("experiment_id") != EXPERIMENT_ID:
            raise ContractError("existing attempt lock belongs to another experiment")
        repair = {
            "schema_version": "p3.parallel_candidate_cycle.technical_repair.20260831.v4",
            "experiment_id": EXPERIMENT_ID,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": "RESUME_AUTHORIZED_AFTER_ZERO_FIT_TIME_UNIT_REPAIR",
            "failure": "Pandas timezone datetime integer unit was us but purge divisor assumed ns.",
            "scope": "Explicitly convert train/test DatetimeIndex to ns before 78h distance calculation.",
            "completed_fit_count_before_failure": 0,
            "completed_prediction_count_before_failure": 0,
            "official_rows_read_before_failure": 0,
            "candidate_or_gate_changes": 0,
            "original_runner_sha256": original_lock.get("runner_sha256"),
            "repaired_runner_sha256": runner_hash,
        }
        write_new(TECHNICAL_REPAIR, json_bytes(repair))
    else:
        if args.resume_after_time_unit_repair:
            raise ContractError("technical-repair resume requested without an attempt lock")
        lock = {
            "schema_version": "p3.parallel_candidate_cycle.attempt.20260831.v4",
            "experiment_id": EXPERIMENT_ID,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": "ATTEMPT_CONSUMED",
            "runner_sha256": runner_hash,
            "candidate_specs": [asdict(spec) for spec in SPECS],
            "gate_contract": {
                "pooled_delta_rmse_lt_0": True,
                "strict_majority_episode_improvement": True,
                "minimum_improved_blocks_of_6": 4,
                "bootstrap_probability_improved_min": 0.80,
                "bootstrap_ci90_upper_lt_0": True,
                "worst_block_delta_rmse_max_m": 0.01,
            },
            "official_inputs_before_internal_gate": 0,
        }
        write_new(ATTEMPT_LOCK, json_bytes(lock))
    frame, profile = load_historical()
    records: list[dict[str, Any]] = []
    passing: list[tuple[dict[str, Any], PhysicalAxisModel]] = []
    for spec_index, spec in enumerate(SPECS):
        record, model = evaluate_candidate(frame, spec, spec_index)
        records.append(record)
        if record["strict_internal_pass"]:
            passing.append((record, model))
    outputs, access = materialize_passes(passing)
    result = {
        "schema_version": "p3.parallel_candidate_cycle.result.20260831.v4",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "COMPLETE_INTERNAL_PASS_READY_NOT_UPLOADED" if outputs else "COMPLETE_NO_INTERNAL_PASS",
        "runtime_seconds": float(time.perf_counter() - started),
        "runner_sha256": runner_hash,
        "attempt_lock_sha256": sha256(ATTEMPT_LOCK),
        "technical_repair": (
            json.loads(TECHNICAL_REPAIR.read_text(encoding="utf-8"))
            if TECHNICAL_REPAIR.exists()
            else None
        ),
        "source_hashes": {
            "blind_oof": sha256(BLIND_PATH),
            "evaluated_oof": sha256(EVALUATED_PATH),
            "historical_anchors": sha256(ANCHOR_PATH),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "data_profile": profile,
        "internal_comparator": "OFFICIAL_CHAMPION_LINEAGE_UNIFORM_KMA_ALPHA_0P425_OOF_PROXY",
        "candidate_count": len(records),
        "passing_candidate_count": len(passing),
        "fit_count": len(SPECS) * 7,
        "candidates": records,
        "outputs": outputs,
        "official_access": access,
        "hidden_truth_rows_read": 0,
        "uploads": 0,
        "result_based_retry_or_tuning": False,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    write_new(ARTIFACT_DIR / "result.json", json_bytes(result))
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    write_new(REPORT_DIR / "report-source.md", report_markdown(result).encode("utf-8"))
    write_new(
        REPORT_DIR / "run-manifest.json",
        json_bytes(
            {
                "experiment_id": EXPERIMENT_ID,
                "runner_sha256": runner_hash,
                "result_sha256": sha256(ARTIFACT_DIR / "result.json"),
                "attempt_lock_sha256": sha256(ATTEMPT_LOCK),
                "fit_count": result["fit_count"],
                "passing_candidate_count": result["passing_candidate_count"],
                "hidden_truth_rows_read": 0,
                "uploads": 0,
            }
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
