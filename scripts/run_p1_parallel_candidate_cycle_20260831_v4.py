"""Train and forward-test two causal P1 event-level add-only challengers.

This is an exactly-once research cycle.  Historical Q2/Q3/Q4 OOF surfaces are
used first.  Official covariates are opened only when at least one challenger
passes the frozen internal gate.  Hidden labels and upload are out of scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_full_internal_submission_cycle_20260831_v2 as prior_cycle  # noqa: E402

from p1_qc.config import load_config  # noqa: E402
from p1_qc.data import load_train_test  # noqa: E402
from p1_qc.pipeline import (  # noqa: E402
    load_or_build_features,
    predict_submission,
    train_full_model,
)
from p1_qc.stratification import (  # noqa: E402
    PeerGateConfig,
    append_stratification_peer_gate,
)

EXPERIMENT_ID = "p1_parallel_candidate_cycle_20260831_v4"
RANDOM_SEED = 20260831
P1_KEYS = ["station", "year", "layer", "time"]
FOLD_ORDER = ["2025_q2", "2025_q3", "2025_q4"]
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
DELIVERY = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260831_P1_PARALLEL_CANDIDATE_CYCLE_V4"
)
P1_DATA = ROOT / "데이터셋 원본" / "데이터셋_P1" / "P1_qc_anomaly"
P1_CHAMPION = prior_cycle.P1_CHAMPION
P1_E150_DEPLOY = prior_cycle.P1_E150_DEPLOY
EXISTING_PASS_RESULT = (
    ROOT / "artifacts/full_internal_submission_cycle_20260831_v3/p1_result.json"
)


class ContractError(RuntimeError):
    """Raised when the frozen data or output contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _causal_group_features(group: pd.DataFrame) -> pd.DataFrame:
    """Create strictly current-or-past state; never inspect a later row."""
    group = group.sort_values("time", kind="stable").copy()
    anchor = group["e150_prediction"].to_numpy(np.int8)
    signal = group["signal"].to_numpy(bool)
    index = np.arange(len(group), dtype=np.int32)
    last_anchor = np.where(anchor == 1, index, -1)
    last_anchor = np.maximum.accumulate(last_anchor)
    since_anchor = index - last_anchor
    since_anchor[last_anchor < 0] = 999
    run_length = np.zeros(len(group), dtype=np.int32)
    current = 0
    for row, active in enumerate(signal):
        current = current + 1 if active else 0
        run_length[row] = current
    group["since_anchor"] = np.minimum(since_anchor, 999)
    group["signal_run_length"] = np.minimum(run_length, 48)
    for column in ("probability_base", "probability_peer", "e150_probability", "pmax"):
        rolling = group[column].rolling(window=6, min_periods=1)
        group[f"{column}_past6_mean"] = rolling.mean().to_numpy()
        group[f"{column}_past6_max"] = rolling.max().to_numpy()
        group[f"{column}_lag1"] = group[column].shift(1).fillna(0.0)
    return group


def add_causal_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return causal row state in original order and the frozen model columns."""
    required = {
        "station",
        "layer",
        "time",
        "probability_base",
        "probability_peer",
        "deployment_prediction_base",
        "deployment_prediction_peer",
        "e150_probability",
        "e150_boundary_start",
        "e150_boundary_end",
        "e150_prediction",
        "station_code",
        "layer_code",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ContractError(f"causal feature inputs missing: {missing}")
    work = frame.copy()
    work["_original_order"] = np.arange(len(work), dtype=np.int64)
    work["time"] = pd.to_datetime(work["time"], utc=True)
    probability_columns = ["probability_base", "probability_peer", "e150_probability"]
    work["pmax"] = work[probability_columns].max(axis=1)
    work["pmean"] = work[probability_columns].mean(axis=1)
    work["probability_disagreement"] = (
        work[probability_columns].max(axis=1) - work[probability_columns].min(axis=1)
    )
    work["signal"] = (
        work["pmax"].ge(0.15)
        | work["deployment_prediction_base"].eq(1)
        | work["deployment_prediction_peer"].eq(1)
    )
    pieces = [
        _causal_group_features(group)
        for _, group in work.groupby(["station", "layer"], sort=False, observed=True)
    ]
    work = pd.concat(pieces, ignore_index=True).sort_values(
        "_original_order", kind="stable"
    )
    base_columns = [
        "probability_base",
        "probability_peer",
        "deployment_prediction_base",
        "deployment_prediction_peer",
        "e150_probability",
        "e150_boundary_start",
        "e150_boundary_end",
        "e150_prediction",
        "station_code",
        "layer_code",
        "month_sin",
        "month_cos",
        "hour_sin",
        "hour_cos",
        "pmax",
        "pmean",
        "probability_disagreement",
        "since_anchor",
        "signal_run_length",
    ]
    rolling_columns = [
        f"{column}_{suffix}"
        for column in ("probability_base", "probability_peer", "e150_probability", "pmax")
        for suffix in ("past6_mean", "past6_max", "lag1")
    ]
    columns = base_columns + rolling_columns
    if work[columns].isna().any().any() or not np.isfinite(
        work[columns].to_numpy(float)
    ).all():
        raise ContractError("nonfinite causal P1 feature")
    return work.drop(columns=["_original_order"]), columns


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    probability_cutoff: float

    def mask(self, frame: pd.DataFrame) -> np.ndarray:
        anchor_zero = frame["e150_prediction"].eq(0).to_numpy()
        if self.family == "boundary":
            return anchor_zero & frame["since_anchor"].between(1, 3).to_numpy()
        return (
            anchor_zero
            & frame["signal"].to_numpy(bool)
            & frame["signal_run_length"].ge(2).to_numpy()
        )

    def build(self) -> Any:
        if self.family == "boundary":
            return RandomForestClassifier(
                n_estimators=400,
                max_depth=9,
                min_samples_leaf=20,
                max_features=0.7,
                class_weight="balanced_subsample",
                random_state=RANDOM_SEED,
                n_jobs=4,
            )
        return ExtraTreesClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=12,
            max_features=0.8,
            class_weight="balanced",
            random_state=RANDOM_SEED + 1,
            n_jobs=4,
        )


SPECS = [
    CandidateSpec("P1_4_CAUSAL_BOUNDARY_EXTENSION_RF", "boundary", 0.75),
    CandidateSpec("P1_5_CAUSAL_RUN_STATE_EXTRA_TREES", "run_state", 0.85),
]


def score_candidate(
    spec: CandidateSpec,
    frame: pd.DataFrame,
    columns: list[str],
) -> tuple[dict[str, Any], np.ndarray]:
    x = frame[columns].to_numpy(np.float64)
    truth = frame["label_base"].to_numpy(np.int8)
    anchor = frame["e150_prediction"].to_numpy(np.int8)
    eligible = spec.mask(frame)
    prediction = anchor.copy()
    by_fold: dict[str, Any] = {}
    fit_count = 0
    for test_fold in FOLD_ORDER[1:]:
        earlier = FOLD_ORDER[: FOLD_ORDER.index(test_fold)]
        train_mask = frame["fold"].isin(earlier).to_numpy() & eligible
        test_mask = frame["fold"].eq(test_fold).to_numpy()
        score_mask = test_mask & eligible
        if train_mask.sum() < 100 or np.unique(truth[train_mask]).size != 2:
            raise ContractError(f"{spec.name}: insufficient two-class training rows")
        model = spec.build()
        model.fit(x[train_mask], truth[train_mask])
        fit_count += 1
        additions = np.zeros(len(frame), dtype=bool)
        if score_mask.any():
            probability = model.predict_proba(x[score_mask])[:, 1]
            additions[np.flatnonzero(score_mask)] = probability >= spec.probability_cutoff
        prediction[test_mask] = np.maximum(
            anchor[test_mask], additions[test_mask]
        ).astype(np.int8)
        reference_f1 = float(f1_score(truth[test_mask], anchor[test_mask]))
        candidate_f1 = float(f1_score(truth[test_mask], prediction[test_mask]))
        added = test_mask & (prediction == 1) & (anchor == 0)
        by_fold[test_fold] = {
            "rows": int(test_mask.sum()),
            "eligible_rows": int(score_mask.sum()),
            "additions": int(added.sum()),
            "true_positive_additions": int((added & (truth == 1)).sum()),
            "false_positive_additions": int((added & (truth == 0)).sum()),
            "anchor_removals": 0,
            "reference_f1": reference_f1,
            "candidate_f1": candidate_f1,
            "delta_f1": candidate_f1 - reference_f1,
        }
    evaluated = frame["fold"].isin(FOLD_ORDER[1:]).to_numpy()
    reference_f1 = float(f1_score(truth[evaluated], anchor[evaluated]))
    candidate_f1 = float(f1_score(truth[evaluated], prediction[evaluated]))
    additions = evaluated & (prediction == 1) & (anchor == 0)
    fold_deltas = [record["delta_f1"] for record in by_fold.values()]
    result = {
        "name": spec.name,
        "family": spec.family,
        "past_only": True,
        "probability_cutoff_frozen": spec.probability_cutoff,
        "fit_count": fit_count,
        "forward_test_rows": int(evaluated.sum()),
        "eligible_rows": int((evaluated & eligible).sum()),
        "additions": int(additions.sum()),
        "true_positive_additions": int((additions & (truth == 1)).sum()),
        "false_positive_additions": int((additions & (truth == 0)).sum()),
        "anchor_removals": 0,
        "reference_f1": reference_f1,
        "candidate_f1": candidate_f1,
        "delta_f1": candidate_f1 - reference_f1,
        "by_fold": by_fold,
        "strict_internal_pass": bool(
            candidate_f1 > reference_f1
            and all(delta >= 0.0 for delta in fold_deltas)
            and int(additions.sum()) > 0
        ),
    }
    return result, prediction


def official_frame(columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build official covariates after internal scoring has selected a PASS."""
    config = load_config(ROOT / "configs/p1.toml", env={"P1_DATA_DIR": str(P1_DATA)})
    train, test = load_train_test(P1_DATA, audit=True, strict=True)
    base_train = load_or_build_features(train, config, kind="train", use_cache=True)
    base_test = load_or_build_features(test, config, kind="test", use_cache=True)
    selection = json.loads(prior_cycle.P1_SELECTION.read_text(encoding="utf-8"))
    base_model = train_full_model(train, base_train, config, selection)
    base_submission, base_probability = predict_submission(base_model, test, base_test)
    gate = PeerGateConfig(mode="offline", window_hours=24, min_period_fraction=0.5)
    peer_train = append_stratification_peer_gate(
        base_train,
        train,
        config=gate,
        cadence_minutes=config.data.cadence_minutes,
        group_columns=config.data.group_columns,
    )
    peer_test = append_stratification_peer_gate(
        base_test,
        test,
        config=gate,
        cadence_minutes=config.data.cadence_minutes,
        group_columns=config.data.group_columns,
    )
    peer_model = train_full_model(train, peer_train, config, selection)
    peer_submission, peer_probability = predict_submission(peer_model, test, peer_test)
    seed_arrays = [
        np.load(path)
        for path in sorted(P1_E150_DEPLOY.glob("full_width_512_seed_*_test_prediction.npz"))
    ]
    if len(seed_arrays) != 3:
        raise ContractError("expected three frozen E150 official seed arrays")
    e150_probability = np.mean([item["row_probability"] for item in seed_arrays], axis=0)
    e150_boundary = np.mean([item["boundary_probability"] for item in seed_arrays], axis=0)
    raw_test = pd.read_csv(
        P1_DATA / "test.csv", dtype={"station": "string", "time": "string"}
    )
    champion = pd.read_csv(
        P1_CHAMPION, dtype={"station": "string", "time": "string", "label": "int8"}
    )
    if len(raw_test) != 169_011 or not champion[P1_KEYS].equals(raw_test[P1_KEYS]):
        raise ContractError("official champion key/order mismatch")
    local = pd.to_datetime(raw_test["time"], utc=True).dt.tz_convert("Asia/Seoul")
    meta = pd.DataFrame(
        {
            "station": raw_test["station"],
            "layer": raw_test["layer"],
            "time": raw_test["time"],
            "probability_base": base_probability,
            "probability_peer": peer_probability,
            "deployment_prediction_base": base_submission["label"].to_numpy(np.int8),
            "deployment_prediction_peer": peer_submission["label"].to_numpy(np.int8),
            "e150_probability": e150_probability,
            "e150_boundary_start": e150_boundary[:, 0],
            "e150_boundary_end": e150_boundary[:, 1],
            "e150_prediction": champion["label"].to_numpy(np.int8),
            "station_code": raw_test["station"].map({"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}),
            "layer_code": raw_test["layer"].astype(float),
            "month_sin": np.sin(2 * np.pi * local.dt.month / 12.0),
            "month_cos": np.cos(2 * np.pi * local.dt.month / 12.0),
            "hour_sin": np.sin(2 * np.pi * local.dt.hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * local.dt.hour / 24.0),
        }
    )
    enriched, actual_columns = add_causal_features(meta)
    if actual_columns != columns:
        raise ContractError("historical/official feature schema mismatch")
    return raw_test, enriched


def materialize_passes(
    historical: pd.DataFrame,
    columns: list[str],
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    passing = [record for record in records if record["strict_internal_pass"]]
    if not passing:
        return [], 0
    raw_test, official = official_frame(columns)
    champion_label = official["e150_prediction"].to_numpy(np.int8)
    x_historical = historical[columns].to_numpy(np.float64)
    y_historical = historical["label_base"].to_numpy(np.int8)
    x_official = official[columns].to_numpy(np.float64)
    outputs = []
    fit_count = 2  # shared base and peer full-data feature-source fits
    for record in passing:
        spec = next(item for item in SPECS if item.name == record["name"])
        train_mask = spec.mask(historical)
        model = spec.build()
        model.fit(x_historical[train_mask], y_historical[train_mask])
        fit_count += 1
        score_mask = spec.mask(official)
        additions = np.zeros(len(official), dtype=bool)
        if score_mask.any():
            probability = model.predict_proba(x_official[score_mask])[:, 1]
            additions[np.flatnonzero(score_mask)] = probability >= spec.probability_cutoff
        label = np.maximum(champion_label, additions).astype(np.int8)
        submission = raw_test[P1_KEYS].copy()
        submission["label"] = label
        if (
            len(submission) != 169_011
            or submission.duplicated(P1_KEYS).any()
            or not set(submission["label"].unique()).issubset({0, 1})
        ):
            raise ContractError(f"submission contract failed: {spec.name}")
        path = DELIVERY / spec.name / "P1_submission.csv"
        path.parent.mkdir(parents=True, exist_ok=False)
        submission.to_csv(path, index=False, lineterminator="\n")
        outputs.append(
            {
                "name": spec.name,
                "path": str(path),
                "rows": int(len(submission)),
                "sha256": sha256_file(path),
                "positive_rows": int(label.sum()),
                "additions_vs_champion": int((additions & (champion_label == 0)).sum()),
                "anchor_removals": 0,
            }
        )
    return outputs, fit_count


def independent_qa(result: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "two_structurally_distinct_candidates": len(result["candidates"]) == 2,
        "all_candidates_past_only": all(
            candidate["past_only"] for candidate in result["candidates"]
        ),
        "all_internal_rows_q3_q4": all(
            candidate["forward_test_rows"] == 287_862
            for candidate in result["candidates"]
        ),
        "all_anchor_removals_zero": all(
            candidate["anchor_removals"] == 0 for candidate in result["candidates"]
        ),
        "only_passes_materialized": {
            output["name"] for output in result["outputs"]
        }
        == {
            candidate["name"]
            for candidate in result["candidates"]
            if candidate["strict_internal_pass"]
        },
        "hidden_truth_reads_zero": result["operations"]["hidden_truth_reads"] == 0,
        "uploads_zero": result["operations"]["uploads"] == 0,
    }
    for output in result["outputs"]:
        frame = pd.read_csv(
            output["path"], dtype={"station": "string", "time": "string"}
        )
        raw = pd.read_csv(
            P1_DATA / "test.csv", dtype={"station": "string", "time": "string"}
        )
        checks[f"{output['name']}_rows"] = len(frame) == 169_011
        checks[f"{output['name']}_key_order"] = frame[P1_KEYS].equals(raw[P1_KEYS])
        checks[f"{output['name']}_duplicate_keys_zero"] = not frame.duplicated(
            P1_KEYS
        ).any()
        checks[f"{output['name']}_binary"] = set(frame["label"].unique()).issubset(
            {0, 1}
        )
        checks[f"{output['name']}_sha256"] = sha256_file(Path(output["path"])) == output[
            "sha256"
        ]
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("--execute required")
    if ARTIFACT.exists() or REPORT.exists() or DELIVERY.exists():
        raise FileExistsError("one-shot output path already exists")
    ARTIFACT.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    started = time.perf_counter()
    write_json(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "pid": os.getpid(),
            "specs": [spec.__dict__ for spec in SPECS],
        },
    )
    existing = json.loads(EXISTING_PASS_RESULT.read_text(encoding="utf-8"))
    existing_passes = [
        output for output in existing["outputs"] if output["internal"]["strict_internal_pass"]
    ]
    if len(existing_passes) != 1:
        raise ContractError("expected exactly one existing v3 P1 PASS")
    historical, _ = prior_cycle.p1_frame()
    historical, columns = add_causal_features(historical)
    candidate_records = []
    for spec in SPECS:
        record, _ = score_candidate(spec, historical, columns)
        candidate_records.append(record)
    outputs, deployment_fits = materialize_passes(
        historical, columns, candidate_records
    )
    result: dict[str, Any] = {
        "schema_version": "p1.parallel_candidate_cycle.20260831.v4",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_FORWARD_TESTED_NOT_UPLOADED",
        "runtime_seconds": time.perf_counter() - started,
        "comparator": "E150_OOF_PROXY",
        "existing_pass": {
            "name": existing_passes[0]["name"],
            "path": existing_passes[0]["path"],
            "sha256": existing_passes[0]["sha256"],
            "delta_f1": existing_passes[0]["internal"]["delta_f1"],
        },
        "candidates": candidate_records,
        "outputs": outputs,
        "fit_count": int(
            sum(record["fit_count"] for record in candidate_records) + deployment_fits
        ),
        "pass_count_including_existing": len(existing_passes)
        + sum(record["strict_internal_pass"] for record in candidate_records),
        "operations": {
            "official_covariate_reads_after_internal_scoring": 1 if outputs else 0,
            "hidden_truth_reads": 0,
            "uploads": 0,
        },
    }
    result["independent_qa"] = independent_qa(result)
    write_json(ARTIFACT / "result.json", result)
    write_json(REPORT / "independent-qa.json", result["independent_qa"])
    if outputs:
        write_json(DELIVERY / "SET_MANIFEST.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
