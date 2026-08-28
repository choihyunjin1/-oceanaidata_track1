"""Run a label-free official-test shadow audit for the P1 MSTCN veto."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_mstcn_deployable_type_veto_stability_20260829_v1 as prior  # noqa: E402

EXPERIMENT_ID = "p1_mstcn_official_shadow_lower_bound_veto_20260829_v1"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
MODEL_CONFIG_PATH = (
    ROOT
    / "configs"
    / "experiments"
    / "p1_mstcn_deployable_type_veto_stability_20260829_v1.json"
)
HISTORICAL_RESULT_PATH = (
    ROOT / "artifacts" / "p1_mstcn_bootstrap_lower_bound_veto_20260829_v1" / "result.json"
)
DEPLOYMENT_DIR = ROOT / "artifacts" / "p1_mstcn_e150_full_deployment_20260827_v1"
DEPLOYMENT_PREFLIGHT_PATH = DEPLOYMENT_DIR / "preflight.json"
DEPLOYMENT_QA_PATH = DEPLOYMENT_DIR / "independent_qa.json"
E150_PATH = DEPLOYMENT_DIR / "P1_MSTCN_E150_ROUTER_UNION_ALL.csv"
CHAMPION_PATH = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260828_DEADLINE_INFORMATION_PROBES_READY"
    r"\P1_1_E150_PLUS_GI_SPIKE2\P1_submission.csv"
)
PREDICTION_PATHS = [
    DEPLOYMENT_DIR / f"full_width_512_seed_{seed}_epoch_150_test_prediction.npz"
    for seed in (20260827, 20260839, 20260863)
]
OUTPUT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
KEYS = ["station", "year", "layer", "time"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_historical_training(config: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    bundles = prior.base.load_bundles()
    frames: list[pd.DataFrame] = []
    utilities: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    for fold in config["historical_training_folds"]:
        bundle = bundles[str(fold)]
        frame = bundle.segments.copy()
        frames.append(frame)
        utilities.append(prior.marginal_utility(bundle))
        fold_groups = prior.truth_event_groups(bundle)
        groups.append(np.asarray([f"{fold}:{value}" for value in fold_groups], dtype=object))
    training = pd.concat(frames, ignore_index=True)
    utility = np.concatenate(utilities)
    event_groups = np.concatenate(groups)
    require(len(training) == len(utility) == len(event_groups), "historical length mismatch")
    require(training[prior.FEATURES].notna().all().all(), "historical feature missing")
    return training, utility, event_groups


def _load_prediction_ensemble(rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_sum = np.zeros(rows, dtype=np.float64)
    boundary_sum = np.zeros((rows, 2), dtype=np.float64)
    type_sum = np.zeros((rows, 5), dtype=np.float64)
    for path in PREDICTION_PATHS:
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == {"row_probability", "boundary_probability", "type_probability"}, f"prediction keys: {path.name}")
            row = archive["row_probability"].astype(np.float64)
            boundary = archive["boundary_probability"].astype(np.float64)
            types = archive["type_probability"].astype(np.float64)
        require(row.shape == (rows,), f"row shape: {path.name}")
        require(boundary.shape == (rows, 2), f"boundary shape: {path.name}")
        require(types.shape == (rows, 5), f"type shape: {path.name}")
        require(np.isfinite(row).all() and np.isfinite(boundary).all() and np.isfinite(types).all(), f"nonfinite prediction: {path.name}")
        row_sum += row
        boundary_sum += boundary
        type_sum += types
    divisor = float(len(PREDICTION_PATHS))
    return row_sum / divisor, boundary_sum / divisor, type_sum / divisor


def _build_shadow_segments(
    frame: pd.DataFrame,
    anchor: np.ndarray,
    e150: np.ndarray,
    row_probability: np.ndarray,
    boundary_probability: np.ndarray,
    type_probability: np.ndarray,
) -> tuple[pd.DataFrame, list[np.ndarray]]:
    addition = (e150 == 1) & (anchor == 0)
    parsed_time = pd.to_datetime(frame["time"], utc=True)
    continuation = (
        pd.Series(addition).shift(fill_value=False).to_numpy()
        & frame["station"].astype(str).eq(frame["station"].astype(str).shift()).to_numpy()
        & frame["layer"].eq(frame["layer"].shift()).to_numpy()
        & parsed_time.diff().eq(pd.Timedelta(minutes=10)).to_numpy()
    )
    segment_ids = np.cumsum(addition & ~continuation)
    entropy = prior.base._entropy(type_probability)
    records: list[dict[str, float | int | str]] = []
    indices: list[np.ndarray] = []
    for segment_id in np.unique(segment_ids[addition]):
        positions = np.flatnonzero(addition & (segment_ids == segment_id))
        row = row_probability[positions]
        boundary = boundary_probability[positions]
        types = type_probability[positions]
        record: dict[str, float | int | str] = {
            "station": str(frame.loc[int(positions[0]), "station"]),
            "layer": str(frame.loc[int(positions[0]), "layer"]),
            "length": int(len(positions)),
            "log_length": float(math.log1p(len(positions))),
            "row_mean": float(np.mean(row)),
            "row_min": float(np.min(row)),
            "row_max": float(np.max(row)),
            "row_q10": float(np.quantile(row, 0.10)),
            "row_median": float(np.median(row)),
            "row_q90": float(np.quantile(row, 0.90)),
            "row_std": float(np.std(row)),
            "boundary_start_mean": float(np.mean(boundary[:, 0])),
            "boundary_start_max": float(np.max(boundary[:, 0])),
            "boundary_end_mean": float(np.mean(boundary[:, 1])),
            "boundary_end_max": float(np.max(boundary[:, 1])),
        }
        for index, name in enumerate(prior.base.TYPE_NAMES):
            record[f"type_{name}_mean"] = float(np.mean(types[:, index]))
        record["type_entropy_mean"] = float(np.mean(entropy[positions]))
        records.append(record)
        indices.append(positions)
    segments = pd.DataFrame(records)
    require(segments[prior.FEATURES].notna().all().all(), "shadow feature missing")
    require(sum(len(value) for value in indices) == int(addition.sum()), "segment accounting")
    return segments, indices


def _bootstrap_frequency(
    training: pd.DataFrame,
    utility: np.ndarray,
    groups: np.ndarray,
    evaluation: pd.DataFrame,
    config: dict,
    model_config: dict,
) -> tuple[np.ndarray, int]:
    targets = training["beneficial"].to_numpy(np.int8)
    available_groups = np.unique(groups)
    replicates = int(config["bootstrap_replicates"])
    rng = np.random.default_rng(int(config["seed"]))
    accepted = np.zeros((replicates, len(evaluation)), dtype=bool)
    completed = 0
    attempts = 0
    while completed < replicates:
        attempts += 1
        require(attempts <= replicates * 30, "bootstrap class balance failure")
        sampled_groups = rng.choice(available_groups, size=len(available_groups), replace=True)
        sampled = np.concatenate([np.flatnonzero(groups == group) for group in sampled_groups])
        if np.unique(targets[sampled]).size < 2:
            continue
        classifier = prior.classifier(model_config)
        regressor = prior.regressor(model_config)
        classifier.fit(training.iloc[sampled][prior.FEATURES], targets[sampled])
        regressor.fit(training.iloc[sampled][prior.FEATURES], utility[sampled])
        accepted[completed] = (
            classifier.predict_proba(evaluation[prior.FEATURES])[:, 1] >= 0.5
        ) & (regressor.predict(evaluation[prior.FEATURES]) > 0.0)
        completed += 1
    return np.mean(accepted, axis=0), attempts


def execute() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model_config = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    historical_result = json.loads(HISTORICAL_RESULT_PATH.read_text(encoding="utf-8"))
    preflight = json.loads(DEPLOYMENT_PREFLIGHT_PATH.read_text(encoding="utf-8"))
    deployment_qa = json.loads(DEPLOYMENT_QA_PATH.read_text(encoding="utf-8"))
    contract = config["deployment_contract"]
    require(config["experiment_id"] == EXPERIMENT_ID, "config id")
    require(historical_result["status"] == "PASS_SHADOW_AUDIT_ELIGIBLE", "historical gate")
    require(contract["write_candidate_csv"] is False and contract["upload"] is False, "write contract")
    anchor_path = Path(preflight["external_inputs"]["current_router"]["path"])
    require(sha256(anchor_path) == preflight["external_inputs"]["current_router"]["sha256"], "anchor hash")
    require(sha256(E150_PATH) == deployment_qa["candidate_all"]["artifact"]["sha256"], "e150 hash")
    anchor_frame = pd.read_csv(anchor_path)
    e150_frame = pd.read_csv(E150_PATH)
    champion_frame = pd.read_csv(CHAMPION_PATH)
    rows = int(contract["official_rows_expected"])
    require(len(anchor_frame) == len(e150_frame) == len(champion_frame) == rows, "row count")
    require(anchor_frame[KEYS].astype(str).equals(e150_frame[KEYS].astype(str)), "anchor/e150 keys")
    require(anchor_frame[KEYS].astype(str).equals(champion_frame[KEYS].astype(str)), "anchor/champion keys")
    anchor = anchor_frame["label"].to_numpy(np.int8)
    e150 = e150_frame["label"].to_numpy(np.int8)
    champion = champion_frame["label"].to_numpy(np.int8)
    require(np.isin(anchor, [0, 1]).all() and np.isin(e150, [0, 1]).all() and np.isin(champion, [0, 1]).all(), "binary labels")
    e150_addition = (e150 == 1) & (anchor == 0)
    gi_only = (champion == 1) & (e150 == 0)
    require(int(e150_addition.sum()) == int(contract["e150_added_rows_expected"]), "e150 addition count")
    require(int(gi_only.sum()) == int(contract["official_champion_gi_only_rows_expected"]), "GI-only count")
    require(int(np.sum((anchor == 1) & (champion == 0))) == 0, "champion removed anchor")
    row_probability, boundary_probability, type_probability = _load_prediction_ensemble(rows)
    shadow_segments, segment_indices = _build_shadow_segments(
        e150_frame[KEYS].copy(),
        anchor,
        e150,
        row_probability,
        boundary_probability,
        type_probability,
    )
    training, utility, groups = _load_historical_training(config)
    frequency, attempts = _bootstrap_frequency(
        training, utility, groups, shadow_segments, config, model_config
    )
    acceptance = frequency >= float(config["acceptance_frequency"])
    shadow = champion.copy()
    for keep, positions in zip(acceptance, segment_indices, strict=True):
        if not keep:
            shadow[positions] = 0
    accepted_rows = np.zeros(rows, dtype=bool)
    for keep, positions in zip(acceptance, segment_indices, strict=True):
        if keep:
            accepted_rows[positions] = True
    historical_numeric_min = training[prior.NUMERIC].min()
    historical_numeric_max = training[prior.NUMERIC].max()
    numeric_in_range = (
        shadow_segments[prior.NUMERIC].ge(historical_numeric_min)
        & shadow_segments[prior.NUMERIC].le(historical_numeric_max)
    ).all(axis=1).to_numpy(bool)
    known_category = (
        shadow_segments["station"].isin(training["station"].unique())
        & shadow_segments["layer"].isin(training["layer"].unique())
    ).to_numpy(bool)
    accepted_index = np.flatnonzero(acceptance)
    station_rows = e150_frame.loc[accepted_rows, "station"].astype(str).value_counts().sort_index()
    gate_checks = {
        "historical_lower_bound_gate_passed": historical_result["status"] == "PASS_SHADOW_AUDIT_ELIGIBLE",
        "accept_at_least_one_e150_segment": bool(np.any(acceptance)),
        "all_accepted_segments_meet_frequency": bool(np.all(frequency[acceptance] >= float(config["acceptance_frequency"]))) if np.any(acceptance) else False,
        "accepted_segments_use_known_categories": bool(np.all(known_category[accepted_index])) if len(accepted_index) else False,
        "accepted_segments_all_numeric_within_historical_minmax": bool(np.all(numeric_in_range[accepted_index])) if len(accepted_index) else False,
        "anchor_rows_removed": int(np.sum((anchor == 1) & (shadow == 0))) == 0,
        "gi_only_rows_removed": int(np.sum(gi_only & (shadow == 0))) == 0,
        "no_submission_files_created": True,
        "no_uploads": True,
    }
    passed = all(gate_checks.values())
    return {
        "schema_version": "p1.mstcn_official_shadow_lower_bound_veto.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "status": "PASS_LABEL_FREE_SHADOW_RELEVANCE" if passed else "NO_GO_LABEL_FREE_SHADOW_GATE",
        "passed_all_shadow_gates": passed,
        "historical_training": {
            "folds": config["historical_training_folds"],
            "segments": int(len(training)),
            "truth_event_groups": int(np.unique(groups).size),
            "beneficial_segments": int(training["beneficial"].sum()),
        },
        "official_shadow": {
            "rows": rows,
            "anchor_positive_rows": int(anchor.sum()),
            "e150_positive_rows": int(e150.sum()),
            "official_champion_positive_rows": int(champion.sum()),
            "e150_added_rows": int(e150_addition.sum()),
            "official_champion_gi_only_rows": int(gi_only.sum()),
            "e150_added_segments": int(len(shadow_segments)),
            "accepted_segments": int(acceptance.sum()),
            "accepted_e150_rows": int(accepted_rows.sum()),
            "accepted_rows_by_station": {str(key): int(value) for key, value in station_rows.items()},
            "shadow_positive_rows": int(shadow.sum()),
            "anchor_rows_removed": int(np.sum((anchor == 1) & (shadow == 0))),
            "gi_only_rows_removed": int(np.sum(gi_only & (shadow == 0))),
            "acceptance_frequency_quantiles": [float(value) for value in np.quantile(frequency, [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0])],
            "accepted_frequency_minimum": float(np.min(frequency[acceptance])) if np.any(acceptance) else None,
            "numeric_in_historical_minmax_segments": int(numeric_in_range.sum()),
            "known_category_segments": int(known_category.sum()),
            "shadow_label_sha256": hashlib.sha256(shadow.tobytes()).hexdigest(),
        },
        "bootstrap": {
            "replicates": int(config["bootstrap_replicates"]),
            "model_fits": int(2 * int(config["bootstrap_replicates"])),
            "resampling_attempts": attempts,
        },
        "gate_checks": gate_checks,
        "input_hashes": {
            "config": sha256(CONFIG_PATH),
            "model_config": sha256(MODEL_CONFIG_PATH),
            "historical_result": sha256(HISTORICAL_RESULT_PATH),
            "deployment_preflight": sha256(DEPLOYMENT_PREFLIGHT_PATH),
            "deployment_qa": sha256(DEPLOYMENT_QA_PATH),
            "anchor": sha256(anchor_path),
            "e150": sha256(E150_PATH),
            "official_champion": sha256(CHAMPION_PATH),
            "prediction_archives": {path.name: sha256(path) for path in PREDICTION_PATHS},
        },
        "operation_counters": {
            "official_prediction_rows_read": rows,
            "official_test_feature_values_read": 0,
            "official_truth_values_read": 0,
            "submission_files_created": 0,
            "uploads": 0,
            "model_fits": int(2 * int(config["bootstrap_replicates"])),
        },
        "claim_limit": config["claim_limit"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(CONFIG_PATH.exists(), "missing config")
    if not args.execute:
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": "READY_LABEL_FREE_SHADOW", "submission_files_created": 0, "uploads": 0}, indent=2))
        return
    result = execute()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / "result.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps({"status": result["status"], "passed_all_shadow_gates": result["passed_all_shadow_gates"], "official_shadow": result["official_shadow"], "gate_checks": result["gate_checks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
