"""Guarded exactly-once official materializer for the immutable P3 v27 PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "scripts", ROOT / "src"):
    if str(entry) not in os.sys.path:
        os.sys.path.insert(0, str(entry))

import run_p3_multiscale_wavelet_scattering_residual_cycle_20260901_v27 as science  # noqa: E402
import run_p3_path_signature_residual_cycle_20260901_v23 as v23  # noqa: E402

EXPERIMENT_ID = "p3_multiscale_wavelet_scattering_materializer_20260901_v27m1"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
RUNNER = Path(__file__)
KEYS = ["case_id", "station", "lead_h"]
VALUE_COLUMNS = ["hs", "tp", "hmax", "wvdir", "wspd", "gust", "wdir", "airt", "relh", "caph"]


class ContractError(RuntimeError):
    """Fail-closed materialization contract violation."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    guard = config["deployment_guards"]
    expected = {"absolute_action_p99_m_lte": 0.12, "absolute_action_max_m_lte": 0.20, "action_rms_m_lte": 0.05, "maximum_station_action_p99_m_lte": 0.15, "maximum_lead_action_p99_m_lte": 0.15, "prediction_min_m_gte": 0.0, "prediction_max_m_lte": 30.0, "row_deletion": 0}
    checks = {
        "schema": config["schema_version"] == "p3.multiscale_wavelet_scattering.materializer.config.v27m1",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "status": config["status"] == "PREREGISTERED_BEFORE_OFFICIAL_READ",
        "candidate": config["candidate"]["name"] == "P3_2_SCATTER336_RIDGE1024_ADD10",
        "alpha": float(config["candidate"]["ridge_alpha"]) == 1024.0,
        "blend": float(config["candidate"]["additive_residual_weight"]) == 0.10,
        "guards": all(float(guard[key]) == value for key, value in expected.items()),
        "limits": all(value == 0 for key, value in config["operation_limits"].items() if key in {"hidden_truth_rows", "score_file_rows", "uploads", "csv_on_guard_fail"}),
        "no_adaptation": not config["operation_limits"]["result_adaptive_tuning"] and not config["operation_limits"]["posthoc_routing"],
    }
    if not all(checks.values()):
        raise ContractError(f"v27m1 contract drift: {checks}")
    return config


def verify_internal(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = config["internal_contract"]
    result_path = ROOT / contract["result_path"]
    qa_path = ROOT / contract["qa_path"]
    if sha256(result_path) != contract["result_sha256"] or sha256(qa_path) != contract["qa_sha256"]:
        raise ContractError("immutable v27 result/QA hash drift")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    candidate = next(item for item in result["candidates"] if item["name"] == config["candidate"]["name"])
    if result["decision"] != contract["required_science_decision"] or candidate["decision"] != contract["required_candidate_decision"] or qa["decision"] != contract["required_qa_decision"]:
        raise ContractError("immutable v27 PASS contract absent")
    return result, qa


def preflight() -> dict[str, Any]:
    config = load_config()
    verify_internal(config)
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("exactly-once materializer namespace is consumed")
    payload = {"schema_version": "p3.multiscale_wavelet_scattering.materializer.preflight.v27m1", "experiment_id": EXPERIMENT_ID, "status": "ZERO_OFFICIAL_ROW_PREFLIGHT_PASS", "candidate": config["candidate"], "guards": config["deployment_guards"], "config_sha256": sha256(CONFIG), "runner_sha256": sha256(RUNNER), "official_rows_read": 0, "hidden_truth_rows_read": 0, "score_file_rows_read": 0, "submission_csv_created": 0, "uploads": 0}
    payload["preflight_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def resolve_inputs(config: dict[str, Any]) -> dict[str, Path]:
    raw = os.environ.get(config["source_contract"]["environment"])
    if not raw:
        raise ContractError("P3_DATA_DIR is required")
    data = Path(raw).resolve()
    source = config["source_contract"]
    paths = {"context": data / source["test_context_filename"], "index": data / source["test_index_filename"], "sample": data / source["sample_filename"], "champion": ROOT / source["champion_path"]}
    expected = {"context": source["test_context_sha256"], "index": source["test_index_sha256"], "sample": source["sample_sha256"], "champion": source["champion_sha256"]}
    for role, path in paths.items():
        if not path.is_file() or sha256(path) != expected[role]:
            raise ContractError(f"official input drift: {role}")
    return paths


def fit_full_model() -> tuple[Ridge, np.ndarray, np.ndarray, dict[str, Any]]:
    cases, targets, reference, _ = v23.case_surface()
    features, receipt = science.surface_features(cases)
    center = np.median(features, axis=0)
    q25, q75 = np.quantile(features, (0.25, 0.75), axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    train_z = np.clip((features - center) / scale, -8.0, 8.0)
    residual = targets - reference
    low, high = np.quantile(residual, (0.025, 0.975), axis=0)
    target = np.clip(residual, low, high)
    model = Ridge(alpha=1024.0, fit_intercept=True, solver="cholesky")
    model.fit(train_z, target)
    model_receipt = {"historical_cases": int(len(cases)), "historical_target_rows": int(targets.size), "feature_receipt": receipt, "ridge_alpha": 1024.0, "additive_residual_weight": 0.10, "target_winsor": [0.025, 0.975], "coefficient_l2": float(np.linalg.norm(model.coef_)), "row_deletion": 0, "full_deployment_fits": 1}
    return model, center, scale, model_receipt


def official_features(context: pd.DataFrame, index: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    pairs = index[["case_id", "station"]].drop_duplicates(ignore_index=True)
    if len(pairs) != 200 or pairs.duplicated(["case_id", "station"]).any():
        raise ContractError("official case grain differs")
    grouped = {(str(case), str(station)): group.sort_values("step_minute") for (case, station), group in context.groupby(["case_id", "station"], sort=False, observed=True)}
    features = np.empty((200, science.FEATURE_COUNT), dtype=np.float64)
    expected_steps = np.arange(-2880, 1, 10, dtype=np.int64)
    for position, row in enumerate(pairs.itertuples(index=False)):
        group = grouped.get((str(row.case_id), str(row.station)))
        if group is None or len(group) != 289 or not np.array_equal(group["step_minute"].to_numpy(np.int64), expected_steps):
            raise ContractError("official case-local context geometry differs")
        features[position] = science.scattering_features(group[VALUE_COLUMNS].to_numpy(np.float64))
    if not np.isfinite(features).all():
        raise ContractError("official scattering features are non-finite")
    return features, pairs, {"rows": 200, "columns": science.FEATURE_COUNT, "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest()}


def geometry(frame: pd.DataFrame, candidate: np.ndarray, champion: np.ndarray, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, bool]]:
    action = candidate - champion
    absolute = np.abs(action)
    station = {str(key): {"rows": int(len(group)), "action_p99_m": float(np.quantile(np.abs(group["action"]), 0.99)), "action_max_m": float(np.max(np.abs(group["action"]))), "action_rms_m": float(np.sqrt(np.mean(np.square(group["action"]))))} for key, group in frame.assign(action=action).groupby("station", observed=True)}
    lead = {str(int(key)): {"rows": int(len(group)), "action_p99_m": float(np.quantile(np.abs(group["action"]), 0.99)), "action_max_m": float(np.max(np.abs(group["action"]))), "action_rms_m": float(np.sqrt(np.mean(np.square(group["action"]))))} for key, group in frame.assign(action=action).groupby("lead_h", observed=True)}
    values = {"rows": int(len(candidate)), "absolute_action_m": {"p99": float(np.quantile(absolute, 0.99)), "max": float(absolute.max()), "rms": float(np.sqrt(np.mean(np.square(action))))}, "prediction_range_m": [float(candidate.min()), float(candidate.max())], "station": station, "lead": lead, "changed_rows": int(np.count_nonzero(action)), "row_deletion": 0}
    guard = config["deployment_guards"]
    checks = {"rows_exact": len(candidate) == 1200, "candidate_finite": bool(np.isfinite(candidate).all()), "prediction_min": values["prediction_range_m"][0] >= guard["prediction_min_m_gte"], "prediction_max": values["prediction_range_m"][1] <= guard["prediction_max_m_lte"], "action_p99": values["absolute_action_m"]["p99"] <= guard["absolute_action_p99_m_lte"], "action_max": values["absolute_action_m"]["max"] <= guard["absolute_action_max_m_lte"], "action_rms": values["absolute_action_m"]["rms"] <= guard["action_rms_m_lte"], "station_set": set(station) == set(guard["required_station_set"]), "lead_set": {int(value) for value in lead} == set(guard["required_lead_set"]), "station_p99": max(item["action_p99_m"] for item in station.values()) <= guard["maximum_station_action_p99_m_lte"], "lead_p99": max(item["action_p99_m"] for item in lead.values()) <= guard["maximum_lead_action_p99_m_lte"], "row_deletion_zero": values["row_deletion"] == guard["row_deletion"]}
    return values, checks


def run() -> dict[str, Any]:
    config = load_config()
    verify_internal(config)
    pre = preflight()
    write_new(LOCK, canonical({"experiment_id": EXPERIMENT_ID, "status": "ATTEMPT_CONSUMED_EXACTLY_ONCE_MATERIALIZER", "created_at_utc": datetime.now(UTC).isoformat(), "config_sha256": sha256(CONFIG), "runner_sha256": sha256(RUNNER), "preflight_sha256": pre["preflight_sha256"], "guards": config["deployment_guards"], "hidden_truth_rows_before_lock": 0, "score_rows_before_lock": 0}))
    ARTIFACT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=True)
    paths = resolve_inputs(config)
    source = config["source_contract"]
    index = pd.read_csv(paths["index"], dtype={"case_id": "string", "station": "string"})
    sample = pd.read_csv(paths["sample"], dtype={"case_id": "string", "station": "string"})
    champion_frame = pd.read_csv(paths["champion"], dtype={"case_id": "string", "station": "string"})
    context = pd.read_parquet(paths["context"], columns=source["context_columns"])
    structure = {"index_schema": list(index.columns) == KEYS, "sample_schema": list(sample.columns) == KEYS + ["hs_pred"], "champion_schema": list(champion_frame.columns) == KEYS + ["hs_pred"], "index_rows": len(index) == 1200, "sample_rows": len(sample) == 1200, "champion_rows": len(champion_frame) == 1200, "context_rows": len(context) == 57800, "index_unique": not index.duplicated(KEYS).any(), "sample_key_order": sample[KEYS].equals(index[KEYS]), "champion_key_order": champion_frame[KEYS].equals(index[KEYS])}
    if not all(structure.values()):
        raise ContractError(f"official schema/key/order failed: {structure}")
    model, center, scale, model_receipt = fit_full_model()
    features, pairs, feature_receipt = official_features(context, index)
    test_z = np.clip((features - center) / scale, -8.0, 8.0)
    residual = np.asarray(model.predict(test_z), dtype=np.float64)
    pair_prediction = pd.DataFrame({"case_id": pairs["case_id"], "station": pairs["station"]})
    for position, lead_h in enumerate(v23.LEADS):
        pair_prediction[str(int(lead_h))] = residual[:, position]
    long_residual = pair_prediction.melt(id_vars=["case_id", "station"], var_name="lead_h", value_name="residual")
    long_residual["lead_h"] = long_residual["lead_h"].astype(int)
    aligned = index.merge(long_residual, on=KEYS, how="left", validate="one_to_one")
    if aligned["residual"].isna().any():
        raise ContractError("official residual alignment failed")
    champion = champion_frame["hs_pred"].to_numpy(np.float64)
    candidate = np.clip(champion + 0.10 * aligned["residual"].to_numpy(np.float64), 0.0, 30.0)
    geometry_values, geometry_checks = geometry(index, candidate, champion, config)
    guard_checks = {**structure, **geometry_checks}
    passed = bool(all(guard_checks.values()))
    arrays_path = ARTIFACT / "official-action-geometry.npz"
    np.savez_compressed(arrays_path, candidate=candidate, champion=champion, action=candidate - champion, case_id=index["case_id"].astype(str).to_numpy(dtype="U8"), station=index["station"].astype(str).to_numpy(dtype="U5"), lead_h=index["lead_h"].to_numpy(np.int16))
    submission = {"created": False, "path": None, "rows": 0, "bytes": 0, "sha256": None}
    if passed:
        output = ARTIFACT / "submission" / "P3_V27M1_SCATTER336_RIDGE1024_ADD10" / "P3_submission.csv"
        frame = index[KEYS].copy()
        frame["hs_pred"] = candidate
        payload = frame.to_csv(index=False, lineterminator="\n").encode()
        write_new(output, payload)
        submission = {"created": True, "path": str(output), "rows": 1200, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        write_new(output.parent / "submission-info.txt", (b"title: P3 v27 fixed wavelet-scattering residual\nsummary: Exact uniform KMA 0.425 champion plus frozen SCATTER336 Ridge1024 residual at weight 0.10; full historical fit, no routing or retuning.\nstatus: READY_NOT_UPLOADED\n"))
        write_new(output.parent / "SET_MANIFEST.json", canonical({"experiment_id": EXPERIMENT_ID, "candidate": config["candidate"], "submission": submission, "uploads": 0}))
    result = {"schema_version": "p3.multiscale_wavelet_scattering.materializer.result.v27m1", "experiment_id": EXPERIMENT_ID, "created_at_utc": datetime.now(UTC).isoformat(), "status": "READY_NOT_UPLOADED" if passed else "DEPLOYMENT_GUARD_FAIL_NO_CSV", "candidate": config["candidate"], "guard_result": {"passed": passed, "checks": guard_checks, "geometry": geometry_values}, "model_receipt": model_receipt, "official_feature_receipt": feature_receipt, "action_artifact": {"path": str(arrays_path), "sha256": sha256(arrays_path)}, "submission": submission, "data_access": {"official_test_context_rows_read": int(len(context)), "official_test_index_rows_read": int(len(index)), "official_sample_rows_read": int(len(sample)), "official_champion_prediction_rows_read": int(len(champion_frame)), "hidden_truth_rows_read": 0, "score_file_rows_read": 0, "uploads": 0}, "provenance": {"config_sha256": sha256(CONFIG), "runner_sha256": sha256(RUNNER), "internal_result_sha256": config["internal_contract"]["result_sha256"], "internal_qa_sha256": config["internal_contract"]["qa_sha256"], "input_sha256": {role: sha256(path) for role, path in paths.items()}}, "execution": {"python": platform.python_version(), "result_based_tuning": False, "posthoc_routing": False, "row_deletion": 0}}
    result_path = ARTIFACT / "result.json"
    write_new(result_path, canonical(result))
    write_new(REPORT / "result.json", canonical(result))
    write_new(REPORT / "report-source.md", (f"# P3 v27m1 guarded materializer\n\n결론: **{result['status']}**. action p99 {geometry_values['absolute_action_m']['p99']:.6f}m, max {geometry_values['absolute_action_m']['max']:.6f}m, RMS {geometry_values['absolute_action_m']['rms']:.6f}m. Hidden/score/upload 0.\n").encode())
    print(json.dumps({"status": result["status"], "guard_pass": passed, "submission": submission, "uploads": 0}, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.execute:
        raise SystemExit("Choose exactly one mode")
    value = preflight() if args.preflight else run()
    if args.preflight:
        print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
