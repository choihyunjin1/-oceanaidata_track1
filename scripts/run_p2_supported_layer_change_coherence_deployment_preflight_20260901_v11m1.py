"""Label-free official geometry preflight for the exact P2 v11r1 action."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_p2_supported_layer_change_coherence_20260901_v11r1 as science  # noqa: E402

from p2_restore.features import _nearest_public_baseline  # noqa: E402
from p2_restore.profile_projection import (  # noqa: E402
    project_profiles_vectorized,
    public_endpoint_frame,
)

EXPERIMENT_ID = "p2_supported_layer_change_coherence_deployment_preflight_20260901_v11m1"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)


class ContractError(RuntimeError):
    """Fail-closed deployment contract violation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    value = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID or config["status"] != "PREREGISTERED_NOT_EXECUTED":
        raise ContractError("deployment preregistration drift")
    guard = config["deployment_guards"]
    expected = {
        "active_share_lte": 0.5,
        "absolute_action_p99_C_lte": 0.5,
        "absolute_action_max_C_lte": 2.5,
        "profile_projection_exact_noop_atol_C": 1e-12,
    }
    if any(float(guard[key]) != value for key, value in expected.items()):
        raise ContractError("deployment guard drift")
    if config["operation_limits"]["result_adaptive_tuning"]:
        raise ContractError("adaptive deployment tuning is forbidden")
    return config


def preflight() -> dict[str, Any]:
    config = load_config()
    internal = config["internal_pass_contract"]
    internal_path = ROOT / internal["result_path"]
    if sha256_file(internal_path) != internal["result_sha256"]:
        raise ContractError("internal PASS result hash drift")
    result = json.loads(internal_path.read_text(encoding="utf-8"))
    if result["status"] != internal["required_status"]:
        raise ContractError("internal PASS condition absent")
    if result["hashes"]["action_npz"] != internal["action_sha256"]:
        raise ContractError("internal action hash drift")
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OFFICIAL_ROW_PREFLIGHT_PASS",
        "internal_pass_verified": True,
        "guards": config["deployment_guards"],
        "config_sha256": sha256_file(CONFIG),
        "runner_sha256": sha256_file(RUNNER),
        "official_rows_read": 0,
        "hidden_truth_rows_read": 0,
        "score_file_rows_read": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }
    payload["preflight_sha256"] = sha256_json(payload)
    return payload


def resolve_inputs(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    raw = os.environ.get(config["source_contract"]["p2_data_environment"])
    if not raw:
        raise ContractError("P2_DATA_DIR is required")
    data_dir = Path(raw).resolve()
    observations = data_dir / config["source_contract"]["observations_filename"]
    test_index = data_dir / config["source_contract"]["test_index_filename"]
    champion = Path(config["source_contract"]["champion_path"])
    pins = {
        observations: config["source_contract"]["observations_sha256"],
        test_index: config["source_contract"]["test_index_sha256"],
        champion: config["source_contract"]["champion_sha256"],
    }
    for path, digest in pins.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ContractError(f"official geometry input drift: {path}")
    return observations, test_index, champion


def endpoint_baseline(
    observations: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, dict[str, Any]]:
    public_layers = (1, 5, 6, 7, 8)
    public = observations.loc[observations["layer"].isin(public_layers)].copy()
    temp = public.pivot(index="time", columns="layer", values="temp")
    nominal = public.pivot(index="time", columns="layer", values="nominal_depth")
    time_index = pd.DatetimeIndex(test["time"])
    public_temp = temp.reindex(time_index).reindex(columns=public_layers).to_numpy(float)
    public_nominal = nominal.reindex(time_index).reindex(columns=public_layers).to_numpy(float)
    if "nominal_depth" in test:
        target_nominal = pd.to_numeric(test["nominal_depth"], errors="coerce").to_numpy(float)
    else:
        target = observations.loc[:, ["station", "time", "layer", "nominal_depth"]]
        aligned = test[["station", "time", "layer"]].merge(
            target, on=["station", "time", "layer"], how="left", validate="one_to_one"
        )
        target_nominal = aligned["nominal_depth"].to_numpy(float)
    baseline = _nearest_public_baseline(public_temp, public_nominal, target_nominal)
    return baseline, {
        "rows": int(len(baseline)),
        "finite": bool(np.isfinite(baseline).all()),
        "minimum_C": float(np.nanmin(baseline)),
        "maximum_C": float(np.nanmax(baseline)),
    }


def evaluate_guards(
    config: dict[str, Any],
    test_raw: pd.DataFrame,
    champion: pd.DataFrame,
    observations: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    keys = config["source_contract"]["key_columns"]
    expected_columns = config["source_contract"]["submission_columns"]
    test = test_raw.copy()
    test["time"] = pd.to_datetime(test["time"], utc=True)
    observations = observations.copy()
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    champion_temp = pd.to_numeric(champion["temp"], errors="coerce").to_numpy(float)
    baseline, baseline_receipt = endpoint_baseline(observations, test)
    if not np.isfinite(baseline).all():
        raise ContractError("official endpoint baseline is nonfinite")
    science_config = science.load_config()
    score, weight, influence = science.build_supported_layer_coherence_writeable(
        observations, test, science_config
    )
    candidate = baseline + weight * (champion_temp - baseline)
    action = candidate - champion_temp
    active = weight < 1.0
    endpoints = public_endpoint_frame(observations)
    projection = project_profiles_vectorized(test[keys], candidate, endpoints)
    projection_delta = np.abs(projection.prediction - candidate)
    guard = config["deployment_guards"]
    geometry = {
        "rows": int(len(candidate)),
        "active_rows": int(active.sum()),
        "active_share": float(active.mean()),
        "absolute_action_C": {
            "p50": float(np.quantile(np.abs(action), 0.5)),
            "p90": float(np.quantile(np.abs(action), 0.9)),
            "p95": float(np.quantile(np.abs(action), 0.95)),
            "p99": float(np.quantile(np.abs(action), 0.99)),
            "p999": float(np.quantile(np.abs(action), 0.999)),
            "max": float(np.max(np.abs(action))),
            "rms": float(np.sqrt(np.mean(np.square(action)))),
        },
        "candidate_range_C": [float(candidate.min()), float(candidate.max())],
        "champion_range_C": [float(champion_temp.min()), float(champion_temp.max())],
        "endpoint_baseline": baseline_receipt,
        "profile_projection": {
            "eligible_rows": int(projection.eligible_mask.sum()),
            "active_rows": int(projection.active_mask.sum()),
            "max_abs_delta_C": float(projection_delta.max()),
        },
        "influence": influence,
        "finite_anomaly_score_rows": int(np.isfinite(score).sum()),
    }
    checks = {
        "rows_exact": len(test_raw) == int(config["source_contract"]["expected_rows"]),
        "test_key_unique": not test_raw.duplicated(keys).any(),
        "champion_schema_exact": list(champion.columns) == expected_columns,
        "champion_key_order_exact": champion[keys].equals(test_raw[keys]),
        "champion_finite": bool(np.isfinite(champion_temp).all()),
        "candidate_finite": bool(np.isfinite(candidate).all()),
        "active_share_lte": geometry["active_share"] <= float(guard["active_share_lte"]),
        "action_p99_lte": geometry["absolute_action_C"]["p99"]
        <= float(guard["absolute_action_p99_C_lte"]),
        "action_max_lte": geometry["absolute_action_C"]["max"]
        <= float(guard["absolute_action_max_C_lte"]),
        "public_range_projection_invariant": geometry["profile_projection"]["active_rows"] == 0,
        "profile_projection_exact_noop": geometry["profile_projection"]["max_abs_delta_C"]
        <= float(guard["profile_projection_exact_noop_atol_C"]),
    }
    return {"checks": checks, "geometry": geometry, "passed": bool(all(checks.values()))}, {
        "candidate": candidate,
        "action": action,
        "weight": weight,
        "score": score,
        "baseline": baseline,
        "champion": champion_temp,
    }


def run() -> dict[str, Any]:
    if ARTIFACT.exists():
        raise FileExistsError(f"exactly-once deployment artifact exists: {ARTIFACT}")
    config = load_config()
    internal = json.loads((ROOT / config["internal_pass_contract"]["result_path"]).read_text(encoding="utf-8"))
    if internal["status"] != config["internal_pass_contract"]["required_status"]:
        raise ContractError("internal PASS required before official geometry read")
    ARTIFACT.mkdir(parents=True)
    atomic_json(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": sha256_file(CONFIG),
            "runner_sha256": sha256_file(RUNNER),
            "internal_result_sha256": config["internal_pass_contract"]["result_sha256"],
            "guards": config["deployment_guards"],
            "hidden_truth_access_before_lock": 0,
            "score_access_before_lock": 0,
        },
    )
    observations_path, test_path, champion_path = resolve_inputs(config)
    observations = pd.read_csv(
        observations_path, dtype={"station": "string", "time": "string"}
    )
    test_raw = pd.read_csv(test_path, dtype={"station": "string", "time": "string"})
    champion = pd.read_csv(champion_path, dtype={"station": "string", "time": "string"})
    guard_result, arrays = evaluate_guards(config, test_raw, champion, observations)
    action_path = ARTIFACT / "official_action_geometry.npz"
    np.savez_compressed(action_path, **arrays)
    action_sha = sha256_file(action_path)
    submission_receipt: dict[str, Any] = {
        "created": False,
        "path": None,
        "sha256": None,
        "rows": 0,
    }
    if guard_result["passed"]:
        keys = config["source_contract"]["key_columns"]
        submission = test_raw[keys].copy()
        submission["temp"] = arrays["candidate"]
        output = ARTIFACT / "submission" / "P2_V11M1_SUPPORTED_LAYER_CHANGE_COHERENCE" / "P2_submission.csv"
        output.parent.mkdir(parents=True)
        submission.to_csv(output, index=False, lineterminator="\n")
        submission_receipt = {
            "created": True,
            "path": str(output),
            "sha256": sha256_file(output),
            "rows": int(len(submission)),
            "bytes": int(output.stat().st_size),
        }
    result = {
        "schema_version": "p2.supported_layer_change_coherence.deployment_preflight.result.20260901.v11m1",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_NOT_UPLOADED" if guard_result["passed"] else "DEPLOYMENT_GUARD_FAIL_NO_CSV",
        "guard_result": guard_result,
        "action_artifact": {"path": str(action_path), "sha256": action_sha},
        "submission": submission_receipt,
        "operation_counters": {
            "observations_rows_read": int(len(observations)),
            "official_test_index_rows_read": int(len(test_raw)),
            "official_champion_rows_read": int(len(champion)),
            "hidden_truth_rows_read": 0,
            "score_file_rows_read": 0,
            "uploads": 0,
        },
        "hashes": {
            "config": sha256_file(CONFIG),
            "runner": sha256_file(RUNNER),
            "observations": sha256_file(observations_path),
            "test_index": sha256_file(test_path),
            "champion": sha256_file(champion_path),
        },
    }
    atomic_json(ARTIFACT / "result.json", result)
    REPORT.mkdir(parents=True, exist_ok=True)
    atomic_json(REPORT / "result.json", result)
    (REPORT / "report-source.md").write_text(
        "# P2 v11m1 deployment geometry preflight\n\n"
        f"결론: `{result['status']}`. active share {guard_result['geometry']['active_share']:.6f}, "
        f"abs action p99 {guard_result['geometry']['absolute_action_C']['p99']:.6f}°C, "
        f"max {guard_result['geometry']['absolute_action_C']['max']:.6f}°C. "
        "Hidden/score/upload access는 모두 0이다.\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.execute:
        raise SystemExit("Choose exactly one mode")
    value = preflight() if args.preflight else run()
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
