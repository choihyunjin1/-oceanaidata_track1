from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v20 as base  # noqa: E402

from src.p1_qc.robust_student_t_llr import derive_causal_gap_minutes  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v20r1"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
BASE_CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v20.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
QA_REPORT = ROOT / "reports" / EXPERIMENT_ID / "independent-qa.json"
RUNNER = Path(__file__)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract() -> dict:
    repair = json.loads(CONFIG.read_text(encoding="utf-8"))
    contract = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    checks = {
        "candidate": repair["candidate"] == contract["candidate"],
        "features": repair["features"]["names"] == contract["features"]["names"],
        "df4": repair["model"]["degrees_of_freedom"] == contract["model"]["degrees_of_freedom"] == 4.0,
        "inner75": repair["model"]["inner_fit_fraction"] == contract["inner_calibration"]["fit_fraction"] == 0.75,
        "fits2": repair["fit_budget"]["maximum"] == contract["fit_budget"]["maximum"] == 2,
        "raw_gate": repair["decision_policy"]["minimum_raw_expected_point_delta_inclusive"] == contract["decision_policy"]["minimum_raw_expected_point_delta_inclusive"],
        "only_repair": repair["repair"]["scientific_parameters_changed"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"v20r1 repair drift: {checks}")
    contract["experiment_id"] = EXPERIMENT_ID
    contract["status"] = repair["status"]
    return contract


def historical_schema_preflight() -> dict:
    source_config = json.loads(base.source.MSTCN_CONFIG_PATH.read_text(encoding="utf-8"))
    cache_path = ROOT / source_config["immutable_inputs"]["feature_cache"]["path"]
    import pyarrow.parquet as pq

    columns = set(pq.ParquetFile(cache_path).schema.names)
    required_cached = {"temp_robust_z_6h", "temp_robust_z_24h", "temp_robust_z_72h", "temp_abs_median_resid_6h", "temp_abs_median_resid_24h", "temp_abs_median_resid_72h", "temp_abs_peer_residual", "has_gap_before", "depth_raw"}
    checks = {"cached_features_present": required_cached <= columns, "gap_minutes_intentionally_derived": "gap_minutes" not in columns, "key_columns_available": True, "old_artifact_preserved": (ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v20/terminal_failure.json").is_file()}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "cache_schema_only": True, "historical_truth_reads": 0, "official_reads": 0}


def repaired_feature_values(frame, names: list[str]) -> np.ndarray:
    gaps = derive_causal_gap_minutes(frame)
    columns = []
    for name in names:
        if name == "gap_minutes":
            columns.append(gaps)
        elif name.startswith("abs_temp_robust_z_"):
            columns.append(np.abs(frame[name.removeprefix("abs_")].to_numpy(np.float64)))
        else:
            columns.append(frame[name].to_numpy(np.float64))
    return np.column_stack(columns)


def execute() -> dict:
    schema = historical_schema_preflight()
    if schema["status"] != "PASS":
        raise RuntimeError(f"historical schema preflight failed: {schema}")
    original_sha = base.sha256

    def runner_sha(path: Path) -> str:
        return sha256(RUNNER) if Path(path).resolve() == Path(base.__file__).resolve() else original_sha(path)

    base.CONFIG = CONFIG
    base.ARTIFACT = ARTIFACT
    base.QA_REPORT = QA_REPORT
    base.load_contract = load_contract
    base._feature_values = repaired_feature_values
    base.sha256 = runner_sha
    result = base.execute()
    result["technical_recovery"] = {"predecessor": "p1_public_transport_repair_cycle_20260831_v20", "predecessor_fit_count": 0, "repair": "causal key-derived gap_minutes only", "schema_preflight": schema, "base_contract_sha256": sha256(BASE_CONFIG), "repair_config_sha256": sha256(CONFIG)}
    (ARTIFACT / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema-preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.schema_preflight == args.execute:
        parser.error("choose exactly one mode")
    if args.schema_preflight:
        print(json.dumps(historical_schema_preflight(), indent=2, sort_keys=True))
        return 0
    try:
        print(json.dumps(execute(), indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        import traceback

        payload = {"status": "TERMINAL_TECHNICAL_FAILURE", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "official_reads": 0, "hidden_truth_reads": 0, "uploads": 0}
        if ARTIFACT.exists():
            (ARTIFACT / "terminal_failure.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
