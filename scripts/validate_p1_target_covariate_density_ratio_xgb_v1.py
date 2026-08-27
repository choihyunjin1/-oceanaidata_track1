"""Independent QA for the completed P1 density/fallback experiment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p1_qc.data import load_dataset
from p1_qc.features import FeatureBundle
from p1_qc.pipeline import apply_postprocess
from p1_qc.rules import detect_plateaus, detect_singleton_spikes
from p1_qc.submission import build_submission, validate_submission, write_submission


ARTIFACT = PROJECT_ROOT / "artifacts/p1_target_covariate_density_ratio_xgb_v1"
CONFIG = PROJECT_ROOT / "configs/experiments/p1_target_covariate_density_ratio_xgb_v1.json"
KST = ZoneInfo("Asia/Seoul")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _cached_bundle(frame: pd.DataFrame, parquet: Path, metadata_path: Path) -> FeatureBundle:
    metadata = _json(metadata_path)
    features = pd.read_parquet(parquet)
    columns = tuple(metadata["feature_columns"])
    categorical = tuple(metadata["categorical_columns"])
    if (
        len(features) != len(frame)
        or tuple(features.columns) != columns
        or metadata["source_sha256"] != frame.attrs["source_sha256"]
        or _sha(parquet) != metadata["parquet_sha256"]
    ):
        raise RuntimeError("test feature cache binding failed")
    features.index = frame.index.copy()
    features.attrs["feature_mode"] = "offline"
    return FeatureBundle(features, columns, categorical)


def main() -> int:
    qa_dir = ARTIFACT / "qa"
    if qa_dir.exists():
        raise FileExistsError(qa_dir)
    config = _json(CONFIG)
    result = _json(ARTIFACT / "result.json")
    domain = _json(ARTIFACT / "domain_audit.json")
    manifest = _json(ARTIFACT / "manifest.json")
    manifest_mismatches = []
    for relative, pin in manifest["artifacts"].items():
        path = PROJECT_ROOT / relative
        if path.stat().st_size != pin["bytes"] or _sha(path) != pin["sha256"]:
            manifest_mismatches.append(relative)

    data_raw = os.environ.get("P1_DATA_DIR")
    if not data_raw:
        raise RuntimeError("P1_DATA_DIR is required")
    data_dir = Path(data_raw).expanduser().resolve(strict=True)
    test = load_dataset(data_dir / "test.csv", kind="test", audit=True)
    test_bundle = _cached_bundle(
        test,
        PROJECT_ROOT / config["paths"]["test_feature_cache"],
        PROJECT_ROOT / config["paths"]["test_feature_metadata"],
    )

    model_path = ARTIFACT / "models/P1_EVENT_DAY_BALANCED_LGBM_FALLBACK_V1.joblib"
    candidate_path = ARTIFACT / "candidate/P1_EVENT_DAY_BALANCED_LGBM_FALLBACK_V1.csv"
    payload = joblib.load(model_path)
    if len(payload["encoder"].feature_columns) != 80:
        raise RuntimeError("fallback encoder feature count changed")
    runner_path = PROJECT_ROOT / "scripts/run_p1_meaningful_learning_curve_generation_v1.py"
    spec = importlib.util.spec_from_file_location("p1_frozen_curve_qa", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen fallback predictor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matrix = payload["encoder"].transform(test_bundle)
    probability = module._predict_loaded_candidate(
        payload["branch"],
        payload["packages"],
        matrix,
        test.loc[:, ["station", "layer", "time"]].reset_index(drop=True),
    )
    plateau = detect_plateaus(test).to_numpy(dtype=bool)
    spike = detect_singleton_spikes(test).to_numpy(dtype=bool)
    prediction = apply_postprocess(
        test, probability, plateau, spike, payload["postprocess"]
    )
    anomaly = np.full(len(test), "", dtype=object)
    anomaly[plateau & prediction.astype(bool)] = "flatline"
    anomaly[spike & prediction.astype(bool)] = "spike"
    reproduced = build_submission(test, prediction, anomaly)
    reproduced_path = qa_dir / "fallback_reproduced.csv"
    qa_dir.mkdir(parents=True, exist_ok=False)
    write_submission(reproduced, reproduced_path)
    fallback_byte_identical = candidate_path.read_bytes() == reproduced_path.read_bytes()

    round_a_expected = PROJECT_ROOT / config["paths"]["round_a_candidate"]
    round_a_reproduced = ARTIFACT / "round_a/reproduced.csv"
    round_a_byte_identical = round_a_expected.read_bytes() == round_a_reproduced.read_bytes()
    primary = domain["seeds"][str(config["validation"]["primary_seed"])]
    structural_failure_exact = {
        "daily_ess_below_0_25": primary["daily_ratio_ess_fraction"] < 0.25,
        "row_ess_below_0_25": primary["row_ratio_ess_fraction"] < 0.25,
        "at_least_one_station_layer_below_0_20": min(
            primary["per_station_layer_daily_ratio_ess_fraction"].values()
        )
        < 0.20,
        "support_complete": not primary["missing_target_station_layer_support"],
        "group_overlap_zero": primary["all_groups_disjoint"],
        "domain_target_reads_zero": domain["label_reads_by_domain_model"] == 0,
        "official_score_reads_zero": domain["official_score_reads"] == 0,
    }
    candidate_validation = validate_submission(candidate_path, test)
    checks = {
        "pre_qa_manifest_mismatch_count": len(manifest_mismatches),
        "round_a_byte_identical": round_a_byte_identical,
        "fallback_byte_identical": fallback_byte_identical,
        "fallback_model_feature_count": len(payload["encoder"].feature_columns),
        "fallback_seed_count": len(payload["packages"]),
        "fallback_branch": payload["branch"],
        "domain_structural_gate_failed": domain["structural_gate_passed"] is False,
        "structural_failure_exact": structural_failure_exact,
        "candidate_validation": candidate_validation,
        "protected_inputs_unchanged": result["protected_inputs_unchanged"],
        "submission_uploads": result["submission_uploads"],
        "test_label_reads": result["test_label_reads"],
    }
    passed = (
        not manifest_mismatches
        and round_a_byte_identical
        and fallback_byte_identical
        and len(payload["encoder"].feature_columns) == 80
        and len(payload["packages"]) == 3
        and payload["branch"] == "event_day_balanced_binary_lgbm"
        and domain["structural_gate_passed"] is False
        and all(structural_failure_exact.values())
        and candidate_validation["rows"] == 169011
        and candidate_validation["test_order_match"]
        and result["protected_inputs_unchanged"]
        and result["submission_uploads"] == 0
        and result["test_label_reads"] == 0
    )
    receipt = {
        "schema_version": "p1_target_covariate_density_ratio.independent_qa.v1",
        "validated_at_kst": datetime.now(KST).isoformat(),
        "decision": "QA_PASS" if passed else "QA_FAIL",
        "checks": checks,
        "candidate": {
            "path": str(candidate_path),
            "bytes": candidate_path.stat().st_size,
            "sha256": _sha(candidate_path),
            "reproduced_path": str(reproduced_path),
            "reproduced_sha256": _sha(reproduced_path),
            "byte_identical": fallback_byte_identical,
            "local_status": "RESEARCH_ONLY_FALLBACK",
            "frozen_local_full_fraction_delta": 0.00418648488187523,
            "frozen_local_ci90": [-0.00942869728988131, 0.016910901472633685],
            "frozen_worst_station_delta": -0.010278109678992675,
        },
        "round_a": {
            "sha256": _sha(round_a_reproduced),
            "byte_identical": round_a_byte_identical,
        },
        "domain_no_go": {
            "primary_daily_ess_fraction": primary["daily_ratio_ess_fraction"],
            "primary_row_ess_fraction": primary["row_ratio_ess_fraction"],
            "minimum_station_layer_daily_ess_fraction": min(
                primary["per_station_layer_daily_ratio_ess_fraction"].values()
            ),
            "oof_auc": primary["oof_auc"],
        },
    }
    _write_json_new(qa_dir / "independent_validation.json", receipt)
    if not passed:
        raise RuntimeError("independent P1 density/fallback QA failed")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
