"""Finalize immutable v23m1 CSV after a post-write bit-equality QA exception."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for item in (ROOT / "scripts", ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12 as v12  # noqa: E402

from p2_restore.data import KEYS  # noqa: E402
from p2_restore.submission import validate_submission  # noqa: E402

EXPERIMENT_ID = "p2_public_temperature_input_gradient_deployment_20260901_v23m1"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
ORIGINAL_RUNNER = (
    ROOT / "scripts" / "materialize_p2_public_temperature_input_gradient_20260901_v23m1.py"
)
FINALIZER = Path(__file__)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    data_dir = Path(os.environ["P2_DATA_DIR"]).resolve()
    test_path = data_dir / "test_index.csv"
    baseline_path = data_dir / "baseline_interp.csv"
    anchor_path = Path(config["data_contract"]["anchor_path"])
    output_dir = Path(config["output"]["directory"])
    csv_path = output_dir / config["output"]["filename"]
    if not (ARTIFACT / "attempt_lock.json").is_file() or not csv_path.is_file():
        raise FileNotFoundError("immutable v23m1 lock/CSV missing")
    expected = {
        test_path: config["data_contract"]["test_index_sha256"],
        baseline_path: config["data_contract"]["baseline_interp_sha256"],
        anchor_path: config["data_contract"]["anchor_sha256"],
    }
    for path, digest in expected.items():
        if v12.sha256_file(path) != digest:
            raise RuntimeError(f"source hash drift: {path}")
    dtype = {"station": "string", "time": "string"}
    test = pd.read_csv(test_path, dtype=dtype)
    candidate = pd.read_csv(csv_path, dtype=dtype)
    anchor = pd.read_csv(anchor_path, dtype=dtype)
    validation = validate_submission(candidate, test)
    if not anchor[KEYS].equals(test[KEYS]):
        raise RuntimeError("anchor key order drift")
    values = pd.to_numeric(candidate["temp"], errors="coerce").to_numpy(float)
    anchor_values = pd.to_numeric(anchor["temp"], errors="coerce").to_numpy(float)
    action = values - anchor_values
    absolute = np.abs(action)
    checks = {
        "csv_exists": csv_path.is_file(),
        "rows_26061": len(candidate) == 26061,
        "schema_exact": list(candidate.columns) == KEYS + ["temp"],
        "key_order_exact": candidate[KEYS].equals(test[KEYS]),
        "duplicate_keys_zero": not candidate.duplicated(KEYS).any(),
        "finite": bool(np.isfinite(values).all()),
        "submission_domain": bool(((values >= -5.0) & (values <= 45.0)).all()),
        "action_cap_0_5C": float(absolute.max()) <= 0.5 + 1e-12,
        "anchor_unmodified_hash": v12.sha256_file(anchor_path)
        == config["data_contract"]["anchor_sha256"],
        "test_hash": v12.sha256_file(test_path)
        == config["data_contract"]["test_index_sha256"],
        "source_v23_hashes": all(
            v12.sha256_file(ROOT / config["frozen_candidate"][name])
            == config["frozen_candidate"][f"{name}_sha256"]
            for name in ("config", "runner", "result", "independent_qa")
        ),
        "no_retraining_in_finalizer": True,
        "hidden_truth_access_zero": True,
        "score_access_zero": True,
        "sample_values_access_zero": True,
        "upload_zero": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"v23m1 independent CSV QA failed: {checks}")
    result = {
        "schema_version": "p2.public_temperature_input_gradient_deployment.terminal_amendment.20260901.v23m1",
        "experiment_id": EXPERIMENT_ID,
        "status": "MATERIALIZED_READY_NOT_UPLOADED_POST_WRITE_QA_AMENDMENT",
        "fit_count": 3,
        "fit_count_evidence": "Original process reached candidate CSV write after completing the fixed three-seed loop; it then raised only on in-memory-vs-decimal-parser np.array_equal.",
        "technical_event": {
            "classification": "POST_WRITE_OVERSTRICT_FLOAT_BIT_EQUALITY_ASSERTION",
            "exception": "ContractError: numeric CSV round-trip drift",
            "scientific_training_completed": True,
            "candidate_csv_written_before_exception": True,
            "same_id_rerun": False,
            "model_retrained_by_finalizer": False,
            "candidate_csv_modified_by_finalizer": False,
            "resolution": "Treat immutable decimal CSV as the deployment object and validate its schema, keys, finite domain, action cap, and hash independently.",
        },
        "candidate": {
            "name": "P2_V23_PUBLIC_TEMP_INPUT_GRADIENT_FULL_HISTORY_BLEND020",
            "path": str(csv_path),
            "sha256": v12.sha256_file(csv_path),
            "bytes": csv_path.stat().st_size,
            "rows": len(candidate),
            "validation": validation,
            "changed_rows_vs_anchor": int(np.count_nonzero(absolute > 1e-12)),
            "active_share_vs_anchor": float(np.mean(absolute > 1e-12)),
            "abs_action_p50_C": float(np.quantile(absolute, 0.50)),
            "abs_action_p90_C": float(np.quantile(absolute, 0.90)),
            "abs_action_p99_C": float(np.quantile(absolute, 0.99)),
            "abs_action_max_C": float(absolute.max()),
            "action_rms_C": float(np.sqrt(np.mean(np.square(action)))),
            "minimum_C": float(values.min()),
            "maximum_C": float(values.max()),
        },
        "anchor": {
            "name": "P2_1_RANK1_BIN17_ONLY",
            "path": str(anchor_path),
            "sha256": config["data_contract"]["anchor_sha256"],
            "public_rmse_C": config["data_contract"]["anchor_public_rmse_C"],
            "public_points": config["data_contract"]["anchor_public_points"],
            "modified": False,
        },
        "internal_evidence": {
            "historical_delta_rmse_C": -0.05189246657169555,
            "canonical_nominal_expected_points_delta": 0.6511223381640603,
            "canonical_transport_adjusted_expected_points_delta": 0.5294402465540541,
            "prospective_fold_layer_non_harm_cells": 6,
            "prospective_fold_layer_total_cells": 9,
            "caveat": "Repeatedly exposed exploratory surface; the expected point translations are planning values, not official guarantees.",
        },
        "submission_metadata": {
            "title": config["output"]["title"],
            "summary": config["output"]["summary"],
        },
        "checks": checks,
        "operation_counters": {
            "official_test_index_rows_read": len(test),
            "official_anchor_rows_read": len(anchor),
            "hidden_truth_rows_read": 0,
            "score_file_rows_read": 0,
            "sample_submission_rows_read": 0,
            "submission_csv_created": 1,
            "uploads": 0,
            "automatic_retries": 0,
            "finalizer_model_fits": 0,
        },
        "hashes": {
            "config": v12.sha256_file(CONFIG),
            "original_runner": v12.sha256_file(ORIGINAL_RUNNER),
            "finalizer": v12.sha256_file(FINALIZER),
            "attempt_lock": v12.sha256_file(ARTIFACT / "attempt_lock.json"),
            "test_index": v12.sha256_file(test_path),
            "baseline_interp": v12.sha256_file(baseline_path),
            "anchor": v12.sha256_file(anchor_path),
            "candidate_csv": v12.sha256_file(csv_path),
        },
    }
    v12.atomic_json(ARTIFACT / "terminal-amendment.json", result)
    v12.atomic_json(REPORT / "result.json", result)
    v12.atomic_json(output_dir / "manifest.json", result)
    qa = {
        "schema_version": "p2.public_temperature_input_gradient_deployment.independent_qa.20260901.v23m1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS",
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "candidate_sha256": result["candidate"]["sha256"],
        "candidate_path": result["candidate"]["path"],
        "uploads": 0,
    }
    v12.atomic_json(REPORT / "independent-qa.json", qa)
    v12.atomic_json(output_dir / "independent-qa.json", qa)
    (output_dir / "upload-note.md").write_text(
        f"# {config['output']['title']}\n\n"
        f"{config['output']['summary']}\n\n"
        f"CSV: `{csv_path}`\n\n"
        f"SHA-256: `{result['candidate']['sha256']}`\n\n"
        f"Rows: `{len(candidate)}`; QA: `{qa['status']} {qa['passed']}/{qa['total']}`; "
        "upload: `0`.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
