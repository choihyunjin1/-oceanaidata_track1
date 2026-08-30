"""Independent QA for the loader-repaired frozen P2 Gaussian-copula v2 pack."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from p2_restore.p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v1 import (  # noqa: E402
    duplicate_receipt,
    ensure_external_output_dir,
    load_json,
    read_keyed_csv,
    sha256_file,
    validate_query_sources,
)
from p2_restore.profile_projection import (  # noqa: E402
    project_profiles_vectorized,
    public_endpoint_frame,
)
from p2_restore.submission import validate_submission  # noqa: E402
from scripts import (  # noqa: E402
    run_p2_alpha50_supervised_rank1_functional_residual_20260828_v1 as base,
)
from scripts.run_p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v2 import (  # noqa: E402
    DEFAULT_BASE_CONFIG,
    DEFAULT_CONFIG,
    DEFAULT_REPORT_DIR,
    EXPECTED_BASE_CONFIG_SHA256,
    EXPECTED_OVERLAY_CONFIG_SHA256,
    EXPERIMENT_ID,
    validate_overlay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--p2-dir", type=Path, required=True)
    parser.add_argument("--base-u", type=Path, required=True)
    parser.add_argument("--alpha50-reference", type=Path, required=True)
    parser.add_argument("--official-incumbent", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _delta_receipt(candidate: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    delta = candidate - reference
    return {
        "changed_rows": int(np.sum(np.abs(delta) > 1e-12)),
        "rms_c": float(np.sqrt(np.mean(np.square(delta)))),
        "p99_abs_c": float(np.quantile(np.abs(delta), 0.99)),
        "maximum_abs_c": float(np.max(np.abs(delta))),
    }


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if repo_root != REPO.resolve():
        raise RuntimeError("repo-root does not match QA repository")
    overlay_path = args.config.resolve()
    base_config_path = args.base_config.resolve()
    report_dir = args.report_dir.resolve()
    output_dir = ensure_external_output_dir(repo_root, args.output_dir)
    qa_path = report_dir / "independent-qa.json"
    result_path = report_dir / "result.json"
    lock_path = report_dir / "attempt.lock.json"
    candidate_path = output_dir / "P2_submission.csv"

    overlay = load_json(overlay_path)
    base_config = load_json(base_config_path)
    result = load_json(result_path)
    lock = load_json(lock_path)
    overlay_sha = sha256_file(overlay_path)
    base_config_sha = sha256_file(base_config_path)
    overlay_receipts = validate_overlay(repo_root, overlay, base_config)
    candidate_sha = sha256_file(candidate_path)
    duplicate = duplicate_receipt(candidate_sha, base_config)

    p2_dir = args.p2_dir.resolve()
    test = read_keyed_csv(p2_dir / "test_index.csv", require_temp=False)
    sample = read_keyed_csv(p2_dir / "sample_submission.csv", require_temp=True)
    base_u = read_keyed_csv(args.base_u.resolve(), require_temp=True)
    alpha50 = read_keyed_csv(args.alpha50_reference.resolve(), require_temp=True)
    incumbent = read_keyed_csv(args.official_incumbent.resolve(), require_temp=True)
    query_contract = validate_query_sources(
        test, sample, base_u, alpha50, incumbent, base_config
    )
    candidate = read_keyed_csv(candidate_path, require_temp=True)
    strict_validation = validate_submission(candidate, test)

    candidate_values = candidate["temp"].to_numpy(np.float64)
    reference_values = alpha50["temp"].to_numpy(np.float64)
    incumbent_values = incumbent["temp"].to_numpy(np.float64)
    base_values = base_u["temp"].to_numpy(np.float64)
    profile_counts = test.groupby("time", sort=False)["layer"].transform("nunique")
    partial = profile_counts.lt(3).to_numpy()
    partial_noop_max = float(
        np.max(np.abs(candidate_values[partial] - reference_values[partial]), initial=0.0)
    )

    observations = base.read_observations(p2_dir / "observations.csv")
    endpoints = public_endpoint_frame(observations)
    projected = project_profiles_vectorized(test, candidate_values, endpoints).prediction
    pava_idempotence = float(np.max(np.abs(projected - candidate_values), initial=0.0))

    note = (output_dir / "제출정보.txt").read_text(encoding="utf-8-sig")
    recomputed = {
        "vs_alpha50": _delta_receipt(candidate_values, reference_values),
        "vs_current_official_incumbent": _delta_receipt(
            candidate_values, incumbent_values
        ),
        "vs_official_base_u": _delta_receipt(candidate_values, base_values),
    }
    stored_prediction = result["materialization"]["prediction"]
    checks = {
        "overlay_hash": overlay_sha == EXPECTED_OVERLAY_CONFIG_SHA256
        and result["overlay_config_sha256"] == overlay_sha,
        "base_scientific_config_hash": base_config_sha
        == EXPECTED_BASE_CONFIG_SHA256
        and result["base_scientific_config_sha256"] == base_config_sha,
        "overlay_pins_and_scope": overlay_receipts == result["overlay_receipts"]
        and result["repair"] == overlay["repair"],
        "attempt_one_of_one": lock["attempt_number"] == 1
        and lock["maximum_executions"] == 1,
        "attempt_hashes": lock["overlay_config_sha256"] == overlay_sha
        and lock["base_scientific_config_sha256"] == base_config_sha,
        "candidate_hash": candidate_sha == result["output"]["sha256"],
        "candidate_bytes": candidate_path.stat().st_size == result["output"]["bytes"],
        "candidate_external_to_repo": not candidate_path.is_relative_to(repo_root),
        "query_contract": query_contract == result["materialization"]["query_contract"],
        "strict_schema_key_order_finite_domain": strict_validation
        == result["materialization"]["validation"],
        "sample_key_order": sample[["station", "layer", "time"]].equals(
            test[["station", "layer", "time"]]
        ),
        "incomplete_profile_exact_noop": partial_noop_max <= 1e-12
        and partial_noop_max
        == stored_prediction["incomplete_profile_max_abs_correction_c"],
        "profile_projection_idempotent": pava_idempotence <= 1e-12,
        "difference_receipts": recomputed["vs_alpha50"]
        == stored_prediction["difference_vs_alpha50"]
        and recomputed["vs_current_official_incumbent"]
        == stored_prediction["difference_vs_current_official_incumbent"]
        and recomputed["vs_official_base_u"]
        == stored_prediction["difference_vs_official_base_u"],
        "model_receipt_exact_historical_match": result["materialization"]["fit"]
        ["model_receipt_sha256"]
        == base_config["frozen_recipe"]["expected_refit_model_receipt_sha256"],
        "one_fit_zero_search": result["execution_receipt"]["copula_fits"] == 1
        and result["execution_receipt"]["inner_search_or_hpo"] == 0,
        "no_exact_prior_submission_hash": not duplicate["exact_hash_duplicate"]
        and duplicate["status"] == "PASS_NO_EXACT_HASH_DUPLICATE",
        "recent_live_history_included": len(duplicate["recent_official_history"])
        == 3
        and duplicate["current_best_public_rmse_c"] == 0.430209,
        "note_title_summary_hash": base_config["title"] in note
        and base_config["one_line_summary"] in note
        and candidate_sha in note,
        "hidden_truth_score_baseline_upload_zero": result["execution_receipt"]
        ["hidden_truth_rows_read"]
        == 0
        and result["execution_receipt"]["score_py_reads"] == 0
        and result["execution_receipt"]["baseline_reads"] == 0
        and result["execution_receipt"]["uploads"] == 0,
    }
    qa = {
        "schema_version": (
            "p2.gaussian_copula_v2.exact_frozen_submission_pack.qa.20260830.v2"
        ),
        "experiment_id": EXPERIMENT_ID,
        "qa_status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "candidate": {
            "path": str(candidate_path),
            "bytes": candidate_path.stat().st_size,
            "sha256": candidate_sha,
            "validation": strict_validation,
            "layer_rows": query_contract["layer_rows"],
            "partial_profile_noop_max_abs_c": partial_noop_max,
            "pava_idempotence_max_abs_c": pava_idempotence,
            "differences": recomputed,
        },
        "duplicate_qa": duplicate,
        "hashes": {
            "overlay_config": overlay_sha,
            "base_scientific_config": base_config_sha,
            "attempt_lock": sha256_file(lock_path),
            "result": sha256_file(result_path),
        },
        "independent_execution_receipt": {
            "model_fits": 0,
            "hidden_truth_rows_read": 0,
            "score_py_reads": 0,
            "baseline_reads": 0,
            "csv_created": 0,
            "uploads": 0,
        },
        "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
    }
    _write_json_exclusive(qa_path, qa)
    print(json.dumps(qa, ensure_ascii=False, sort_keys=True))
    return 0 if qa["qa_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
