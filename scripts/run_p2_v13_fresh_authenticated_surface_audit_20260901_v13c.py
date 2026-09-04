"""Seal the absence of a genuinely fresh authenticated P2 v13 surface."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12 as v12  # noqa: E402

EXPERIMENT_ID = "p2_v13_fresh_authenticated_surface_audit_20260901_v13c"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID:
        raise v12.ContractError("experiment ID drift")
    if config["status"] != "PREREGISTERED_ZERO_FIT_NOT_EXECUTED":
        raise v12.ContractError("audit is not preregistered")
    limits = config["operation_limits"]
    if any(int(value) != 0 for value in limits.values()):
        raise v12.ContractError("zero-operation audit contract drift")
    if config["source_contract"]["fresh_authenticated_scoring_frames"]:
        raise v12.ContractError("fresh authenticated surface appeared after sealing")
    return config


def _verify_hash(path: Path, expected: str) -> str:
    observed = v12.sha256_file(path)
    if observed != expected:
        raise v12.ContractError(f"hash drift: {path}")
    return observed


def month_coverage(times: pd.Series) -> list[str]:
    parsed = pd.to_datetime(times, utc=True)
    return sorted(parsed.dt.strftime("%Y-%m").unique().tolist())


def preflight() -> dict[str, Any]:
    config = load_config()
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "decision": config["decision"],
        "fresh_authenticated_scoring_frames": 0,
        "model_fits": 0,
        "candidate_predictions": 0,
        "official_rows_read": 0,
        "hidden_rows_read": 0,
        "submission_csv_created": 0,
        "uploads": 0,
        "namespace_fresh": not ARTIFACT.exists(),
        "config_sha256": v12.sha256_file(CONFIG),
        "runner_sha256": v12.sha256_file(RUNNER),
    }
    payload["preflight_sha256"] = v12.sha256_json(payload)
    return payload


def run() -> dict[str, Any]:
    if ARTIFACT.exists():
        raise FileExistsError(ARTIFACT)
    config = load_config()
    candidate = config["candidate_commitment"]
    authority = config["freshness_authority"]
    source = config["source_contract"]
    verified = {
        "v13_result": _verify_hash(ROOT / candidate["result"], candidate["result_sha256"]),
        "v13_independent_qa": _verify_hash(
            ROOT / candidate["independent_qa"], candidate["independent_qa_sha256"]
        ),
        "authority_report": _verify_hash(ROOT / authority["report"], authority["report_sha256"]),
        "authority_gap_matrix": _verify_hash(
            ROOT / authority["gap_matrix"], authority["gap_matrix_sha256"]
        ),
        "authority_independent_qa": _verify_hash(
            ROOT / authority["independent_qa"], authority["independent_qa_sha256"]
        ),
        "existing_exposed_scoring_frame": _verify_hash(
            ROOT / source["existing_exposed_scoring_frame"],
            source["existing_exposed_scoring_frame_sha256"],
        ),
    }
    authority_qa = json.loads((ROOT / authority["independent_qa"]).read_text(encoding="utf-8"))
    if authority_qa.get("decisions", {}).get("P2") != authority["required_decision"]:
        raise v12.ContractError("freshness authority decision drift")
    v13_result = json.loads((ROOT / candidate["result"]).read_text(encoding="utf-8"))
    v13_qa = json.loads((ROOT / candidate["independent_qa"]).read_text(encoding="utf-8"))
    if v13_result["status"] != "EXPLORATORY_SAFETY_PASS_REQUIRES_FRESH_CONFIRMATION":
        raise v12.ContractError("v13 terminal status drift")
    if v13_qa.get("status") != "PASS":
        raise v12.ContractError("v13 independent QA is not PASS")

    data_dir = os.environ.get(source["environment_variable"])
    if not data_dir:
        raise v12.ContractError("P2_DATA_DIR is required")
    observations_path = Path(data_dir) / source["only_source_filename"]
    _verify_hash(observations_path, source["observations_sha256"])
    observations_meta = pd.read_csv(observations_path, usecols=["time", "layer"])
    exposed = pd.read_parquet(
        ROOT / source["existing_exposed_scoring_frame"], columns=["time", "layer", "fold"]
    )

    ARTIFACT.mkdir(parents=True)
    v12.atomic_json(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": v12.sha256_file(CONFIG),
            "runner_sha256": v12.sha256_file(RUNNER),
            "model_fits": 0,
            "candidate_predictions": 0,
        },
    )
    result = {
        "schema_version": "p2.v13_fresh_authenticated_surface_audit.result.20260901.v13c",
        "experiment_id": EXPERIMENT_ID,
        "status": "HOLD_NO_FRESH_AUTHENTICATED_SURFACE",
        "model_fits": 0,
        "candidate_predictions": 0,
        "finding": {
            "observations_calendar_months": month_coverage(observations_meta["time"]),
            "observations_rows": int(len(observations_meta)),
            "existing_exposed_calendar_months": month_coverage(exposed["time"]),
            "existing_exposed_rows": int(len(exposed)),
            "existing_exposed_folds": sorted(exposed["fold"].astype(str).unique().tolist()),
            "fresh_authenticated_scoring_frame_count": 0,
            "raw_labels_do_not_mint_freshness": True,
            "post_v13_repartition_forbidden": True,
            "fresh_confirmation_executed": False,
        },
        "decision": {
            "v13_commitment_preserved": True,
            "v13_posthoc_routing": False,
            "next_action": "NEW_NONDUPLICATE_REPRESENTATION_ONLY",
            "performance_claim_allowed": False,
        },
        "operation_counters": {
            "observations_metadata_rows_read": int(len(observations_meta)),
            "existing_exposed_scoring_metadata_rows_read": int(len(exposed)),
            "target_value_rows_read": 0,
            "model_fits": 0,
            "candidate_predictions": 0,
            "official_rows_read": 0,
            "hidden_rows_read": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
        "hashes": {
            "config": v12.sha256_file(CONFIG),
            "runner": v12.sha256_file(RUNNER),
            "observations": v12.sha256_file(observations_path),
            **verified,
        },
    }
    v12.atomic_json(ARTIFACT / "result.json", result)
    REPORT.mkdir(parents=True, exist_ok=True)
    v12.atomic_json(REPORT / "result.json", result)
    (REPORT / "report-source.md").write_text(
        "# P2 v13 fresh authenticated surface audit\n\n"
        "## 결론\n\n"
        "`HOLD_NO_FRESH_AUTHENTICATED_SURFACE`. observations에 다른 달의 원시 label이 "
        "존재해도 v13 개발 전에 blind seal된 comparator/scoring surface가 아니므로 fresh가 아니다. "
        "기존 authenticated scoring frame은 이미 노출된 3개 fold 69,850행뿐이다. 따라서 임의 재분할, "
        "v13 재학습, 성능 주장은 0이고, 다음 행동은 독립 representation 연구뿐이다.\n\n"
        "official/hidden/CSV/upload=0, target-value read=0, model fit=0.\n",
        encoding="utf-8",
    )
    (REPORT / "claim-source-ledger.md").write_text(
        "# Claim-source ledger\n\n"
        "| Claim | Source | Scope |\n|---|---|---|\n"
        "| P2 has no untouched same-season 61-day surface | `reports/next_action_meta_deep_research_20260831_v1/gap-matrix.md` | authenticated local meta-audit |\n"
        "| v13 needs fresh confirmation | `reports/p2_prefix_safe_domain_balanced_deepset_20260901_v13/result.json` | immutable terminal receipt |\n"
        "| current comparator covers only three exposed folds | `artifacts/p2_alpha50_supervised_rank1_threeway_crossfit_regime_veto_20260828_v2/scored_predictions_no_truth.parquet` | keys/fold metadata only |\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.execute:
        raise SystemExit("Choose exactly one of --preflight/--execute")
    payload = preflight() if args.preflight else run()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
