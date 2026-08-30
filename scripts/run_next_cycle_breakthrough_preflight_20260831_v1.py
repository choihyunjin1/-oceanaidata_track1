from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/goals/next_cycle_breakthrough_preflight_20260831_v1.json"
DEFAULT_OUTPUT = ROOT / "reports/next_cycle_breakthrough_preflight_20260831_v1"


class ContractError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ContractError(f"expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _verify_inputs(config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    payloads: dict[str, Any] = {}
    for name, spec in config["inputs"].items():
        path = ROOT / spec["path"]
        if not path.is_file():
            raise ContractError(f"missing immutable input {name}: {path}")
        actual_bytes = path.stat().st_size
        actual_sha = _sha256(path)
        if actual_bytes != spec["bytes"] or actual_sha != spec["sha256"]:
            raise ContractError(
                f"immutable input drift for {name}: bytes={actual_bytes}, sha256={actual_sha}"
            )
        receipts[name] = {
            "path": spec["path"],
            "bytes": actual_bytes,
            "sha256": actual_sha,
        }
        payloads[name] = _load_json(path)
    return receipts, payloads


def evaluate(config: dict[str, Any], payloads: dict[str, Any]) -> dict[str, Any]:
    p1_exposure = payloads["p1_validation_audit"][
        "outer_label_exposure_and_selection_multiplicity"
    ]
    p2_exposure = payloads["p2_validation_audit"]["adaptive_exposure_audit"]
    p3_exposure = payloads["p3_validation_audit"]["adaptive_exposure"]
    gaps = payloads["promotion_gap_matrix"]["global_gaps"]
    fresh_gap = next(item for item in gaps if item["gate"] == "G2_FRESH_SURFACE")
    p2_official = payloads["p2_official_result"]
    p3_official = payloads["p3_official_result"]
    p3_era5 = payloads["p3_era5_official_receipt"]
    historical_ledger = payloads["historical_candidate_ledger"]

    checks = {
        "p1_virgin_tail_zero": p1_exposure["virgin_local_tail_rows_after_fixed_q4"] == 0,
        "p1_outer_not_independent": not p1_exposure[
            "outer_is_independent_holdout_after_exposure_history"
        ],
        "p2_repeated_generations": p2_exposure["executed_generation_lower_bound"] >= 18,
        "p2_no_fresh_surface": fresh_gap["P2"] == "absent",
        "p2_public_rule_freezes_bin17": p2_official["conclusion"]["new_champion"]
        == "P2_1_RANK1_BIN17_ONLY",
        "p2_same_cycle_forbids_adaptation": "DO_NOT_ADAPT_FURTHER_FROM_THIS_CYCLE"
        in p2_official["conclusion"]["decision"],
        "p3_repeated_same_key_oof": p3_exposure["persisted_same_key_oof_lower_bound"][
            "exact_same_key_oof_artifact_count"
        ]
        >= 10,
        "p3_no_fresh_surface": fresh_gap["P3"].startswith("absent"),
        "p3_kma_station_removals_worse": all(
            item["delta_rmse_vs_champion_m"] > 0
            for item in p3_official["candidate_reconciliation"]
        ),
        "p3_era5_official_worse": p3_era5["delta_vs_champion"]["rmse_m"] > 0,
        "historical_ledger_is_exhaustive": historical_ledger["status"]
        == "COMPLETE_EXHAUSTIVE_REAUDIT"
        and historical_ledger["coverage"]["historical_family"] == 48,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ContractError(f"preflight evidence contract failed: {failed}")

    p3_base = p3_official["axis_endpoints"]["alpha_0_base"]["official_rmse_m"]
    p3_champion = p3_official["conclusion"]["current_champion"]["official_rmse_m"]
    chart_rows = [
        {
            "candidate": "Uniform KMA 0.425",
            "candidate_rmse_m": p3_champion,
            "baseline_rmse_m": p3_base,
            "delta_rmse_m": p3_champion - p3_base,
            "baseline": "alpha=0",
            "outcome": "kept champion",
        },
        *[
            {
                "candidate": item["id"].replace("P3_1_KMA_", "").replace(
                    "P3_2_KMA_", ""
                ),
                "candidate_rmse_m": item["official_rmse_m"],
                "baseline_rmse_m": p3_champion,
                "delta_rmse_m": item["delta_rmse_vs_champion_m"],
                "baseline": "uniform KMA 0.425",
                "outcome": "worse",
            }
            for item in p3_official["candidate_reconciliation"]
        ],
        {
            "candidate": "ERA5 Hs-squared residual",
            "candidate_rmse_m": p3_era5["official_public_result"]["rmse_m"],
            "baseline_rmse_m": p3_era5["previous_public_champion"]["rmse_m"],
            "delta_rmse_m": p3_era5["delta_vs_champion"]["rmse_m"],
            "baseline": "2026-08-28 champion",
            "outcome": "worse",
        },
    ]

    decisions = {
        "P1": {
            "state": "BLOCKED_NO_FRESH_LABEL_SURFACE",
            "action": "NO_NEW_FIT",
            "reason": "The local tail is fully exposed; another historical-fold winner would add selection, not confirmation.",
            "reopen_trigger": "A newly labeled chronological block that was not used for model, feature, checkpoint, threshold, or slice selection.",
        },
        "P2": {
            "state": "HOLD_BIN17_CHAMPION_NO_SAME_PUBLIC_ADAPTATION",
            "action": "NO_NEW_FIT_OR_BIN_EXPANSION",
            "reason": "Bin17 is the Public-positive factor, bin18 reversed sign, and the same official cycle explicitly forbids further adaptation.",
            "reopen_trigger": "A presealed same-season block or a mechanistic low-rank factor selected entirely without the 2026-08-30 Public feedback.",
        },
        "P3": {
            "state": "HOLD_KMA_0425_CLOSE_KMA_AND_ERA5_EXACT_AXES",
            "action": "NO_NEW_KMA_OR_ERA5_MICROTUNE",
            "reason": "Uniform KMA improved the base, every station removal worsened it, and the champion-matched ERA5 residual also worsened its contemporary champion.",
            "reopen_trigger": "Multiple fresh episode-disjoint storms plus a materially new forcing-error target, with the correction rule frozen before those labels are opened.",
        },
    }
    return {
        "schema_version": "oceanaidata.next_cycle_breakthrough_preflight.result.v1",
        "experiment_id": config["experiment_id"],
        "decision": "NO_NEW_ONE_SHOT_AUTHORIZED_UNTIL_NEW_INFORMATION",
        "decision_kind": "FAIL_FAST_SCIENTIFIC_NO_GO",
        "checks": checks,
        "decisions": decisions,
        "evidence_summary": {
            "P1": {
                "outer_result_lower_bound": p1_exposure[
                    "outer_evaluations_or_closed_runs_with_outer_result"
                ],
                "candidate_fold_evaluation_lower_bound": p1_exposure[
                    "candidate_fold_evaluation_lower_bounds"
                ]["combined"],
                "virgin_local_tail_rows": 0,
            },
            "P2": {
                "executed_generation_lower_bound": p2_exposure[
                    "executed_generation_lower_bound"
                ],
                "same_surface_result_artifacts": p2_exposure[
                    "result_artifacts_explicitly_containing_same_three_blocks_or_69850_rows"
                ],
                "bin17_delta_rmse_vs_alpha50_c": p2_official[
                    "official_pairwise_results"
                ]["bin17_only_minus_alpha50"]["delta_rmse_c"],
                "bin18_delta_rmse_vs_alpha50_c": p2_official[
                    "official_pairwise_results"
                ]["bin18_only_minus_alpha50"]["delta_rmse_c"],
            },
            "P3": {
                "same_key_oof_artifact_lower_bound": p3_exposure[
                    "persisted_same_key_oof_lower_bound"
                ]["exact_same_key_oof_artifact_count"],
                "uniform_kma_delta_rmse_vs_base_m": p3_champion - p3_base,
                "era5_delta_rmse_vs_contemporary_champion_m": p3_era5[
                    "delta_vs_champion"
                ]["rmse_m"],
            },
        },
        "p3_official_axis_chart_rows": chart_rows,
        "outlier_policy": config["outlier_policy"],
        "literature": config["literature"],
        "operations": {
            "model_fits": 0,
            "prediction_generations": 0,
            "official_test_sample_submission_rows_read": 0,
            "public_aggregate_receipts_read": 3,
            "csv_reads": 0,
            "submission_files_created": 0,
            "uploads": 0,
        },
    }


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    receipts, payloads = _verify_inputs(config)
    result = evaluate(config, payloads)
    result["provenance"] = {
        "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": _sha256(config_path),
        "runner_path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "immutable_inputs": receipts,
    }
    _atomic_json(output_dir / "result.json", result)
    result_sha = _sha256(output_dir / "result.json")
    qa = {
        "schema_version": "oceanaidata.next_cycle_breakthrough_preflight.qa.v1",
        "status": "PASS",
        "result_sha256": result_sha,
        "checks": {
            "all_evidence_contract_checks_pass": all(result["checks"].values()),
            "all_three_actions_are_no_fit": all(
                item["action"].startswith("NO_NEW") for item in result["decisions"].values()
            ),
            "zero_model_fits": result["operations"]["model_fits"] == 0,
            "zero_official_rows": result["operations"][
                "official_test_sample_submission_rows_read"
            ]
            == 0,
            "zero_csv_and_upload": result["operations"]["csv_reads"] == 0
            and result["operations"]["uploads"] == 0,
            "automatic_outlier_removal_prohibited": result["outlier_policy"][
                "automatic_target_or_extreme_removal"
            ]
            == "PROHIBITED",
        },
    }
    if not all(qa["checks"].values()):
        raise ContractError("independent QA construction failed")
    _atomic_json(output_dir / "independent-qa.json", qa)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps({"decision": result["decision"], "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
