#!/usr/bin/env python3
"""Feasibility-corrected two-block confirmation for the sealed P2 challenger.

Only the confirmation blocks and technical availability guard differ from the
first screen.  Architecture, seed, losses, optimizer, epochs, masking, metric,
bootstrap, and science gates are inherited unchanged from the sealed engine.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

import run_p2_ts_continuous_depth_challenger_20260827_v1 as sealed


RUN_ID = "p2_ts_continuous_depth_confirmation_20260827_v2"
CONFIRMATION_FOLDS = (
    sealed.Fold("2024_may_jun", "2024-05-01", "2024-07-01"),
    sealed.Fold("2024_jul_aug", "2024-07-01", "2024-09-01"),
)
OUTPUT_DIR = (
    sealed.REPOSITORY_ROOT
    / "artifacts"
    / "structural_challenger_20260827_v1"
    / "p2_confirmation_v2"
)
MINIMUM_JOINT_TIMESTAMPS_PER_LAYER = 6000
EXPECTED_AVAILABILITY = {
    "2024_may_jun": {
        "timestamps": 8784,
        "joint_finite_target_ts_rows_by_layer": {"2": 7610, "3": 7617, "4": 7611},
    },
    "2024_jul_aug": {
        "timestamps": 8928,
        "joint_finite_target_ts_rows_by_layer": {"2": 8680, "3": 8646, "4": 8582},
    },
}
CONFIRMATION_RULE = {
    "performance_confirm": {
        "overall_temperature_rmse_delta_max": -0.001,
        "paired_day_bootstrap_90ci_upper_max": 0.0,
        "required_improved_blocks": 2,
        "minimum_improved_layers": 2,
        "worst_layer_delta_max": 0.0025,
    },
    "performance_reject": {
        "paired_day_bootstrap_90ci_lower_min": 0.0,
        "or_both_blocks_nonimproving": True,
        "or_overall_delta_min": 0.003,
        "or_worst_block_delta_min": 0.01,
        "or_worst_layer_delta_min": 0.01,
    },
    "mechanism_confirm": {
        "required_adjacent_n2_pairs_improved": 2,
        "note": "secondary; distinguishes physical confirmation from regularization-only",
    },
    "otherwise": "INCONCLUSIVE",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ["P2_DATA_DIR"]) if os.environ.get("P2_DATA_DIR") else None,
        help="Directory containing observations.csv; may also use P2_DATA_DIR.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _performance_decide(
    overall_delta: float,
    bootstrap: dict[str, Any],
    block_deltas: list[float],
    layer_deltas: list[float],
) -> tuple[str, list[str]]:
    low, high = bootstrap["ci90"]
    improved_blocks = sum(value < 0.0 for value in block_deltas)
    improved_layers = sum(value < 0.0 for value in layer_deltas)
    worst_block = max(block_deltas)
    worst_layer = max(layer_deltas)
    if (
        overall_delta <= -0.001
        and high < 0.0
        and improved_blocks == 2
        and improved_layers >= 2
        and worst_layer <= 0.0025
    ):
        return "CONFIRM", ["all preregistered out-of-screen performance guards passed"]
    rejection_reasons: list[str] = []
    if low > 0.0:
        rejection_reasons.append("paired-day 90% CI is strictly harmful")
    if improved_blocks == 0:
        rejection_reasons.append("both confirmation blocks are nonimproving")
    if overall_delta >= 0.003:
        rejection_reasons.append("overall confirmation harm is >=0.003 RMSE")
    if worst_block >= 0.01:
        rejection_reasons.append("at least one confirmation block worsened by >=0.01 RMSE")
    if worst_layer >= 0.01:
        rejection_reasons.append("at least one layer worsened by >=0.01 RMSE")
    if rejection_reasons:
        return "REJECT", rejection_reasons
    return "INCONCLUSIVE", ["confirmation and rejection guards were both incomplete"]


def _availability_preflight(data_dir: Path) -> dict[str, Any]:
    """Count only timestamps and finite joint T/S indicators; never score values."""

    observations_path = data_dir.resolve() / "observations.csv"
    frame = pd.read_csv(observations_path, usecols=["layer", "time", "temp", "psal"])
    frame["layer"] = frame["layer"].astype(int)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    joint = frame["temp"].notna() & frame["psal"].notna()
    audit: dict[str, Any] = {}
    for fold in CONFIRMATION_FOLDS:
        left = pd.Timestamp(fold.start, tz="Asia/Seoul").tz_convert("UTC")
        right = pd.Timestamp(fold.stop, tz="Asia/Seoul").tz_convert("UTC")
        expected_timestamps = int((pd.Timestamp(fold.stop) - pd.Timestamp(fold.start)).total_seconds() / 600)
        block = (frame["time"] >= left) & (frame["time"] < right)
        timestamp_count = int(frame.loc[block, "time"].nunique())
        counts = {
            str(layer): int((block & joint & frame["layer"].eq(layer)).sum())
            for layer in sealed.TARGET_LAYERS
        }
        observed = {
            "timestamps": timestamp_count,
            "expected_timestamps": expected_timestamps,
            "complete_kst_coverage": timestamp_count == expected_timestamps,
            "joint_finite_target_ts_rows_by_layer": counts,
            "minimum_required_per_layer": MINIMUM_JOINT_TIMESTAMPS_PER_LAYER,
            "eligible": (
                timestamp_count == expected_timestamps
                and min(counts.values()) >= MINIMUM_JOINT_TIMESTAMPS_PER_LAYER
            ),
        }
        expected = EXPECTED_AVAILABILITY[fold.name]
        if timestamp_count != expected["timestamps"] or counts != expected[
            "joint_finite_target_ts_rows_by_layer"
        ]:
            raise RuntimeError(f"aggregate availability drifted for {fold.name}")
        if not observed["eligible"]:
            raise RuntimeError(f"confirmation block is not technically eligible: {fold.name}")
        audit[fold.name] = observed
    return audit


def _configure_sealed_engine() -> None:
    original_config = sealed._sealed_config
    sealed.RUN_ID = RUN_ID
    sealed.FOLDS = CONFIRMATION_FOLDS
    sealed.PROMOTION_RULE = CONFIRMATION_RULE
    sealed._decide = _performance_decide

    def confirmation_config() -> dict[str, Any]:
        config = original_config()
        config["hypothesis"] = (
            "The frozen density-gradient arm repeats its row-pooled temperature advantage "
            "over the matched no-physics control on two technically scorable seasonal blocks."
        )
        config["confirmation_of"] = "p2_ts_continuous_depth_challenger_20260827_v1"
        config["technical_recovery_of"] = "p2_ts_continuous_depth_confirmation_20260827_v1"
        config["block_selection_rule"] = {
            "eligible": (
                "first-screen-disjoint two-calendar-month block with complete KST coverage "
                "and >=6000 joint finite target T/S timestamps in each of L2/L3/L4"
            ),
            "selection": (
                "chronological earliest two eligible blocks whose start-month season bins differ"
            ),
            "selected": [
                {"season_bin": "spring", "name": "2024_may_jun"},
                {"season_bin": "summer", "name": "2024_jul_aug"},
            ],
            "hidden_season_proximity_used": False,
            "first_screen_metric_outcomes_used": False,
            "expected_aggregate_availability": EXPECTED_AVAILABILITY,
        }
        config["prior_score_bias_audit"] = {
            "metric_values_read_during_v2_selection": False,
            "known_prior_scoring": (
                "Both selected block labels occur in the completed p2_m2_local_phase_v1 "
                "artifact, so prior team exposure and adaptive-history bias are nonzero."
            ),
            "selection_mitigation": (
                "Selection is mechanical chronology after a finite-count-only feasibility filter; "
                "no prior metric direction or magnitude is used."
            ),
            "residual_bias_risk": "material; confirmation is internal evidence, not pristine holdout evidence",
        }
        config["confirmation_rule"] = CONFIRMATION_RULE
        return config

    sealed._sealed_config = confirmation_config


def _postprocess_confirmation(
    output_dir: Path,
    availability: dict[str, Any],
) -> dict[str, Any]:
    result_path = output_dir / "result.json"
    manifest_path = output_dir / "manifest.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    n2_deltas = [
        float(value["physics_minus_control_rmse"])
        for value in result["density_n2_secondary_by_adjacent_pair"].values()
    ]
    mechanism_confirmed = len(n2_deltas) == 2 and all(value < 0.0 for value in n2_deltas)
    performance_decision = result["decision"]
    if performance_decision == "CONFIRM" and mechanism_confirmed:
        overall = "CONFIRM_PHYSICS"
    elif performance_decision == "CONFIRM":
        overall = "CONFIRM_PERFORMANCE_ONLY"
    elif performance_decision == "REJECT":
        overall = "REJECT"
    else:
        overall = "INCONCLUSIVE"
    result["availability_preflight"] = availability
    result["confirmation_gate"] = {
        "performance_decision": performance_decision,
        "mechanism_confirmed": mechanism_confirmed,
        "mechanism_rule": "both adjacent-pair validation N2-proxy RMSE deltas must be <0",
        "overall_confirmation": overall,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    wrapper_path = Path(__file__).resolve()
    wrapper_key = wrapper_path.relative_to(sealed.REPOSITORY_ROOT).as_posix()
    manifest["source_sha256"][wrapper_key] = sealed._sha256_file(wrapper_path)
    manifest["result_sha256"] = sealed._sha256_file(result_path)
    manifest["confirmation_wrapper_postprocess"] = {
        "aggregate_only": True,
        "raw_values_opened": False,
        "overall_confirmation": overall,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "run_id": RUN_ID,
        "performance_decision": performance_decision,
        "mechanism_confirmed": mechanism_confirmed,
        "overall_confirmation": overall,
        "result": str(result_path.relative_to(sealed.REPOSITORY_ROOT)),
    }


def main() -> None:
    args = _parse_args()
    _configure_sealed_engine()
    if not args.execute:
        print(json.dumps(sealed._sealed_config(), indent=2, sort_keys=True))
        return
    if args.data_dir is None:
        raise SystemExit("--data-dir or P2_DATA_DIR is required")
    availability = _availability_preflight(args.data_dir)
    output_dir = args.output_dir.resolve()
    sealed._execute(args)
    print(json.dumps(_postprocess_confirmation(output_dir, availability), sort_keys=True))


if __name__ == "__main__":
    main()
