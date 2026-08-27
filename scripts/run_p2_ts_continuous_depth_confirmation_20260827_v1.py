#!/usr/bin/env python3
"""Two-block confirmation wrapper for the sealed P2 T/S challenger.

This wrapper deliberately reuses the first-screen implementation and all of
its architecture, optimization, seed, loss, masking, metric, and bootstrap
constants.  It changes only the predeclared confirmation blocks, output/run
identity, and confirmation decision language.  No score-driven tuning surface
is exposed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import run_p2_ts_continuous_depth_challenger_20260827_v1 as sealed


RUN_ID = "p2_ts_continuous_depth_confirmation_20260827_v1"
CONFIRMATION_FOLDS = (
    sealed.Fold("2024_mar_apr", "2024-03-01", "2024-05-01"),
    sealed.Fold("2025_jan_feb", "2025-01-01", "2025-03-01"),
)
OUTPUT_DIR = (
    sealed.REPOSITORY_ROOT
    / "artifacts"
    / "structural_challenger_20260827_v1"
    / "p2_confirmation_v1"
)
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
            "over the matched no-physics control on two out-of-screen seasonal blocks."
        )
        config["confirmation_of"] = "p2_ts_continuous_depth_challenger_20260827_v1"
        config["block_selection_rule"] = {
            "eligible": (
                "nonoverlapping two-calendar-month blocks not scored by the first screen, "
                "with complete KST coverage"
            ),
            "season_rule": (
                "cover the missing winter and spring regimes; within each regime choose "
                "the earliest complete block, independent of target scores"
            ),
            "selected": [
                {"season": "spring", "name": "2024_mar_apr"},
                {"season": "winter", "name": "2025_jan_feb"},
            ],
            "hidden_season_proximity_used": False,
            "first_screen_block_outcomes_used": False,
        }
        config["prior_score_bias_audit"] = {
            "metric_values_read_while_selecting_confirmation_blocks": False,
            "known_prior_declaration": (
                "Both names exist in src/p2_restore/research.py STABILITY_BLOCKS, so prior "
                "human exposure cannot be ruled out."
            ),
            "completed_artifact_search": (
                "No completed metric artifact containing either exact block label was found."
            ),
            "related_failed_attempt": (
                "A preexecution seal mentions a 2024-03/04 one-month trajectory pseudo-gap; "
                "the located attempts terminated before materialization and produced no score."
            ),
            "residual_bias_risk": "nonzero and explicitly retained",
        }
        config["confirmation_rule"] = CONFIRMATION_RULE
        return config

    sealed._sealed_config = confirmation_config


def _postprocess_confirmation(output_dir: Path) -> dict[str, Any]:
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
    output_dir = args.output_dir.resolve()
    sealed._execute(args)
    print(json.dumps(_postprocess_confirmation(output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
