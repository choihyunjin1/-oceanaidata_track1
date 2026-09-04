"""Completion-only support-contract repair for the sealed P2 copula pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _directory in (ROOT, SRC):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from scripts import (  # noqa: E402
    run_p2_gaussian_copula_conditional_mean_20260830_v1 as engine,
)

EXPERIMENT_ID = "p2_gaussian_copula_conditional_mean_20260830_v2"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
BASE_CONFIG = ROOT / "configs/experiments/p2_gaussian_copula_conditional_mean_20260830_v1.json"


def repaired_row_correction(
    query: pd.DataFrame,
    query_times: pd.DatetimeIndex,
    profile_prediction: np.ndarray,
) -> np.ndarray:
    """Predict complete 3-layer profiles and use exact no-op for partial ones."""
    time_to_row = {pd.Timestamp(value): row for row, value in enumerate(query_times)}
    layer_to_column = {
        int(layer): column for column, layer in enumerate(engine.TARGET_LAYERS)
    }
    result = np.zeros(len(query), dtype=np.float64)
    for row, value in enumerate(query.itertuples(index=False)):
        profile_row = time_to_row.get(pd.Timestamp(value.time))
        if profile_row is None:
            continue
        result[row] = profile_prediction[
            profile_row, layer_to_column[int(value.layer)]
        ]
    return result


def load_repaired_config(data_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    overlay = json.loads(CONFIG.read_text(encoding="utf-8"))
    if overlay.get("experiment_id") != EXPERIMENT_ID:
        raise engine.base.ContractError("repair experiment ID drifted")
    base_record = overlay["base_config"]
    if (
        ROOT / base_record["path"] != BASE_CONFIG
        or not BASE_CONFIG.is_file()
        or engine.base.sha256_file(BASE_CONFIG) != base_record["sha256"]
    ):
        raise engine.base.ContractError("sealed v1 base config changed")
    policy = overlay["execution_policy"]
    if any(
        (
            policy["official_hidden_gap_values_read_allowed"],
            policy["official_test_sample_submission_read_allowed"],
            policy["submission_csv_generation_allowed"],
            policy["official_upload_authorized"],
            policy["result_based_retry"],
        )
    ) or int(policy["maximum_executions"]) != 1:
        raise engine.base.ContractError("repair execution policy drifted")
    base_config = engine.load_config(data_dir)
    config = json.loads(json.dumps(base_config))
    config["experiment_id"] = EXPERIMENT_ID
    config["artifact_directory"] = overlay["artifact_directory"]
    config["report_directory"] = overlay["report_directory"]
    config["completion_only_repair"] = overlay["repair_contract"]
    return config, overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.check == args.execute:
        raise SystemExit("choose exactly one of --check or --execute")
    data_dir = args.data_dir.expanduser().resolve()
    config, overlay = load_repaired_config(data_dir)
    if args.check:
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "status": "READY_GUARDED_COMPLETION_ONLY_REPAIR",
                    "base_config_sha256": overlay["base_config"]["sha256"],
                    "model_hyperparameters_changed": False,
                    "incomplete_profile_fallback": "exact_zero_residual_correction",
                    "maximum_conceptual_copula_fits": config["resource_contract"][
                        "maximum_conceptual_copula_fits"
                    ],
                    "official_rows_read": 0,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    engine.EXPERIMENT_ID = EXPERIMENT_ID
    engine.CONFIG = CONFIG
    engine.row_correction = repaired_row_correction
    result = engine.run(config, data_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
