from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_p3_target_shift_retroaudit_20260828_v1 import (
    KEYS,
    biased_mmd_permutation,
    candidate_metrics,
    cross_fitted_domain_weights,
    load_candidate,
    nearest_neighbor_weights,
    sha256_file,
)


def build_champion_rows(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    replay = pd.read_parquet(config["champion_replay"])
    cases = replay[["anchor_id", "station"]].drop_duplicates()
    anchors = pd.read_parquet(config["anchor_metadata"])
    targets = anchors.melt(
        id_vars=["anchor_id", "station", "current_hs"],
        value_vars=["target_3", "target_6", "target_9", "target_12", "target_18", "target_24"],
        var_name="lead_name",
        value_name="target_hs",
    )
    targets["lead_h"] = targets["lead_name"].str.replace("target_", "", regex=False).astype(int)
    rows = replay[KEYS + ["champion_prediction"]].merge(
        targets[KEYS + ["target_hs", "current_hs"]], on=KEYS, how="left", validate="one_to_one"
    )
    return cases, rows


def evaluate_surface(
    name: str,
    source: pd.DataFrame,
    target: pd.DataFrame,
    feature_columns: list[str],
    champion_rows: pd.DataFrame,
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    weights, domain, matrix = cross_fitted_domain_weights(
        source, target, feature_columns, config["domain_classifier"]
    )
    nn_weight, nn_diagnostics = nearest_neighbor_weights(matrix, len(source))
    weights["nearest_neighbor_weight"] = nn_weight
    surface_rows = champion_rows.merge(
        source[["anchor_id", "station"]], on=["anchor_id", "station"], how="inner", validate="many_to_one"
    )
    metrics: dict[str, Any] = {}
    persistence = surface_rows[KEYS + ["current_hs"]].rename(columns={"current_hs": "candidate_prediction"})
    metrics["persistence"] = candidate_metrics(
        surface_rows, persistence, weights, config["bootstrap"]
    )
    for candidate_spec in candidates:
        metrics[candidate_spec["name"]] = candidate_metrics(
            surface_rows, load_candidate(candidate_spec), weights, config["bootstrap"]
        )
    return (
        {
            "surface": name,
            "source_cases": int(len(source)),
            "target_cases": int(len(target)),
            "feature_columns": feature_columns,
            "domain_classifier": domain,
            "nearest_neighbor": nn_diagnostics,
            "mmd": biased_mmd_permutation(matrix, len(source), config["mmd"]),
            "candidate_metrics": metrics,
        },
        weights.assign(surface=name),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/p3_target_shift_retroaudit_20260828_v2.json",
    )
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    candidate_config = json.loads(Path(config["candidate_config"]).read_text(encoding="utf-8"))
    candidates = candidate_config["candidates"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cases, champion_rows = build_champion_rows(config)
    all_columns = config["wave_state_columns"] + config["wind_columns"]
    train = pd.read_parquet(
        config["source_features"], columns=["anchor_id", "station"] + all_columns
    )
    test = pd.read_parquet(
        config["target_features"], columns=["case_id", "station"] + all_columns
    )
    source = cases.merge(train, on=["anchor_id", "station"], how="left", validate="one_to_one")
    target = test.copy()

    wave_surface, wave_weights = evaluate_surface(
        "wave_state_all_181",
        source,
        target,
        config["wave_state_columns"],
        champion_rows,
        candidates,
        config,
    )

    complete_mask = source[config["wind_columns"]].notna().all(axis=1)
    complete_source = source.loc[complete_mask].reset_index(drop=True)
    complete_surface, complete_weights = evaluate_surface(
        "complete_wind_source",
        complete_source,
        target,
        all_columns,
        champion_rows,
        candidates,
        config,
    )

    weights = pd.concat([wave_weights, complete_weights], ignore_index=True)
    weights.to_parquet(output_dir / "surface_case_weights.parquet", index=False)
    result = {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "status": "COMPLETED_RESEARCH_ONLY_NO_SUBMISSION",
        "parent_result": config["parent_result"],
        "historical_atmosphere_coverage": {
            "all_champion_cases": int(len(source)),
            "complete_wind_cases": int(len(complete_source)),
            "complete_wind_fraction": float(len(complete_source) / len(source)),
            "by_station": {
                str(station): {
                    "all": int(len(group)),
                    "complete": int(group[config["wind_columns"]].notna().all(axis=1).sum()),
                }
                for station, group in source.groupby("station", sort=True)
            },
        },
        "surfaces": {
            "wave_state_all_181": wave_surface,
            "complete_wind_source": complete_surface,
        },
        "decision_contract": {
            "purpose": "separate wave-state selection shift from historical atmosphere-coverage shift",
            "known_official_probe": "era5_hs2_energy_residual failed official transport despite local and weighted-local improvement",
            "promotion_prohibited": True,
            "next_structure_only_if_supported": "local direct residual expert trained and validated on atmosphere-complete periods with cadence-matched features",
        },
        "scope_guards": config["scope_guards"],
        "inputs": {
            "runner": {"path": str(Path(__file__)).replace("\\", "/"), "sha256": sha256_file(Path(__file__))},
            "config": {"path": str(config_path).replace("\\", "/"), "sha256": sha256_file(config_path)},
            "candidate_config": {
                "path": config["candidate_config"],
                "sha256": sha256_file(Path(config["candidate_config"])),
            },
            "source_features": {
                "path": config["source_features"],
                "sha256": sha256_file(Path(config["source_features"])),
            },
            "target_features": {
                "path": config["target_features"],
                "sha256": sha256_file(Path(config["target_features"])),
            },
        },
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
