"""Run the append-only key-aligned P2 dynamic sigmoid precheck generation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from p2_restore.dynamic_sigmoid_key_alignment import (
    key_aligned_gate_1,
    load_key_only_population,
    sha256,
)
from p2_restore.dynamic_sigmoid_profile import build_public_features, feature_columns

EXPERIMENT_ID = "p2_dynamic_sigmoid_profile_v2_key_aligned"
ROOT = Path(__file__).resolve().parents[1]
V1_RUNNER_PATH = ROOT / "scripts/run_p2_dynamic_sigmoid_profile.py"


def _load_v1_runner() -> Any:
    spec = importlib.util.spec_from_file_location("_p2_dynamic_sigmoid_v1_runner", V1_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the pinned v1 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXPERIMENT_ID = EXPERIMENT_ID
    return module


V1 = _load_v1_runner()


def _load_contract(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected v2 experiment id")
    if value.get("status") != "preregistered_precheck_only":
        raise ValueError("v2 is not precheck-only")
    if value.get("research_only") is not True or value.get("upload_allowed") is not False:
        raise ValueError("v2 must remain local research-only")
    generation = value["generation"]
    parent_path = (ROOT / generation["parent_config_path"]).resolve()
    diagnostic_path = (ROOT / generation["denominator_diagnostic_path"]).resolve()
    if sha256(parent_path) != generation["parent_config_sha256"]:
        raise ValueError("pinned v1 config SHA differs")
    if sha256(diagnostic_path) != generation["denominator_diagnostic_sha256"]:
        raise ValueError("pinned denominator diagnostic SHA differs")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    unchanged = (
        "objective",
        "problem_contract",
        "validation",
        "sigmoid",
        "public_features",
        "gates",
        "incumbent",
        "provenance",
    )
    differing = [section for section in unchanged if value[section] != parent[section]]
    if differing:
        raise ValueError(f"v2 changed frozen v1 sections: {differing}")
    gate_population = value["gate_1_population"]
    expected = {
        "columns_allowed": ["time", "layer"],
        "unique_time_denominator": True,
        "truth_read": False,
        "prediction_read": False,
        "block_read": False,
        "support_definition_unchanged_from_v1": True,
        "threshold_unchanged_from_v1": True,
    }
    if any(gate_population.get(key) != current for key, current in expected.items()):
        raise ValueError("key-aligned Gate 1 contract is not fail-closed")
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if diagnostic["decision"]["v1_failure_is_denominator_artifact"] is not True:
        raise ValueError("pinned audit does not authorize the denominator correction")
    if diagnostic["decision"]["all_key_aligned_support_shares_at_least_0_8"] is not True:
        raise ValueError("pinned audit does not pass every corrected Gate 1")
    return value, parent


def _source_hashes(
    *, config_path: Path, contract: dict[str, Any], observation_path: Path
) -> dict[str, object]:
    paths = {
        "v2_config": config_path,
        "v2_helper": ROOT / "src/p2_restore/dynamic_sigmoid_key_alignment.py",
        "v2_runner": ROOT / "scripts/run_p2_dynamic_sigmoid_profile_v2_key_aligned.py",
        "v2_tests": ROOT / "tests/test_p2_dynamic_sigmoid_profile_v2_key_aligned.py",
        "v1_config": ROOT / contract["generation"]["parent_config_path"],
        "v1_helper": ROOT / "src/p2_restore/dynamic_sigmoid_profile.py",
        "v1_runner": V1_RUNNER_PATH,
        "denominator_diagnostic": ROOT / contract["generation"]["denominator_diagnostic_path"],
        "observations": observation_path,
        "incumbent_oof": ROOT / contract["incumbent"]["path"],
    }
    return {
        name: {"sha256": sha256(path), "bytes": path.stat().st_size} for name, path in paths.items()
    }


def _base_receipt(
    *,
    config_path: Path,
    contract: dict[str, Any],
    observation_path: Path,
    mode: str,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "created_at": datetime.now().astimezone().isoformat(),
        "research_only": True,
        "adaptive_after_prior_outer_exposure": True,
        "fresh_holdout_claimed": False,
        "aggregate_only": True,
        "external_values_used": False,
        "hidden_target_values_read": False,
        "test_index_values_read": False,
        "outer_predictions_created": False,
        "outer_truth_scored": False,
        "submission_created": False,
        "upload_attempted": False,
        "generation": contract["generation"],
        "provenance": {
            **V1._git_provenance(ROOT),
            "python": sys.version.split()[0],
            "files": _source_hashes(
                config_path=config_path,
                contract=contract,
                observation_path=observation_path,
            ),
            "literature": contract["provenance"]["primary_literature"],
        },
    }


def _key_population(contract: dict[str, Any]) -> Any:
    incumbent = contract["incumbent"]
    return load_key_only_population(
        (ROOT / incumbent["path"]).resolve(), expected_sha256=incumbent["sha256"]
    )


def _correct_gate_1(
    *, public_features: Any, key_population: Any, block: Any, contract: dict[str, Any]
) -> dict[str, object]:
    sigmoid = contract["sigmoid"]
    threshold = float(contract["gates"]["gate_1_public_support"]["minimum_share"])
    return key_aligned_gate_1(
        public_features,
        key_population,
        block,
        minimum_public_points=int(sigmoid["minimum_public_points"]),
        minimum_depth_span_m=float(sigmoid["minimum_public_depth_span_m"]),
        threshold=threshold,
    )


def _dry_run(
    *,
    config_path: Path,
    contract: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    status_path: Path,
    started: float,
) -> dict[str, object]:
    observations = V1._read_observations(data_dir)
    hidden = V1._audit_hidden(observations, contract)
    features = build_public_features(observations)
    keys = _key_population(contract)
    blocks = V1._blocks(contract)
    gate_1 = {
        block.name: _correct_gate_1(
            public_features=features,
            key_population=keys,
            block=block,
            contract=contract,
        )
        for block in blocks
    }
    receipt = {
        **_base_receipt(
            config_path=config_path,
            contract=contract,
            observation_path=data_dir / "observations.csv",
            mode="dry-run",
        ),
        "status": "dry_run_pass",
        "data_contract": {
            "observation_rows": len(observations),
            **hidden,
            "incumbent_key_rows": len(keys),
            "public_feature_rows": len(features),
        },
        "gate_1_key_only_dry": gate_1,
        "forbidden_operations": {
            "oof_truth_reads": 0,
            "oof_prediction_reads": 0,
            "model_fits": 0,
            "alpha_computations": 0,
            "gate5_runs": 0,
            "outer_scores": 0,
            "test_index_reads": 0,
            "submission_rows": 0,
            "upload_attempts": 0,
        },
    }
    V1._write_json(output_dir / "dry_run.json", receipt)
    V1._status(
        status_path,
        started=started,
        progress=100.0,
        phase="complete",
        detail="v2 key-only dry-run PASS; no model or prediction opened",
        state="complete",
    )
    return receipt


def _precheck(
    *,
    config_path: Path,
    contract: dict[str, Any],
    data_dir: Path,
    output_dir: Path,
    status_path: Path,
    started: float,
) -> dict[str, object]:
    observations = V1._read_observations(data_dir)
    hidden_audit = V1._audit_hidden(observations, contract)
    blocks = V1._blocks(contract)
    hidden = V1.TimeBlock.from_strings(
        "hidden", contract["problem_contract"]["hidden_interval_kst"]
    )
    spec = V1._sigmoid_spec(contract)
    feature_config = contract["public_features"]
    V1._status(
        status_path,
        started=started,
        progress=5.0,
        phase="features",
        detail="building frozen v1 public-only features and key-only Gate 1 population",
    )
    public_features = build_public_features(
        observations,
        public_layers=tuple(
            int(current) for current in feature_config["temperature_and_salinity_layers"]
        ),
        gradient_pairs=feature_config["adjacent_gradient_pairs"],
        change_hours=tuple(int(current) for current in feature_config["causal_change_hours"]),
    )
    keys = _key_population(contract)
    fold_results: dict[str, object] = {}
    catalogs: dict[str, Any] = {}
    for number, outer in enumerate(blocks):
        progress_start = 8.0 + number * 25.0
        progress_stop = progress_start + 25.0
        result, catalog = V1._precheck_fold(
            observations=observations,
            public_features=public_features,
            outer=outer,
            hidden=hidden,
            contract=contract,
            spec=spec,
            status_path=status_path,
            started=started,
            progress_start=progress_start,
            progress_stop=progress_stop,
        )
        original_gate = result["gates"]["gate_1"]
        corrected_gate = _correct_gate_1(
            public_features=public_features,
            key_population=keys,
            block=outer,
            contract=contract,
        )
        corrected_gate["v1_full_grid_original"] = original_gate
        result["gates"]["gate_1"] = corrected_gate
        result["pass_gates_1_to_4"] = bool(
            all(current["pass"] for current in result["gates"].values())
        )
        result["catalog_reused_from_v1"] = False
        result["catalog_refit_from_source"] = True
        fold_results[outer.name] = result
        catalogs[outer.name] = catalog
        V1._status(
            status_path,
            started=started,
            progress=progress_stop,
            phase=f"precheck {outer.name}",
            detail=f"corrected gates 1-4 {'PASS' if result['pass_gates_1_to_4'] else 'FAIL'}",
        )
    pass_1_to_4 = bool(all(value["pass_gates_1_to_4"] for value in fold_results.values()))
    if pass_1_to_4:
        V1._status(
            status_path,
            started=started,
            progress=84.0,
            phase="inner gate",
            detail="all corrected gates 1-4 passed; opening incumbent prediction for Gate 5 only",
        )
        incumbent = V1._load_incumbent(ROOT, contract)
        gate_5 = V1._gate_5(
            observations=observations,
            public_features=public_features,
            catalogs=catalogs,
            blocks=blocks,
            contract=contract,
            spec=spec,
            incumbent=incumbent,
        )
        oof_prediction_reads_for_gate_5 = 1
    else:
        gate_5 = {
            "executed": False,
            "reason": "at least one outer fold failed corrected gates 1-4",
            "outer_truth_scored": False,
        }
        oof_prediction_reads_for_gate_5 = 0
    receipt = {
        **_base_receipt(
            config_path=config_path,
            contract=contract,
            observation_path=data_dir / "observations.csv",
            mode="precheck",
        ),
        "status": "inner_gate_evaluated" if gate_5["executed"] else "precheck_failed",
        "data_contract": {
            "observation_rows": len(observations),
            **hidden_audit,
            "incumbent_key_rows": len(keys),
            "public_feature_rows": len(public_features),
            "public_model_feature_count": len(feature_columns(public_features)),
        },
        "gates_1_to_4": {"pass": pass_1_to_4, "folds": fold_results},
        "gate_5_inner_only": gate_5,
        "access_log": {
            "gate_1_oof_columns": ["time", "layer"],
            "gate_1_truth_reads": 0,
            "gate_1_prediction_reads": 0,
            "gate_5_prediction_read_operations": oof_prediction_reads_for_gate_5,
        },
        "forbidden_operations": {
            "hidden_target_value_reads": 0,
            "outer_truth_scores": 0,
            "outer_prediction_rows": 0,
            "test_index_reads": 0,
            "submission_rows": 0,
            "upload_attempts": 0,
        },
    }
    V1._write_json(output_dir / "precheck.json", receipt)
    V1._status(
        status_path,
        started=started,
        progress=100.0,
        phase="complete",
        detail=(
            "v2 Gate 5 inner-only evaluated; no outer score"
            if gate_5["executed"]
            else "v2 fail-fast before Gate 5"
        ),
        state="complete" if gate_5["executed"] else "stopped",
    )
    return receipt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/p2_dynamic_sigmoid_profile_v2_key_aligned.json"),
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-path", type=Path)
    parser.add_argument("--mode", choices=("dry-run", "precheck"), required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_path = (ROOT / args.config).resolve() if not args.config.is_absolute() else args.config
    contract, _ = _load_contract(config_path)
    data_dir = V1._resolve_data_dir(args.data_dir)
    output_dir = (
        (ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = args.status_path or Path(contract["outputs"]["status_path"])
    status_path = (ROOT / status_path).resolve() if not status_path.is_absolute() else status_path
    started = time.perf_counter()
    try:
        if args.mode == "dry-run":
            result = _dry_run(
                config_path=config_path,
                contract=contract,
                data_dir=data_dir,
                output_dir=output_dir,
                status_path=status_path,
                started=started,
            )
        else:
            result = _precheck(
                config_path=config_path,
                contract=contract,
                data_dir=data_dir,
                output_dir=output_dir,
                status_path=status_path,
                started=started,
            )
    except Exception as exc:
        V1._status(
            status_path,
            started=started,
            progress=100.0,
            phase="failed",
            detail=f"{type(exc).__name__}: {exc}",
            state="failed",
        )
        raise
    print(json.dumps(V1._json_safe(result), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
