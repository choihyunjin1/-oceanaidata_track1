from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from p1_qc.config import load_config
from p1_qc.data import load_train_test
from p1_qc.experiment import environment_summary, git_sha, sha256_file, stable_hash, write_json
from p1_qc.pipeline import load_or_build_features, resolve_data_dir
from p1_qc.r1_experiment import change_point_proposal_builder, run_r1_nested_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preregistered P1 R1 nested CV.")
    parser.add_argument("--config", default="configs/p1.toml")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preregistration", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    start_time = datetime.now().astimezone().isoformat()
    config_path = Path(args.config).resolve(strict=True)
    preregistration = Path(args.preregistration).resolve(strict=True)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    data_dir = resolve_data_dir(config, args.data_dir)
    train, test = load_train_test(data_dir, audit=True, strict=True)
    bundle = load_or_build_features(train, config, kind="train", use_cache=True)
    print(
        f"loaded train={len(train)} test={len(test)} features={len(bundle.feature_columns)}",
        flush=True,
    )
    result = run_r1_nested_cv(
        train,
        test,
        bundle,
        config,
        change_point_proposal_builder,
        backend="xgboost",
        augmentation=False,
        primary_metric="micro_f1",
    )
    oof_path = output_dir / "oof.parquet"
    result.oof.to_parquet(oof_path, index=False, compression="zstd")
    metrics_path = write_json(output_dir / "metrics.json", result.metrics)
    selection_path = write_json(output_dir / "selection.json", result.selection)
    manifest = {
        "artifact_kind": "research_only_r1_nested_cv",
        "competition_upload": False,
        "commit_or_push_performed": False,
        "config": asdict(config),
        "config_sha256": sha256_file(config_path),
        "created_at": start_time,
        "data": {
            "train_rows": len(train),
            "test_rows": len(test),
            "train_sha256": train.attrs["source_sha256"],
            "test_sha256": test.attrs["source_sha256"],
        },
        "environment": environment_summary(),
        "feature_contract_sha256": stable_hash(
            {
                "columns": bundle.feature_columns,
                "categorical": bundle.categorical_columns,
                "mode": config.features.mode,
            }
        ),
        "finished_at": datetime.now().astimezone().isoformat(),
        "git_sha": git_sha(),
        "preregistration_sha256": sha256_file(preregistration),
        "runtime_seconds": time.perf_counter() - started,
        "working_tree_expected_dirty": True,
        "artifacts": {
            "metrics.json": sha256_file(metrics_path),
            "oof.parquet": sha256_file(oof_path),
            "selection.json": sha256_file(selection_path),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    summary = result.metrics["aggregate"]["micro"]
    base = result.metrics["base_aggregate"]["micro"]
    print(
        json.dumps(
            {
                "candidate_f1": summary["f1"],
                "base_f1": base["f1"],
                "delta": summary["f1"] - base["f1"],
                "runtime_seconds": manifest["runtime_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
