"""Run the preregistered inner-only matched-filter comparison; never access outer folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1_qc.config import load_config  # noqa: E402
from p1_qc.data import load_train_test  # noqa: E402
from p1_qc.experiment import sha256_file, write_json  # noqa: E402
from p1_qc.matched_filter import (  # noqa: E402
    append_matched_filter_features,
    build_matched_filter_features,
)
from p1_qc.matched_filter_experiment import (  # noqa: E402
    load_and_validate_contract,
    run_inner_comparison,
)
from p1_qc.pipeline import load_or_build_features, resolve_data_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/p1.toml")
    parser.add_argument(
        "--contract", default="configs/experiments/p1_offset_drift_matched_filter_v1.json"
    )
    parser.add_argument("--data-dir")
    parser.add_argument("--output", default="artifacts/matched_filter_inner_v1")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    contract_path = (PROJECT_ROOT / args.contract).resolve()
    output = (PROJECT_ROOT / args.output).resolve()
    config = load_config(config_path)
    contract = load_and_validate_contract(contract_path, project_root=PROJECT_ROOT)
    data_dir = resolve_data_dir(config, args.data_dir)
    hashes = contract["hashes"]
    exact_files = {
        "config_sha256": config_path,
        "train_sha256": data_dir / "train.csv",
        "test_sha256": data_dir / "test.csv",
        "base_feature_cache_sha256": PROJECT_ROOT
        / "artifacts/cache/train_offline_e9fe1eb46cb7431f.parquet",
        "incumbent_metrics_sha256": PROJECT_ROOT
        / "artifacts/runs/20260813T153038+0900_cv_378a4e89/metrics.json",
        "incumbent_selection_sha256": PROJECT_ROOT
        / "artifacts/runs/20260813T153038+0900_cv_378a4e89/selection.json",
    }
    for key, path in exact_files.items():
        if sha256_file(path) != hashes[key]:
            raise RuntimeError(f"preregistered input hash mismatch: {key}")

    train, test = load_train_test(data_dir, audit=True, strict=True)
    baseline = load_or_build_features(train, config, kind="train", use_cache=True)
    matched = build_matched_filter_features(train)
    candidate = append_matched_filter_features(baseline, train)
    if not candidate.frame.loc[:, matched.columns].equals(matched):
        raise RuntimeError("matched-filter append did not preserve exact feature values")
    result = run_inner_comparison(train, test, baseline, candidate, config, contract)

    output.mkdir(parents=True, exist_ok=True)
    oof_path = output / "inner_oof.parquet"
    metrics_path = output / "metrics.json"
    manifest_path = output / "manifest.json"
    result.oof.to_parquet(oof_path, index=False, compression="zstd")
    write_json(metrics_path, result.metrics)
    manifest = {
        "experiment_id": contract["experiment_id"],
        "contract_sha256": sha256_file(contract_path),
        "metrics_sha256": sha256_file(metrics_path),
        "oof_sha256": sha256_file(oof_path),
        "rows": len(result.oof),
        "passed": result.passed,
        "outer_accessed": False,
        "test_values_used": False,
        "test_group_keys_used_for_weighting": True,
        "submission_created": False,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
