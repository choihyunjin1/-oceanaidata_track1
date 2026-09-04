"""Run the sealed P2 Gaussian-copula zero-fit deployability preflight once."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from p2_restore.p2_gaussian_copula_frozen_deployability_preflight_20260830_v3 import (  # noqa: E402
    evaluate_preflight,
    load_json,
    reserve_attempt,
    sha256_file,
    write_json_exclusive,
)

EXPECTED_CONFIG_SHA256 = "3eaa563d544c66e4739189badd9b5faca7802b4d00a2d503eef440cae4443bd3"
DEFAULT_CONFIG = Path(
    "configs/experiments/"
    "p2_gaussian_copula_frozen_deployability_preflight_20260830_v3.json"
)
DEFAULT_OUTPUT = Path(
    "reports/p2_gaussian_copula_frozen_deployability_preflight_20260830_v3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _under_root(repo_root: Path, path: Path) -> Path:
    root = repo_root.resolve()
    candidate = path if path.is_absolute() else root / path
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository root: {path}") from exc
    return candidate


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    config_path = _under_root(repo_root, args.config)
    output_dir = _under_root(repo_root, args.output_dir)
    config_sha = sha256_file(config_path)
    if config_sha != EXPECTED_CONFIG_SHA256:
        raise RuntimeError(
            f"Canonical config hash mismatch: expected {EXPECTED_CONFIG_SHA256}, got {config_sha}"
        )
    config = load_json(config_path)

    started = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat()
    lock_path = reserve_attempt(output_dir, config_sha)
    result = evaluate_preflight(repo_root, config, config_sha)
    result["runtime"] = {
        "started_at_kst": started_at,
        "completed_at_kst": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "python": sys.version.split()[0],
    }
    result_path = output_dir / "result.json"
    write_json_exclusive(result_path, result)
    receipt = {
        "experiment_id": result["experiment_id"],
        "status": result["status"],
        "official_probe_ready": result["official_probe_ready"],
        "model_fits": result["execution_receipt"]["model_fits"],
        "thread_budget": result["execution_receipt"]["thread_budget"],
        "config_sha256": config_sha,
        "attempt_lock_sha256": sha256_file(lock_path),
        "result_sha256": sha256_file(result_path),
        "result_path": str(result_path),
    }
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
