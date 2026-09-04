"""Dependency-only recovery for the frozen P3 ERA5 context-transfer experiment.

This wrapper verifies the exact ML environment and a CatBoost continuation smoke
test before delegating to the frozen v1 scientific runner under a new append-only
experiment ID and output path.  It does not alter any scientific parameter.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_REL = Path(
    "configs/experiments/p3_era5_context_transfer_dependency_recovery_20260828_v2.json"
)
EXPECTED_ID = "p3_era5_context_transfer_dependency_recovery_20260828_v2"
EXPECTED_OUTPUT_REL = Path("artifacts/p3_era5_context_transfer_dependency_recovery_20260828_v2")
EXPECTED_VERSIONS = {
    "python": "3.12.10",
    "catboost": "1.2.10",
    "scikit_learn": "1.9.0",
    "numpy": "2.3.5",
    "pandas": "3.0.1",
    "pyarrow": "25.0.1",
}


class RecoveryContractError(RuntimeError):
    """Raised when the dependency-only recovery contract drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoveryContractError(f"JSON root is not an object: {path}")
    return value


def _load_frozen_runner(root: Path):
    path = root / "scripts" / "run_p3_era5_context_transfer_v1.py"
    spec = importlib.util.spec_from_file_location("_p3_era5_frozen_v1", path)
    if spec is None or spec.loader is None:
        raise RecoveryContractError("could not load the frozen v1 runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _environment_preflight() -> dict[str, Any]:
    import catboost
    import pandas
    import pyarrow
    import sklearn
    from catboost import CatBoostRegressor

    observed = {
        "python": platform.python_version(),
        "catboost": catboost.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "pyarrow": pyarrow.__version__,
    }
    if observed != EXPECTED_VERSIONS:
        raise RecoveryContractError(
            f"environment version drift: expected={EXPECTED_VERSIONS}, observed={observed}"
        )

    rng = np.random.default_rng(20260828)
    features = rng.normal(size=(32, 286)).astype(np.float64)
    features[0, 7] = np.nan
    target = rng.normal(size=32).astype(np.float64)
    parameters = {
        "iterations": 2,
        "depth": 3,
        "learning_rate": 0.05,
        "loss_function": "RMSE",
        "random_seed": 20260828,
        "thread_count": 1,
        "task_type": "CPU",
        "verbose": False,
        "allow_writing_files": False,
    }
    stage1 = CatBoostRegressor(**parameters)
    stage1.fit(features, target)
    cloned = copy.deepcopy(stage1)
    if cloned is stage1 or cloned.tree_count_ != stage1.tree_count_:
        raise RecoveryContractError("CatBoost deepcopy smoke failed")
    stage2 = CatBoostRegressor(**parameters)
    stage2.fit(features, target, init_model=cloned)
    prediction = np.asarray(stage2.predict(features), dtype=np.float64)
    if stage2.tree_count_ <= stage1.tree_count_ or not np.isfinite(prediction).all():
        raise RecoveryContractError("CatBoost continuation smoke failed")

    return {
        "passed": True,
        "sys_executable": sys.executable,
        "versions": observed,
        "feature_shape": [32, 286],
        "nan_smoke": True,
        "stage1_tree_count": int(stage1.tree_count_),
        "stage2_tree_count": int(stage2.tree_count_),
        "deepcopy_distinct_object": cloned is not stage1,
        "finite_predictions": True,
        "device": "CPU",
    }


def _install_dependency_recovery_contract(module, root: Path) -> Mapping[str, Any]:
    amendment_path = (root / AMENDMENT_REL).resolve()
    amendment = _read_json(amendment_path)
    if amendment.get("experiment_id") != EXPECTED_ID:
        raise RecoveryContractError("wrong recovery experiment ID")
    base_binding = amendment.get("base_experiment", {})
    base_path = (root / str(base_binding.get("path", ""))).resolve()
    if _sha256(base_path) != str(base_binding.get("sha256", "")).lower():
        raise RecoveryContractError("frozen base experiment hash drifted")
    base = _read_json(base_path)
    merged = copy.deepcopy(base)
    merged["experiment_id"] = EXPECTED_ID
    merged["access_and_output"]["artifact_dir"] = EXPECTED_OUTPUT_REL.as_posix()

    validation_view = copy.deepcopy(merged)
    validation_view["experiment_id"] = "p3_era5_context_transfer_v1"
    scope_path = (root / str(merged["bindings"]["external_scope"]["path"])).resolve()
    scope = _read_json(scope_path)
    module._validate_prereg(validation_view, scope)

    old_paths = module._resolve_paths(root, merged, scope)
    output = (root / EXPECTED_OUTPUT_REL).resolve()
    paths = module.RunPaths(
        **{
            **old_paths.__dict__,
            "experiment_config": amendment_path,
            "output": output,
            "attempt_lock": output.with_name(f"{output.name}.attempt.lock"),
        }
    )

    def load_contract(_root: Path):
        resolved = Path(_root).resolve()
        if resolved != root.resolve():
            raise RecoveryContractError("runtime root differs from preflight root")
        return copy.deepcopy(merged), copy.deepcopy(scope), paths

    module._load_contract = load_contract
    return {
        "amendment_path": str(amendment_path),
        "amendment_sha256": _sha256(amendment_path),
        "base_path": str(base_path),
        "base_sha256": _sha256(base_path),
        "output": str(output),
        "scientific_overrides": [],
        "operational_overrides": ["experiment_id", "artifact_dir"],
    }


def check_only(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    environment = _environment_preflight()
    module = _load_frozen_runner(root)
    recovery = _install_dependency_recovery_contract(module, root)
    frozen_check = module.check_only(root)
    return {
        "schema_version": "p3_era5_context_transfer_dependency_recovery.check.v2",
        "experiment_id": EXPECTED_ID,
        "mode": "check-only",
        "passed": True,
        "writes": 0,
        "environment": environment,
        "recovery_contract": recovery,
        "frozen_scientific_check": frozen_check,
        "research_only": True,
        "official_access": False,
    }


def execute_once(root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    environment = _environment_preflight()
    module = _load_frozen_runner(root)
    recovery = _install_dependency_recovery_contract(module, root)
    result = module.execute_once(root)
    return {
        "schema_version": "p3_era5_context_transfer_dependency_recovery.result.v2",
        "experiment_id": EXPECTED_ID,
        "environment": environment,
        "recovery_contract": recovery,
        "frozen_scientific_result": result,
        "research_only": True,
        "official_access": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = execute_once(args.root) if args.execute else check_only(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
