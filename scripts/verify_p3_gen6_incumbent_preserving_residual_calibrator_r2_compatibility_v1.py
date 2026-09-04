from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HELPER_MODULE = "p3_wave.gen6_incumbent_preserving_residual_calibrator_r2_compatibility_verifier_v1"
HELPER_RELATIVE = (
    "src/p3_wave/gen6_incumbent_preserving_residual_calibrator_r2_compatibility_verifier_v1.py"
)
NUMERICAL_PREFIXES = ("numpy", "pandas", "pyarrow", "scipy", "sklearn", "torch")


def _loaded_numerical_roots() -> list[str]:
    return sorted(
        prefix
        for prefix in NUMERICAL_PREFIXES
        if any(name == prefix or name.startswith(prefix + ".") for name in sys.modules)
    )


def _load_helper(root: Path):
    path = (root / HELPER_RELATIVE).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(HELPER_MODULE, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the canonical P3 compatibility verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[HELPER_MODULE] = module
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only verification of the frozen P3 Gen6r2 research result."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("check-only",), default="check-only")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve(strict=True)
    numerical_before = _loaded_numerical_roots()
    verifier = _load_helper(root)
    requested = args.config
    if requested is not None and not requested.is_absolute():
        requested = root / requested
    result = verifier.verify_static_compatibility(
        root,
        requested_config=requested,
    )
    numerical_after = _loaded_numerical_roots()
    result["check_only_parent_process"] = {
        "mode": args.mode,
        "numerical_modules_before": numerical_before,
        "new_numerical_modules": sorted(set(numerical_after) - set(numerical_before)),
        "helper_imported": HELPER_MODULE in sys.modules,
        "r2_contract_imported": verifier.R2_MODULE in sys.modules,
        "r2_engine_imported": (
            "p3_wave.gen6_incumbent_preserving_residual_calibrator_execution_r2" in sys.modules
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
