from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HELPER_MODULE = "p2_restore.joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v1"
HELPER_RELATIVE = (
    "src/p2_restore/joint_hydrographic_multitask_layer4_r3_compatibility_verifier_v1.py"
)
NUMERICAL_PREFIXES = ("numpy", "pandas", "scipy", "sklearn", "torch")


def _loaded_numerical_modules() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in NUMERICAL_PREFIXES)
    )


def _load_helper(root: Path):
    path = (root / HELPER_RELATIVE).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(HELPER_MODULE, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the canonical compatibility verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[HELPER_MODULE] = module
    spec.loader.exec_module(module)
    return module


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only verification of the frozen P2 Layer-4 r3 result."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--mode", choices=("check-only",), default="check-only")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve(strict=True)
    numerical_before = _loaded_numerical_modules()
    verifier = _load_helper(root)

    requested = args.config
    if requested is not None and not requested.is_absolute():
        requested = root / requested
    result = verifier.verify_static_compatibility(
        root,
        requested_config=requested,
    )
    numerical_after = _loaded_numerical_modules()
    result["check_only_parent_process"] = {
        "numerical_modules_before": numerical_before,
        "numerical_modules_after": numerical_after,
        "new_numerical_modules": sorted(set(numerical_after) - set(numerical_before)),
        "helper_imported": HELPER_MODULE in sys.modules,
        "r3_guard_imported": verifier.R3_MODULE in sys.modules,
        "r3_engine_imported": (
            "p2_restore.joint_hydrographic_multitask_layer4_execution_r3" in sys.modules
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
