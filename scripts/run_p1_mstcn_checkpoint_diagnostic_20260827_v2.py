"""Windows-safe append-only launcher for P1 checkpoint diagnostic v2.

The scientific implementation is the byte-pinned v1 runner.  v2 changes only
the torch state durability primitive: serialization occurs through the same
writable temporary handle that is flushed and fsynced before atomic replace.
No failed-v1 model, optimizer, prediction, or receipt is reused.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "p1_mstcn_checkpoint_diagnostic_20260827_v2"
BASE_EXPERIMENT_ID = "p1_mstcn_checkpoint_diagnostic_20260827_v1"
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
ATTEMPT_LOCK = ROOT / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
BASE_RUNNER_PATH = ROOT / "scripts" / f"run_{BASE_EXPERIMENT_ID}.py"
EXPECTED_BASE_RUNNER_SHA256 = "5197715acb4305335e1f91435da96c2df2dba5a41c94e7b037103b6d588a374f"
EXPECTED_CONFIG_SHA256 = "437c5c0aa2d2c1508518c48c0a469aa2ae08b9b927a918fc7c7144784c2c0d0c"


class ContractError(RuntimeError):
    """Raised when the isolated recovery launcher contract fails."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_atomic_torch_save(path: Path, value: Any, torch: Any) -> str:
    """Serialize, flush, and fsync one writable handle before atomic replace."""

    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return _sha256(path)


def _load_implementation(*, root: Path = ROOT) -> Any:
    base_path = root / "scripts" / f"run_{BASE_EXPERIMENT_ID}.py"
    if not base_path.is_file() or _sha256(base_path) != EXPECTED_BASE_RUNNER_SHA256:
        raise ContractError("byte-pinned v1 scientific implementation changed")
    config_path = root / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
    if not config_path.is_file() or _sha256(config_path) != EXPECTED_CONFIG_SHA256:
        raise ContractError("v2 recovery config changed")
    name = f"{EXPERIMENT_ID}_pinned_implementation"
    spec = importlib.util.spec_from_file_location(name, base_path)
    if spec is None or spec.loader is None:
        raise ContractError("cannot load pinned v1 scientific implementation")
    implementation = importlib.util.module_from_spec(spec)
    sys.modules[name] = implementation
    spec.loader.exec_module(implementation)

    # Patch only namespace/config identity plus the failed durability primitive.
    implementation.EXPERIMENT_ID = EXPERIMENT_ID
    implementation.ROOT = root
    implementation.CONFIG_PATH = config_path
    implementation.ARTIFACT_DIR = root / "artifacts" / EXPERIMENT_ID
    implementation.ATTEMPT_LOCK = root / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
    implementation.EXPECTED_CONFIG_SHA256 = EXPECTED_CONFIG_SHA256
    implementation.__file__ = str(Path(__file__).resolve())
    implementation._atomic_torch_save = _safe_atomic_torch_save
    return implementation


def check_only(*, root: Path = ROOT) -> dict[str, Any]:
    implementation = _load_implementation(root=root)
    result = implementation.check_only(root=root)
    if not (
        result.get("experiment_id") == EXPERIMENT_ID
        and result.get("config_sha256") == EXPECTED_CONFIG_SHA256
        and result.get("runner_sha256") == _sha256(Path(__file__))
        and result.get("official_interface_reads") == 0
        and result.get("q3_q4_truth_columns_read") == 0
        and result.get("result") == "PASS"
    ):
        raise ContractError("v2 delegated preflight contract changed")
    return {
        **result,
        "recovery_from": BASE_EXPERIMENT_ID,
        "failed_v1_state_reused": False,
        "scientific_contract_changed": False,
        "durability_primitive": "writable_handle_torch_save_flush_fsync_atomic_replace",
    }


def execute(*, expected_runner_sha256: str, root: Path = ROOT) -> dict[str, Any]:
    observed = _sha256(Path(__file__))
    if expected_runner_sha256.casefold() != observed:
        raise ContractError("--expected-runner-sha256 must match reviewed v2 launcher bytes")
    implementation = _load_implementation(root=root)
    return implementation.execute(expected_runner_sha256=observed, root=root)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-runner-sha256")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.check_only:
        result = check_only()
    else:
        if not args.expected_runner_sha256:
            raise ContractError("--execute requires --expected-runner-sha256")
        result = execute(expected_runner_sha256=args.expected_runner_sha256)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
