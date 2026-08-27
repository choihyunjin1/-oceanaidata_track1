"""Fail-closed canonical-contract and one-shot filesystem guards."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def authorize_canonical_contract(
    *,
    root: Path,
    requested_config: Path,
    requested_cache: Path,
    requested_output: Path,
    canonical_config_relative: str,
    canonical_cache_relative: str,
    canonical_output_relative: str,
    expected_config_sha256: str,
    expected_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Authorize only the one exact path/SHA/deep-equal experiment contract."""

    workspace = root.resolve(strict=True)
    canonical = {
        "config": (workspace / canonical_config_relative).resolve(strict=True),
        "cache": (workspace / canonical_cache_relative).resolve(strict=True),
        "output": (workspace / canonical_output_relative).resolve(strict=False),
    }
    requested = {
        "config": requested_config.resolve(strict=True),
        "cache": requested_cache.resolve(strict=True),
        "output": requested_output.resolve(strict=False),
    }
    for name in canonical:
        if requested[name] != canonical[name]:
            raise PermissionError(f"non-canonical {name} path is forbidden")
    content = canonical["config"].read_bytes()
    observed_sha = sha256_bytes(content)
    if observed_sha != expected_config_sha256:
        raise PermissionError("canonical config SHA differs from the compiled contract")
    parsed = json.loads(content)
    if parsed != expected_config:
        raise PermissionError("canonical config fails full deep equality")
    if canonical["output"].exists():
        raise FileExistsError("canonical append-only output already exists")
    return parsed, canonical


def acquire_persistent_attempt_lock(
    lock_path: Path,
    *,
    experiment_id: str,
    config_sha256: str,
    created_at: str,
) -> dict[str, Any]:
    """Consume one experiment attempt using an immutable O_EXCL receipt."""

    target = lock_path.resolve(strict=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": created_at,
        "status": "ATTEMPT_CONSUMED_ONE_SHOT",
        "experiment_id": experiment_id,
        "canonical_config_sha256": config_sha256,
        "o_excl": True,
        "rerun_forbidden": True,
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(target, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # The lock intentionally remains consumed even if receipt serialization fails.
        raise
    persisted = json.loads(target.read_text(encoding="utf-8"))
    if persisted != payload:
        raise RuntimeError("persistent attempt-lock receipt failed round-trip verification")
    return {**payload, "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def safe_new_stage_path(
    stage: Path,
    relative: str,
    *,
    protected_roots: tuple[Path, ...],
) -> Path:
    """Resolve an output below a fresh stage and reject traversal or overwrite targets."""

    relative_path = Path(relative)
    if relative_path.is_absolute() or not relative_path.parts or ".." in relative_path.parts:
        raise PermissionError("absolute or traversing output path is forbidden")
    stage_root = stage.resolve(strict=True)
    target = (stage_root / relative_path).resolve(strict=False)
    if target == stage_root or not target.is_relative_to(stage_root):
        raise PermissionError("output target escapes its fresh stage")
    for protected in protected_roots:
        protected_root = protected.resolve(strict=False)
        if target == protected_root or target.is_relative_to(protected_root):
            raise PermissionError("output target intersects a protected root")
    if target.exists():
        raise FileExistsError("output target already exists before write")
    current = target.parent
    while current != stage_root:
        if current.exists() and current.is_symlink():
            raise PermissionError("symlinked output parent is forbidden")
        if current.parent == current:
            raise PermissionError("output parent traversal escaped the stage")
        current = current.parent
    return target
