"""Small, dependency-light guards shared by copied standalone final packages.

The build step copies this file into every problem directory.  A packaged
problem therefore never imports code from another problem directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    """Raised when a frozen final-submission contract does not match disk."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: str | Path, expected_sha256: str, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ContractError(f"missing {label}: {resolved}")
    actual = sha256_file(resolved)
    if actual != expected_sha256:
        raise ContractError(f"SHA-256 mismatch for {label}: {actual} != {expected_sha256}")
    return resolved


def load_contract(package_dir: str | Path) -> dict[str, Any]:
    path = Path(package_dir).resolve() / "contract.json"
    if not path.is_file():
        raise ContractError(f"missing contract.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def verify_official_files(data_dir: str | Path, contract: dict[str, Any]) -> dict[str, Any]:
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise ContractError(f"official data directory does not exist: {root}")
    checked: dict[str, str] = {}
    for name, expected in contract["official_inputs"].items():
        path = require_file(root / name, expected, f"official input {name}")
        checked[name] = sha256_file(path)
    return {"data_dir": str(root), "sha256": checked}


def verify_package_files(
    package_dir: str | Path, contract: dict[str, Any], section: str
) -> list[dict[str, Any]]:
    """Verify every package-relative file declared in one contract section."""

    root = Path(package_dir).resolve()
    checked: list[dict[str, Any]] = []
    for record in contract.get(section, []):
        relative = Path(record["path"])
        path = require_file(root / relative, record["sha256"], f"{section} {relative}")
        checked.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": record["sha256"],
            }
        )
    return checked


def bounded_receipt(payload: dict[str, Any]) -> dict[str, Any]:
    """Return metadata only; never expose prediction rows in notebook output."""

    allowed = {
        "status",
        "candidate_id",
        "rows",
        "columns",
        "sha256",
        "positive_rows",
        "changed_rows",
        "minimum",
        "maximum",
        "key_order_exact",
        "official_input_hashes_ok",
        "candidate_hash_exact",
        "package_atomic",
        "lineage",
        "caveat",
        "checkpoint_files_loaded",
        "training_fit_count",
        "prediction_source",
        "historical_champion_hash_exact",
        "historical_champion_sha256",
    }
    return {key: value for key, value in payload.items() if key in allowed}
