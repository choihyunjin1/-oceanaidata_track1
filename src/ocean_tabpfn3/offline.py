"""Fail-closed, offline-only access to the synthetic TabPFN-3 weights.

The organizer permits synthetic-only pretrained table foundation models, but
the model license still has to be accepted by the user.  This module never
downloads weights, opens a browser, or consumes a token.  It only accepts two
explicit local checkpoint paths after validating a user-authored receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

PACKAGE_VERSION = "8.5.0"
MODEL_VERSION = "v3"
CLASSIFIER_FILENAME = "tabpfn-v3-classifier-v3_default.ckpt"
REGRESSOR_FILENAME = "tabpfn-v3-regressor-v3_default.ckpt"
CLASSIFIER_BYTES = 212_804_803
REGRESSOR_BYTES = 233_289_807
CLASSIFIER_SHA256 = "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988"
REGRESSOR_SHA256 = "311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301"
DEFAULT_ROOT = Path("artifacts/tabpfn3")
LICENSE_RECEIPT_ENV = "TABPFN3_LICENSE_RECEIPT_PATH"
CLASSIFIER_PATH_ENV = "TABPFN3_CLASSIFIER_PATH"
REGRESSOR_PATH_ENV = "TABPFN3_REGRESSOR_PATH"


class TabPFN3ContractError(RuntimeError):
    """Raised before model construction when the offline contract is incomplete."""


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _resolve_path(
    env: dict[str, str], name: str, default: Path, *, workspace: Path
) -> Path:
    raw = env.get(name)
    candidate = Path(raw) if raw else default
    if not candidate.is_absolute():
        candidate = workspace / candidate
    return candidate.expanduser().resolve(strict=False)


def _read_receipt(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    blockers: list[str] = []
    if not path.is_file():
        return None, [f"missing user license receipt: {path}"]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"invalid user license receipt: {type(exc).__name__}"]
    if not isinstance(value, dict):
        return None, ["user license receipt must be a JSON object"]
    required = {
        "schema_version": "ocean.tabpfn3.user_license_receipt.v1",
        "license_accepted_by_user": True,
        "model_version": MODEL_VERSION,
        "synthetic_only_provenance_reviewed": True,
        "competition_use_terms_reviewed": True,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            blockers.append(f"license receipt field mismatch: {key}")
    accepted = value.get("accepted_at_utc")
    try:
        parsed = datetime.fromisoformat(str(accepted).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            blockers.append("accepted_at_utc must include a timezone")
    except ValueError:
        blockers.append("accepted_at_utc is not ISO-8601")
    if value.get("source_url") != "https://platform.priorlabs.ai":
        blockers.append("license receipt source_url mismatch")
    forbidden = {"token", "api_key", "password", "authorization"}
    if forbidden.intersection({str(key).lower() for key in value}):
        blockers.append("license receipt must not contain credentials")
    return value, blockers


def _weight_identity(
    path: Path,
    basename: str,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    if path.name != basename:
        return None, [f"checkpoint basename mismatch: expected {basename}"]
    if not path.is_file():
        return None, [f"missing local checkpoint: {path}"]
    size = int(path.stat().st_size)
    if size != expected_bytes:
        return None, [
            f"checkpoint byte count mismatch for {basename}: "
            f"expected {expected_bytes}, got {size}"
        ]
    observed_sha256 = sha256(path)
    if observed_sha256 != expected_sha256:
        return None, [f"checkpoint SHA-256 mismatch for {basename}"]
    return {"path": str(path), "bytes": size, "sha256": observed_sha256}, []


def inspect_preflight(
    *, workspace: str | Path, env: dict[str, str] | None = None
) -> dict[str, Any]:
    """Return a non-mutating readiness report without triggering any download."""

    root = Path(workspace).resolve(strict=True)
    environment = dict(os.environ if env is None else env)
    receipt_path = _resolve_path(
        environment,
        LICENSE_RECEIPT_ENV,
        DEFAULT_ROOT / "user-license-receipt.json",
        workspace=root,
    )
    classifier_path = _resolve_path(
        environment,
        CLASSIFIER_PATH_ENV,
        DEFAULT_ROOT / "weights" / CLASSIFIER_FILENAME,
        workspace=root,
    )
    regressor_path = _resolve_path(
        environment,
        REGRESSOR_PATH_ENV,
        DEFAULT_ROOT / "weights" / REGRESSOR_FILENAME,
        workspace=root,
    )

    blockers: list[str] = []
    receipt, receipt_blockers = _read_receipt(receipt_path)
    classifier, classifier_blockers = _weight_identity(
        classifier_path,
        CLASSIFIER_FILENAME,
        expected_bytes=CLASSIFIER_BYTES,
        expected_sha256=CLASSIFIER_SHA256,
    )
    regressor, regressor_blockers = _weight_identity(
        regressor_path,
        REGRESSOR_FILENAME,
        expected_bytes=REGRESSOR_BYTES,
        expected_sha256=REGRESSOR_SHA256,
    )
    blockers.extend(receipt_blockers)
    blockers.extend(classifier_blockers)
    blockers.extend(regressor_blockers)

    package: dict[str, Any] = {}
    try:
        import tabpfn
        import torch
        from tabpfn.constants import ModelVersion

        package = {
            "tabpfn": str(tabpfn.__version__),
            "torch": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "v3_enum_available": ModelVersion.V3.value == MODEL_VERSION,
        }
        if package["tabpfn"] != PACKAGE_VERSION:
            blockers.append(
                f"tabpfn package mismatch: expected {PACKAGE_VERSION}, got {package['tabpfn']}"
            )
        if not package["v3_enum_available"]:
            blockers.append("installed tabpfn package lacks ModelVersion.V3")
        if not package["cuda_available"]:
            blockers.append("CUDA is unavailable for the six-hour reproduction budget")
    except (ImportError, AttributeError) as exc:
        blockers.append(f"TabPFN runtime import failed: {type(exc).__name__}: {exc}")

    return {
        "schema_version": "ocean.tabpfn3.offline_preflight.v1",
        "status": "READY" if not blockers else "BLOCKED",
        "model_version": MODEL_VERSION,
        "synthetic_only_pretraining": True,
        "auto_download_allowed": False,
        "browser_login_allowed_by_runner": False,
        "token_read_allowed_by_runner": False,
        "tabpfn_no_browser_required": True,
        "paths": {
            "license_receipt": str(receipt_path),
            "classifier": str(classifier_path),
            "regressor": str(regressor_path),
        },
        "license_receipt": receipt,
        "weights": {"classifier": classifier, "regressor": regressor},
        "runtime": package,
        "blockers": blockers,
    }


def require_ready(
    *, workspace: str | Path, env: dict[str, str] | None = None
) -> dict[str, Any]:
    report = inspect_preflight(workspace=workspace, env=env)
    if report["status"] != "READY":
        raise TabPFN3ContractError("; ".join(report["blockers"]))
    os.environ["TABPFN_NO_BROWSER"] = "1"
    return report


def make_classifier(
    model_path: str | Path,
    *,
    seed: int,
    categorical_features_indices: list[int] | None = None,
    n_estimators: int = 8,
) -> Any:
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    path = Path(model_path).resolve(strict=True)
    if path.name != CLASSIFIER_FILENAME:
        raise TabPFN3ContractError("classifier checkpoint basename changed")
    os.environ["TABPFN_NO_BROWSER"] = "1"
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V3,
        model_path=path,
        device="cuda",
        n_estimators=n_estimators,
        categorical_features_indices=categorical_features_indices,
        ignore_pretraining_limits=False,
        fit_mode="fit_preprocessors",
        memory_saving_mode="auto",
        random_state=seed,
        n_preprocessing_jobs=1,
        show_progress_bar=False,
    )


def make_regressor(
    model_path: str | Path,
    *,
    seed: int,
    categorical_features_indices: list[int] | None = None,
    n_estimators: int = 8,
) -> Any:
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    path = Path(model_path).resolve(strict=True)
    if path.name != REGRESSOR_FILENAME:
        raise TabPFN3ContractError("regressor checkpoint basename changed")
    os.environ["TABPFN_NO_BROWSER"] = "1"
    return TabPFNRegressor.create_default_for_version(
        ModelVersion.V3,
        model_path=path,
        device="cuda",
        n_estimators=n_estimators,
        categorical_features_indices=categorical_features_indices,
        ignore_pretraining_limits=False,
        fit_mode="fit_preprocessors",
        memory_saving_mode="auto",
        random_state=seed,
        n_preprocessing_jobs=1,
        show_progress_bar=False,
    )
