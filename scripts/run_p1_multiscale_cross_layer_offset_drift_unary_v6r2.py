"""Authenticated runner body for the P1 Gen6r2 research curve."""

from __future__ import annotations

import sys
from typing import Any

try:
    _CONTEXT = _P1_V6R2_BOOTSTRAP_CONTEXT  # type: ignore[name-defined]  # noqa: F821
    contract = _P1_V6R2_AUTH_CONTRACT  # type: ignore[name-defined]  # noqa: F821
    science = _P1_V6R2_AUTH_SCIENCE  # type: ignore[name-defined]  # noqa: F821
except NameError as exc:  # pragma: no cover - direct-import guard
    raise RuntimeError("P1 Gen6r2 runner requires the authenticated bootstrap") from exc

if not isinstance(_CONTEXT, dict) or _CONTEXT.get("all_owner_roles_authenticated") is not True:
    raise RuntimeError("P1 Gen6r2 runner loaded before owner source authentication")


def check_only() -> dict[str, Any]:
    if _CONTEXT.get("mode") != "check-only":
        raise PermissionError("static check requires check-only mode")
    if _CONTEXT.get("engine_loaded") is not False or any(
        name.endswith("multiscale_cross_layer_offset_drift_execution_v6r2") for name in sys.modules
    ):
        raise PermissionError("execution engine loaded during check-only")
    science_audit = science.static_contract_audit()
    report = contract.static_preflight(science_audit)
    if _CONTEXT.get("writes_observed") != 0:
        raise PermissionError("check-only observed a filesystem write")
    report["numerical_modules_loaded_by_check_only"] = sorted(
        set(sys.modules).intersection({"numpy", "pandas", "sklearn", "torch"})
        - set(_CONTEXT["modules_before_owner_load"])
    )
    if report["numerical_modules_loaded_by_check_only"]:
        raise PermissionError("check-only imported a numerical package")
    return report


def execute() -> dict[str, Any]:
    if _CONTEXT.get("mode") != "execute":
        raise PermissionError("execution runner requires execute mode")
    engine = _CONTEXT.get("engine_module")
    if engine is None or _CONTEXT.get("engine_loaded") is not True:
        raise PermissionError("authenticated execution engine is unavailable")
    prelock = contract.authorize_entry()
    capability = contract.acquire_attempt_lock(prelock)
    contract.create_output_tree(capability)
    contract.enter_phase(capability, expected="locked", new="loaded")
    contract.enter_phase(capability, expected="loaded", new="curve")
    return engine.run_curve(capability)


def main() -> dict[str, Any]:
    mode = _CONTEXT.get("mode")
    if mode == "check-only":
        return check_only()
    if mode == "execute":
        return execute()
    raise PermissionError("canonical bootstrap mode differs")


__all__ = ["main"]
