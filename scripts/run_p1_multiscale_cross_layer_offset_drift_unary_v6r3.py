"""Authenticated dispatcher for P1 multiscale Gen6r3.

Normal Python import and direct execution are forbidden.  The external launcher
enters through the noncyclic bootstrap, which injects only authenticated module
objects and an opaque live capability.
"""

from __future__ import annotations

from typing import Any

try:
    _CONTEXT = _P1_V6R3_BOOTSTRAP_CONTEXT  # type: ignore[name-defined]  # noqa: F821
    contract = _P1_V6R3_AUTH_CONTRACT  # type: ignore[name-defined]  # noqa: F821
    engine = _P1_V6R3_AUTH_ENGINE  # type: ignore[name-defined]  # noqa: F821
except NameError as exc:  # pragma: no cover - direct-import guard
    raise RuntimeError("P1 Gen6r3 runner requires the authenticated bootstrap") from exc


def check_only() -> dict[str, Any]:
    if _CONTEXT.get("mode") != "check-only":
        raise PermissionError("check_only is unavailable outside canonical check-only mode")
    return contract.static_preflight(require_future_state_absent=True)


def execute(capability: object) -> dict[str, Any]:
    mode = _CONTEXT.get("mode")
    if mode == "parent":
        return engine.run_parent(capability)
    if mode == "cell_worker":
        cell = _CONTEXT.get("cell")
        prior = _CONTEXT.get("prior_event_sha256")
        if type(cell) is not int or not contract._is_sha(prior):
            raise PermissionError("cell-worker dispatch identity differs")
        return engine.run_cell(capability, cell=cell, prior_event_sha256=prior)
    raise PermissionError("execute is unavailable outside an authenticated execution mode")


__all__ = ["check_only", "execute"]
