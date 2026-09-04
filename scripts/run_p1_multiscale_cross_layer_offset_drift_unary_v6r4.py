"""Authenticated private dispatcher for P1 multiscale Gen6r4.

There is deliberately no command-line worker, broker, session, prior, cell, or
handle argument.  The authenticated bootstrap obtains role and identity only
from one-use inherited kernel handles and injects this dispatcher afterward.
"""

from __future__ import annotations

from typing import Any

try:
    _CONTEXT = _P1_V6R4_RUNNER_CONTEXT  # type: ignore[name-defined]  # noqa: F821
    engine = _P1_V6R4_AUTH_ENGINE  # type: ignore[name-defined]  # noqa: F821
except NameError as exc:  # pragma: no cover - direct-import guard
    raise RuntimeError("P1 Gen6r4 runner requires authenticated private bootstrap") from exc

if (
    not isinstance(_CONTEXT, dict)
    or _CONTEXT.get("private_inherited_entry_verified") is not True
    or _CONTEXT.get("public_cli_fields_absent") is not True
    or _CONTEXT.get("session_or_prior_from_cli") is not False
):
    raise RuntimeError("P1 Gen6r4 runner rejected public or unverified entry")


def execute() -> dict[str, Any]:
    role = _CONTEXT.get("role")
    if role == "parent":
        return engine.run_parent()
    if role == "cell_worker":
        cell = _CONTEXT.get("cell")
        if type(cell) is not int or type(cell) is bool:
            raise PermissionError("private worker cell identity differs")
        return engine.run_cell(cell=cell)
    raise PermissionError("runner role is not an executable parent or cell worker")


__all__ = ["execute"]
