"""Authenticated CLI payload for the Gen6r2 compatibility-v2 bootstrap."""

from __future__ import annotations

if (
    "__trusted_bootstrap_context__" not in globals()
    or "__trusted_bootstrap_token__" not in globals()
):
    raise RuntimeError("compatibility-v2 CLI must be executed by its trusted bootstrap")

from pathlib import Path
from typing import Any

_TRUST_CONTEXT = globals()["__trusted_bootstrap_context__"]
_TRUST_TOKEN = globals()["__trusted_bootstrap_token__"]
_HELPER = globals().get("__trusted_v2_helper__")
if _TRUST_CONTEXT.get("token") is not _TRUST_TOKEN or _HELPER is not _TRUST_CONTEXT.get(
    "v2_helper"
):
    raise RuntimeError("compatibility-v2 CLI trust identity differs")


def run(
    *,
    root: Path,
    requested_config: Path | None,
    mode: str,
) -> dict[str, Any]:
    """Consume the bootstrap's one-shot CLI phase and run check-only verification."""

    if mode != "check-only":
        raise RuntimeError("compatibility-v2 supports check-only mode exclusively")
    _TRUST_CONTEXT["claim_phase"]("CLI_VERIFY_ONCE")
    workspace = Path(root)
    if (
        workspace.absolute() != _TRUST_CONTEXT["root"].absolute()
        or workspace.resolve(strict=True) != _TRUST_CONTEXT["root"]
    ):
        raise RuntimeError("compatibility-v2 CLI root identity differs")
    return _HELPER.verify_trusted(workspace, requested_config=requested_config)


__all__ = ["run"]
