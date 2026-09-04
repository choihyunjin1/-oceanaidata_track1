"""CLI body executed only from the authenticated P2 v5 bootstrap buffer."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    _CONTEXT = _P2_V5_BOOTSTRAP_CONTEXT  # type: ignore[name-defined]  # noqa: F821
    _HELPER = _P2_V5_AUTHENTICATED_HELPER  # type: ignore[name-defined]  # noqa: F821
except NameError as exc:  # pragma: no cover - subprocess contract test
    raise RuntimeError("P2 v5 CLI requires the authenticated bootstrap") from exc


def run_authenticated() -> dict[str, Any]:
    _CONTEXT["assert_runtime"]()
    arguments = _CONTEXT["arguments"]
    root = Path(arguments["root"]).resolve(strict=True)
    if root != _CONTEXT["workspace"]:
        raise RuntimeError("P2 v5 CLI workspace changed")
    if arguments["mode"] != "check-only":
        raise RuntimeError("P2 v5 CLI is check-only")
    requested = arguments.get("config")
    if requested is not None:
        requested_path = Path(requested)
        if not requested_path.is_absolute():
            requested_path = root / requested_path
        canonical = root / _CONTEXT["trusted_pins"]["CONFIG"]["path"]
        if requested_path.resolve(strict=True) != canonical.resolve(strict=True):
            raise RuntimeError("alternate P2 v5 config is forbidden")
    report = _HELPER.verify_static_compatibility()
    _CONTEXT["assert_runtime"]()
    report["authenticated_cli"] = {
        "mode": "check-only",
        "helper_module": _HELPER.__name__,
        "r3_guard_module": _HELPER.R3_MODULE,
        "r3_engine_imported": _HELPER.R3_ENGINE_MODULE in sys.modules,
        "sys_dont_write_bytecode": sys.dont_write_bytecode,
        "isolated": sys.flags.isolated,
        "no_site": sys.flags.no_site,
        "safe_path": sys.flags.safe_path,
        "canonical_pycache_sentinel": _CONTEXT["pycache_prefix_relative"],
        "pycache_sentinel_regular_held_intact": _CONTEXT["pycache_sentinel_intact"](),
        "external_launcher_required": True,
        "external_launcher_self_authentication_claimed": False,
    }
    return report


__all__ = ["run_authenticated"]
