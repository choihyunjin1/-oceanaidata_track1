"""Run v2 by rebinding only the identity/output of the fixed v1 engine."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

EXPERIMENT_ID = "p1_frozen_direct_event_verifier_blocked_20260828_v2"
ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER = (
    ROOT / "scripts" / "run_p1_frozen_direct_event_verifier_blocked_20260828_v1.py"
)
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID


def load_fixed_engine() -> Any:
    """Load the corrected common engine and change identity/output bindings only."""

    spec = importlib.util.spec_from_file_location(f"{EXPERIMENT_ID}_engine", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load fixed base runner: {BASE_RUNNER}")
    engine = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = engine
    spec.loader.exec_module(engine)
    engine.EXPERIMENT_ID = EXPERIMENT_ID
    engine.CONFIG_PATH = CONFIG_PATH
    engine.ARTIFACT_DIR = ARTIFACT_DIR
    engine.__file__ = str(Path(__file__).resolve())
    return engine


def main(argv: Sequence[str] | None = None) -> int:
    return int(load_fixed_engine().main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
