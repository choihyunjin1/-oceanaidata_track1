from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_p1_frozen_direct_event_verifier_blocked_20260828_v2.py"


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("p1_frozen_verifier_v2_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _semantic_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    config.pop("experiment_id")
    config["artifacts"] = dict(config["artifacts"])
    config["artifacts"].pop("directory")
    return config


def test_v2_rebinds_only_identity_and_output() -> None:
    runner = _load_runner()
    engine = runner.load_fixed_engine()
    assert engine.EXPERIMENT_ID == runner.EXPERIMENT_ID
    assert engine.CONFIG_PATH == runner.CONFIG_PATH
    assert engine.ARTIFACT_DIR == runner.ARTIFACT_DIR
    assert Path(engine.__file__).resolve() == RUNNER


def test_v2_config_is_semantically_identical_to_v1() -> None:
    v1 = ROOT / "configs/experiments/p1_frozen_direct_event_verifier_blocked_20260828_v1.json"
    v2 = ROOT / "configs/experiments/p1_frozen_direct_event_verifier_blocked_20260828_v2.json"
    assert _semantic_config(v1) == _semantic_config(v2)
