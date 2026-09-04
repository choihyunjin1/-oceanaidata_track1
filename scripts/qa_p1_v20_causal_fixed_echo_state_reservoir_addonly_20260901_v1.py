"""Lifecycle-safe QA for P1 v20."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v20_causal_fixed_echo_state_reservoir_addonly_20260901_v1"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("module load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def verify(data_dir: Path) -> dict[str, object]:
    qa = _module(
        ROOT / "scripts/qa_p1_v11_causal_wavelet_scattering_addonly_20260901_v1.py",
        "p1_v20_shared_qa",
    )
    qa.EXPERIMENT_ID = EXPERIMENT_ID
    qa.CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
    qa.RUNNER = ROOT / f"scripts/run_{EXPERIMENT_ID}.py"
    qa.ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
    qa.RESULT, qa.COMPLETE = qa.ARTIFACT / "result.json", qa.ARTIFACT / "predictions_complete.json"
    qa.LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
    payload = qa.verify(data_dir)
    runner = _module(qa.RUNNER, "p1_v20_boundary")
    result = json.loads(qa.RESULT.read_text(encoding="utf-8"))
    boundary = runner.shared.boundary_recall_from_artifacts(data_dir)
    payload["checks"]["boundary_recall"] = boundary == result["long_event_boundary"]
    payload["recomputed"]["long_event_boundary"] = boundary
    payload["verdict"] = "PASS" if all(payload["checks"].values()) else "FAIL"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = verify(args.data_dir)
    print(json.dumps(payload, sort_keys=True, ensure_ascii=True))
    raise SystemExit(0 if payload["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
