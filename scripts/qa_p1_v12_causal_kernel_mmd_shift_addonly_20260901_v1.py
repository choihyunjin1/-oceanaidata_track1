"""Lifecycle-safe independent QA for P1 v12 kernel MMD."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_v12_causal_kernel_mmd_shift_addonly_20260901_v1"


def _qa_module():
    path = ROOT / "scripts/qa_p1_v11_causal_wavelet_scattering_addonly_20260901_v1.py"
    spec = importlib.util.spec_from_file_location("p1_v12_shared_qa", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("shared QA load failed")
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def verify(data_dir: Path) -> dict[str, object]:
    qa = _qa_module()
    qa.EXPERIMENT_ID = EXPERIMENT_ID
    qa.CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
    qa.RUNNER = ROOT / f"scripts/run_{EXPERIMENT_ID}.py"
    qa.ARTIFACT = ROOT / f"artifacts/{EXPERIMENT_ID}"
    qa.RESULT = qa.ARTIFACT / "result.json"
    qa.COMPLETE = qa.ARTIFACT / "predictions_complete.json"
    qa.LOCK = ROOT / f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
    return qa.verify(data_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = verify(args.data_dir)
    print(json.dumps(payload, sort_keys=True, ensure_ascii=True))
    if payload["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
