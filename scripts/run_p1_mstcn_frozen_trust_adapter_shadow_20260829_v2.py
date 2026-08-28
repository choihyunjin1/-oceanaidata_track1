"""Entrypoint for the frozen trust-region adapter shadow experiment."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_mstcn_frozen_trust_adapter_shadow_20260829_v2"


def _load_base() -> object:
    path = ROOT / "scripts" / "run_p1_mstcn_type_boundary_cascade_shadow_20260829_v1.py"
    spec = importlib.util.spec_from_file_location("p1_mstcn_trust_adapter_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the shared shadow runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    base = _load_base()
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
    base.ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
    args = base._parse_args()  # noqa: SLF001
    if not args.execute:
        print(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": "CHECK_ONLY",
                "config_sha256": base._sha256(base.CONFIG_PATH),  # noqa: SLF001
            }
        )
        return 0
    if args.resume_q4_after_pin_correction:
        raise RuntimeError("v2 has no preregistration recovery amendment")
    base.execute(candidate_module_name="p1_qc.ms_tcn_frozen_trust_adapter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
