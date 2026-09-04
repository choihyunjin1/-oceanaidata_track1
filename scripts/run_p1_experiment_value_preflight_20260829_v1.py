"""Evaluate a P1 experiment proposal before allocating training compute."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from p1_qc.experiment_value_preflight import evaluate_experiment_value

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "configs" / "experiments" / "p1_experiment_value_registry_20260829_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("proposal", type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    proposal_path = args.proposal if args.proposal.is_absolute() else ROOT / args.proposal
    registry_path = args.registry if args.registry.is_absolute() else ROOT / args.registry
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    result = {
        "schema_version": "p1.experiment_value_preflight.result.v1",
        "experiment_id": proposal["experiment_id"],
        **evaluate_experiment_value(proposal, registry).as_dict(),
        "proposal": str(proposal_path.relative_to(ROOT)),
        "registry": str(registry_path.relative_to(ROOT)),
    }
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["decision"] == "PASS_TO_LOW_FIDELITY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
