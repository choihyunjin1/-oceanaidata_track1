"""Apply the registered full-fidelity promotion gate to the completed screen."""

from __future__ import annotations

import json
from pathlib import Path

from p1_qc.low_fidelity_gate import evaluate_low_fidelity_screen

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "artifacts" / "p1_environment_balanced_replay_screen_20260829_v1" / "run"


def main() -> int:
    result = json.loads((RUN_DIR / "result.json").read_text(encoding="utf-8"))
    gate = {
        "schema_version": "p1.low_fidelity_gate.result.v1",
        "experiment_id": result["experiment_id"],
        **evaluate_low_fidelity_screen(result),
        "claim_limit": "Retrospective low-fidelity screen; no fresh or official performance claim.",
    }
    payload = json.dumps(gate, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    (RUN_DIR.parent / "postrun_gate.json").write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
