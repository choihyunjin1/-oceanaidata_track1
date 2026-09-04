"""Independent QA wrapper for P2 v11r1."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import qa_p2_public_sensor_influence_shrink_20260901_v10 as qa  # noqa: E402

qa.EXPERIMENT_ID = "p2_supported_layer_change_coherence_20260901_v11r1"
qa.ARTIFACT = ROOT / "artifacts" / qa.EXPERIMENT_ID
qa.REPORT = ROOT / "reports" / qa.EXPERIMENT_ID
qa.CONFIG = ROOT / "configs" / "experiments" / f"{qa.EXPERIMENT_ID}.json"


if __name__ == "__main__":
    qa.main()
