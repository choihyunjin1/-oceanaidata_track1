from __future__ import annotations

from dataclasses import replace

from p1_qc.config import P1QCConfig


def test_resume_explicitly_forces_causal_feature_mode() -> None:
    raw = P1QCConfig()
    corrected = replace(raw, mode="causal", features=replace(raw.features, mode="causal"))
    assert corrected.mode == "causal"
    assert corrected.features.mode == "causal"
