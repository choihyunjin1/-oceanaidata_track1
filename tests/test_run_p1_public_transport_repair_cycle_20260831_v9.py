from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v9 as cycle  # noqa: E402


def test_duration_bins_are_frozen() -> None:
    assert cycle.duration_bin(48) == "le96"
    assert cycle.duration_bin(96) == "le96"
    assert cycle.duration_bin(192) == "192_to_384"
    assert cycle.duration_bin(384) == "192_to_384"
    assert cycle.duration_bin(519) == "ge519"


def test_family_aware_gate_is_exact() -> None:
    config = cycle.load_contract()
    assert config["transport_family"]["tier_id"] == "HARD_CONDITIONAL_ROUTER"
    assert config["transport_family"]["representation_changed"] is True
    assert config["transport_family"]["routing_discontinuous"] is True
    assert np.isclose(config["decision_policy"]["transport_penalty_points"], 0.3219056897594759)
    assert np.isclose(config["decision_policy"]["minimum_raw_expected_point_delta_inclusive"], 0.33190568975947593)


def test_flat_selector_lcb_uses_only_training_counts() -> None:
    proposal = {"fold": "train", "start": 0, "stop": 4, "model": "mean_shift", "duration_bin": "le96", "month": 4}
    folds = {"train": {"truth": np.array([1, 1, 0, 0]), "incumbent": np.zeros(4, dtype=np.int8)}}
    state = cycle.selector_state("flat", [proposal], folds, 0.0)
    assert state["global_tp"] == 2
    assert state["global_fp"] == 2
    lcb, source = cycle.proposal_lcb(proposal, state, 0.1)
    assert source == "global"
    assert 0.0 < lcb < 0.5


def test_whole_proposal_daily_cap_rejects_oversize() -> None:
    frame = pd.DataFrame({"time": ["2025-01-01T00:00:00Z"] * 1000})
    data = {
        "frame": frame,
        "truth": np.zeros(1000, dtype=np.int8),
        "incumbent": np.zeros(1000, dtype=np.int8),
        "proposals": [{"id": "x", "fold": "q", "model": "mean_shift", "month": 1, "duration_bin": "le96", "window_rows": 6, "start": 0, "stop": 6, "score": 10.0, "score_per_row": 1.0, "start_time": "2025-01-01T00:00:00Z"}],
    }
    state = {"kind": "flat", "global_alpha": 100.0, "global_beta": 1.0}
    prediction, receipt = cycle.apply_selector(data, state, 0.1, {"scope": {"daily_full_surface_fraction_cap": 0.005}, "proposal_policy": {"posterior_precision_lcb_quantile": 0.1}})
    assert prediction.sum() == 0
    assert receipt["decisions"][0]["decision"] == "WHOLE_PROPOSAL_EXCEEDS_DAILY_CAP"


def test_native_handles_numpy_scalars() -> None:
    assert cycle.native({"ok": np.bool_(True), "n": np.int64(2)}) == {"ok": True, "n": 2}
