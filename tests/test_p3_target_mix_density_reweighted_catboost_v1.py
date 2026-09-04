from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p3_target_mix_density_reweighted_catboost_v1.py"
SPEC = importlib.util.spec_from_file_location("p3_density_runner", RUNNER)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _frame(seed: int, rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        rng.normal(size=(rows, len(runner.OBSERVED_FEATURES))),
        columns=runner.OBSERVED_FEATURES,
    )
    frame["station"] = np.resize(np.array(["G-ORS", "I-ORS", "S-ORS"]), rows)
    return frame


def test_grouped_domain_weights_are_deterministic_and_complete() -> None:
    config = json.loads((ROOT / runner.CONFIG_REL).read_text(encoding="utf-8"))
    source = _frame(7, 60)
    target = _frame(11, 60)
    source_groups = np.array([f"s{index // 2}" for index in range(60)], dtype=object)
    target_groups = np.array([f"t{index // 2}" for index in range(60)], dtype=object)
    first, first_receipt = runner._density_weights(
        source=source,
        target=target,
        source_groups=source_groups,
        target_groups=target_groups,
        base_weight=np.ones(60),
        config=config,
    )
    second, second_receipt = runner._density_weights(
        source=source,
        target=target,
        source_groups=source_groups,
        target_groups=target_groups,
        base_weight=np.ones(60),
        config=config,
    )
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert (first > 0).all()
    assert np.isclose(first.mean(), 1.0)
    assert first_receipt["combined_weight_sha256"] == second_receipt["combined_weight_sha256"]
    assert all(row["group_overlap"] == 0 for row in first_receipt["folds"])


def test_runner_has_no_era5_module_import() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "from p3_wave.era5" not in source
    assert "import p3_wave.era5" not in source

