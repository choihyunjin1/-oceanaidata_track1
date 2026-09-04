from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/materialize_p3_lead_continuous_score_priority_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p3_lead_continuous_score_priority_deployment_20260901_v1.json"


def _load_runner():
    spec = importlib.util.spec_from_file_location("p3_lead_score_priority_materializer_test", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _official_keys() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": np.repeat([f"c{index:03d}" for index in range(200)], 6),
            "station": np.repeat(["G-ORS", "I-ORS", "S-ORS", "G-ORS"] * 50, 6),
            "lead_h": np.tile([3, 6, 9, 12, 18, 24], 200),
        }
    )


def test_sealed_config_records_score_priority_and_fresh_harm() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "SEALED_BEFORE_OFFICIAL_MATERIALIZATION"
    assert config["decision"]["estimated_point_gain_vs_clean_incumbent"] == pytest.approx(
        0.06646686239290145
    )
    assert config["stability_disclosure"][
        "fresh_delta_candidate_minus_incumbent_rmse_m"
    ] == pytest.approx(0.022617090405150586)
    assert config["lineage"]["organizer_distributed_data_only"] is True
    assert config["lineage"]["kma_era5_v21_v81_predictions_or_metrics_used"] == 0


def test_validate_key_surface_accepts_exact_1200_and_rejects_duplicate() -> None:
    runner = _load_runner()
    keys = _official_keys()
    runner._validate_key_surface(keys, "synthetic")
    broken = keys.copy()
    broken.loc[1, runner.KEYS] = broken.loc[0, runner.KEYS].to_numpy()
    with pytest.raises(runner.ContractError):
        runner._validate_key_surface(broken, "broken")


def test_build_model_frame_preserves_key_order_and_uses_only_causal_features() -> None:
    runner = _load_runner()
    keys = _official_keys()
    cases = keys[["case_id", "station"]].drop_duplicates().reset_index(drop=True)
    features = cases.copy()
    names = ("hs_delta_3h", "hs_std_6h", "wspd_delta_3h", "caph_delta_6h")
    for offset, name in enumerate(names):
        features[name] = np.arange(len(features), dtype=float) + offset
    baseline = keys.copy()
    baseline["hs_pred"] = 2.0
    incumbent = keys.copy()
    incumbent["hs_pred"] = 1.8
    frame = runner._build_model_frame(keys, features, baseline, incumbent, names)
    assert frame[runner.KEYS].equals(keys)
    assert list(frame.columns) == [*runner.KEYS, *names, "persistence", "final_prediction"]
    assert np.allclose(frame["persistence"], 2.0)
    assert np.allclose(frame["final_prediction"], 1.8)


def test_array_hash_is_dtype_and_shape_sensitive() -> None:
    runner = _load_runner()
    values = np.asarray([1.0, 2.0], dtype=np.float64)
    assert runner.array_sha256(values) != runner.array_sha256(values.astype(np.float32))
    assert runner.array_sha256(values) != runner.array_sha256(values.reshape(1, 2))


def test_title_and_summary_disclose_low_stability() -> None:
    runner = _load_runner()
    assert "Clean" in runner.TITLE
    assert "+0.022617" in runner.SUMMARY
    assert "안정성이 낮" in runner.SUMMARY


def test_materializer_has_no_upload_or_network_dependency() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "requests" not in source
    assert "selenium" not in source
    assert "playwright" not in source
    assert "upload_count\": 0" in source
