from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from p2_restore import (
    p2_availability_aware_continuous_sparse_copula_20260830_v1 as runner,
)


def _config() -> dict[str, object]:
    return runner.load_config()


def test_sealed_config_uses_metric_aligned_policy_and_no_slice_hard_veto() -> None:
    config = _config()

    assert config["experiment_id"] == runner.EXPERIMENT_ID
    assert tuple(config["dependence"]["stage0_exposed_edges"]) == runner.EXPOSED_EDGES
    assert tuple(config["state"]["continuous_features"]) == runner.STATE_FEATURES
    assert config["classification"] == "HISTORICALLY_EXPOSED_RESEARCH_ONLY"
    assert config["resource_contract"]["outer_dependence_model_fits"] == 3
    assert config["resource_contract"]["inner_selection_fits"] == 0
    assert config["resource_contract"]["hpo_trials"] == 0
    assert config["execution_policy"]["maximum_executions"] == 1
    assert config["execution_policy"]["result_based_retry"] is False
    assert config["execution_policy"]["csv_output_allowed"] is False
    assert config["primary_decision"]["minimum_improved_windows_is_hard_veto"] is False
    assert config["primary_decision"]["worst_season_cap_is_hard_veto"] is False
    assert config["primary_decision"]["all_layers_nonworse_is_hard_veto"] is False
    assert config["correction"]["magnitude_is_promotion_veto"] is False
    canonical = json.dumps(
        config, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == runner.SEALED_CONFIG_CANONICAL_SHA256


def _synthetic_profiles(rows: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(20260830)
    contrast = rng.normal(loc=3.0, scale=0.6, size=rows)
    change12 = rng.normal(scale=0.25, size=rows)
    change24 = 0.5 * change12 + rng.normal(scale=0.2, size=rows)
    temp = contrast + rng.normal(scale=0.3, size=rows)
    psal = 0.2 * temp + rng.normal(size=rows)
    residual_l2 = (0.2 + 0.08 * change12) * temp + rng.normal(scale=0.5, size=rows)
    residual_l3 = (0.15 + 0.05 * change24) * temp + rng.normal(scale=0.6, size=rows)
    residual_l4 = (
        0.1 * temp
        + (0.2 + 0.05 * contrast) * psal
        + 0.15 * change24
        + rng.normal(scale=0.7, size=rows)
    )
    return pd.DataFrame(
        {
            "temp_contrast_signed": temp,
            "psal_contrast_signed": psal,
            "thermal_change_24h_signed": change24,
            "thermal_contrast_abs": contrast,
            "thermal_change_12h_signed": change12,
            "residual_l2": residual_l2,
            "residual_l3": residual_l3,
            "residual_l4": residual_l4,
            "kst_day": [f"2024-05-{index % 30 + 1:02d}" for index in range(rows)],
            "block": np.where(np.arange(rows) % 2 == 0, "block_a", "block_b"),
        }
    )


def test_continuous_sparse_model_uses_fixed_edges_and_missing_ood_are_exact_noop() -> None:
    config = _config()
    profiles = _synthetic_profiles()
    model = runner._fit_continuous_model(profiles, config)

    assert model.edge_coefficients.shape == (7, 4)
    assert set(model.receipt["edge_receipts"]) == set(runner.EXPOSED_EDGES)
    assert model.receipt["ridge"] == 1.0
    assert model.receipt["diagonal_shrinkage"] == 0.8
    query_columns = list(dict.fromkeys(runner.CONDITIONERS + runner.STATE_FEATURES))
    median = profiles[query_columns].median(axis=0).to_dict()
    query = pd.DataFrame([median, median, median])
    query.loc[1, "thermal_change_12h_signed"] = np.nan
    query.loc[2, "thermal_contrast_abs"] = 1e9
    prediction = model.predict(query)

    assert prediction.active.tolist() == [True, False, False]
    assert prediction.missing.tolist() == [False, True, False]
    assert prediction.ood.tolist() == [False, False, True]
    assert np.isfinite(prediction.correction).all()
    assert np.array_equal(prediction.correction[1:], np.zeros((2, 3)))


def _synthetic_scored() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    windows = ("window_a", "window_b", "window_c")
    for window_index, window in enumerate(windows):
        start = pd.Timestamp("2024-05-01", tz="Asia/Seoul") + pd.Timedelta(
            days=30 * window_index
        )
        for day in range(14):
            for layer in (2, 3, 4):
                rows.append(
                    {
                        "window": window,
                        "time": (start + pd.Timedelta(days=day)).tz_convert("UTC"),
                        "layer": layer,
                        "truth": 0.0,
                        "reference": 1.0 + 0.01 * layer,
                        "candidate": 0.9 + 0.01 * layer,
                    }
                )
    return pd.DataFrame(rows)


def test_moving_block_bootstrap_is_deterministic_joint_and_primary_favorable() -> None:
    config = copy.deepcopy(_config())
    config["primary_decision"]["paired_interval"]["replicates"] = 200
    scored = _synthetic_scored()

    first = runner._moving_block_bootstrap(scored, config)
    second = runner._moving_block_bootstrap(scored, config)

    assert first == second
    assert first["block_length_days"] == 7
    assert first["layers_preserved_together_within_day"] is True
    assert first["windows_resampled_separately"] is True
    assert first["ci90_high"] < 0.0
    assert runner._evidence_state(-0.1, first) == "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY"


def test_tail_risk_is_diagnostic_only() -> None:
    diagnostic = runner._tail_risk_diagnostic(_synthetic_scored(), block_length_days=7)

    assert diagnostic["role"] == "DIAGNOSTIC_SENSITIVITY_ONLY_NOT_A_PROMOTION_GATE"
    assert diagnostic["cells"] == 18
    assert diagnostic["positive_part_cvar80_rmse_c"] == 0.0


def _write_synthetic_source(p2_dir: Path) -> bytes:
    p2_dir.mkdir()
    observations = pd.DataFrame(
        [
            {
                "station": "synthetic",
                "year": 2024,
                "layer": 1,
                "time": "2024-05-01T00:00:00+09:00",
                "temp": 10.0,
                "psal": 34.0,
                "depth": 1.0,
                "nominal_depth": 1.0,
            }
        ]
    )
    observations_path = p2_dir / "observations.csv"
    observations.to_csv(observations_path, index=False)
    payload = observations_path.read_bytes()
    for forbidden in (
        "README.md",
        "test_index.csv",
        "sample_submission.csv",
        "baseline_interp.csv",
        "score.py",
        "submission.csv",
        "query_support.json",
    ):
        (p2_dir / forbidden).write_text("must-not-open\n", encoding="utf-8")
    return payload


def test_source_reader_opens_only_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = copy.deepcopy(_config())
    p2_dir = tmp_path / "explicit-p2"
    observations = _write_synthetic_source(p2_dir)
    config["source_contract"]["observations"] = {
        "bytes": len(observations),
        "sha256": hashlib.sha256(observations).hexdigest(),
    }
    original_open = Path.open
    opened: list[str] = []
    source_root = p2_dir.resolve()

    def recording_open(path: Path, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if path.parent.resolve() == source_root:
            opened.append(path.name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    frame, receipt, access = runner._read_training_source(p2_dir, config)

    assert len(frame) == 1
    assert opened == ["observations.csv"]
    assert access.open_counts == {"observations.csv": 1}
    assert receipt["rows"] == 1


def test_exclusive_result_writer_refuses_source_and_overwrite(tmp_path: Path) -> None:
    config = copy.deepcopy(_config())
    p2_dir = tmp_path / "p2"
    p2_dir.mkdir()
    inside = p2_dir / "result.json"
    config["artifact_path"] = str(inside)
    with pytest.raises(runner.ExperimentContractError, match="inside --p2-dir"):
        runner._write_result({"aggregate": True}, inside, p2_dir, config)

    output = tmp_path / "result.json"
    config["artifact_path"] = str(output)
    runner._write_result({"aggregate": True}, output, p2_dir, config)
    assert json.loads(output.read_text(encoding="utf-8")) == {"aggregate": True}
    with pytest.raises(FileExistsError):
        runner._write_result({"aggregate": False}, output, p2_dir, config)
