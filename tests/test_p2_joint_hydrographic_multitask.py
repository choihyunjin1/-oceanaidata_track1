from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from p2_restore.joint_hydrographic_multitask import (
    JointHydrographicNormalizer,
    JointHydrographicTCN,
    build_joint_hydrographic_panel,
    layer4_only_ablation,
    materialize_joint_chunks,
    stage_a_prefix_times,
)

ROOT = Path(__file__).resolve().parents[1]


def _observations(rows: int = 20) -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="Asia/Seoul")
    records: list[dict[str, object]] = []
    for number, time in enumerate(times):
        for layer in range(1, 9):
            records.append(
                {
                    "station": "S-ORS",
                    "year": int(time.year),
                    "layer": layer,
                    "time": time.isoformat(),
                    "temp": 15.0 + 0.4 * layer + np.sin(number / 5),
                    "psal": 31.0 + 0.05 * layer + np.cos(number / 7) * 0.1,
                    "depth": float(layer * 3),
                    "nominal_depth": float(layer * 3),
                }
            )
    return pd.DataFrame.from_records(records)


def test_target_layer_mutation_never_changes_inputs() -> None:
    source = _observations()
    original = build_joint_hydrographic_panel(source)
    changed = source.copy()
    target = changed["layer"].isin((2, 3, 4))
    changed.loc[target, "temp"] += 1000.0
    changed.loc[target, "psal"] -= 1000.0
    rebuilt = build_joint_hydrographic_panel(changed)
    np.testing.assert_array_equal(original.inputs, rebuilt.inputs)
    assert not np.array_equal(original.target_temperature, rebuilt.target_temperature)
    assert not np.array_equal(original.target_salinity, rebuilt.target_salinity)
    assert not any(name in original.input_names for name in ("temp_2", "psal_4"))


def test_joint_mask_requires_temperature_and_salinity() -> None:
    source = _observations()
    chosen_time = source["time"].iloc[0]
    source.loc[source["time"].eq(chosen_time) & source["layer"].eq(4), "psal"] = np.nan
    panel = build_joint_hydrographic_panel(source)
    assert panel.joint_target_mask[0].tolist() == [True, True, False]


def test_reference_prefix_mask_is_independent_of_salinity_baseline() -> None:
    source = _observations(12)
    chosen_time = source["time"].iloc[0]
    public = source["layer"].isin((1, 5, 6, 7, 8))
    source.loc[source["time"].eq(chosen_time) & public, "psal"] = np.nan
    panel = build_joint_hydrographic_panel(source)
    assert panel.reference_target_mask[0].all()
    assert not panel.joint_target_mask[0].any()
    prefix = stage_a_prefix_times(
        panel,
        outer_start=pd.Timestamp("2024-02-01", tz="Asia/Seoul"),
        embargo_days=7,
        fraction=1.0,
    )
    assert prefix.equals(panel.times)


def test_normalizer_and_chunks_use_selected_training_rows() -> None:
    panel = build_joint_hydrographic_panel(_observations(600))
    selected = np.zeros(len(panel.times), dtype=bool)
    selected[:550] = True
    normalizer = JointHydrographicNormalizer.fit(panel, selected)
    inputs, targets, mask, bounds = materialize_joint_chunks(
        panel,
        normalizer,
        selected,
        minimum_joint_values=24,
    )
    assert inputs.ndim == 3
    assert targets.shape[-2:] == (3, 2)
    assert mask.shape == targets.shape
    assert bounds
    for chunk, (start, stop) in enumerate(bounds):
        width = stop - start
        expected = np.repeat(
            (panel.joint_target_mask[start:stop] & selected[start:stop, None])[:, :, None],
            2,
            axis=2,
        )
        np.testing.assert_array_equal(mask[chunk, :width].numpy(), expected)
        assert not bool(mask[chunk, width:].any())


def test_joint_model_output_and_loss_are_finite() -> None:
    torch.manual_seed(7)
    model = JointHydrographicTCN(12, hidden=16, dilations=(1, 2), dropout=0.0)
    inputs = torch.randn(2, 24, 12)
    targets = torch.randn(2, 24, 3, 2)
    mask = torch.ones_like(targets, dtype=torch.bool)
    prediction = model(inputs)
    assert prediction.shape == targets.shape
    loss = model.training_loss(inputs, targets, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_layer4_ablation_preserves_layers_2_and_3_exactly() -> None:
    reference = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]])
    multitask = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
    result = layer4_only_ablation(reference, multitask)
    np.testing.assert_array_equal(result[:, :2], reference[:, :2])
    np.testing.assert_array_equal(result[:, 2], multitask[:, 2])


def test_design_and_pure_model_structure_match() -> None:
    design = json.loads(
        (
            ROOT / "configs/experiments/p2_joint_hydrographic_multitask_layer4_v2_design.json"
        ).read_text(encoding="utf-8")
    )
    architecture = design["hypothesis"]["architecture"]
    training = design["hypothesis"]["training"]
    loss = design["hypothesis"]["loss"]
    assert architecture["shared_width"] == 160
    assert architecture["dilations"] == [1, 2, 4, 8, 16, 32]
    assert architecture["output_dimensions"] == 6
    assert architecture["expected_public_input_channels"] == 54
    assert training["epochs"] == 28
    assert training["fixed_seed_ids"] == [20260823, 20260824, 20260825]
    assert loss["vertical_first_difference_consistency_weight"] == 0.25
    model = JointHydrographicTCN(54)
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_021_602
    assert (
        design["static_resource_estimate"]["trainable_parameter_count_at_54_input_channels"]
        == 1_021_602
    )
