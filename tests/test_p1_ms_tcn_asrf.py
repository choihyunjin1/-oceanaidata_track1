from __future__ import annotations

import pytest
import torch

from p1_qc.ms_tcn_asrf import (
    MSTCNASRF,
    STANDARD_DILATIONS,
    MSTCNASRFConfig,
    MSTCNASRFLossConfig,
    MSTCNASRFOutput,
    compute_ms_tcn_asrf_loss,
)


def _tiny_config(*, input_feature_count: int = 6) -> MSTCNASRFConfig:
    return MSTCNASRFConfig(
        input_feature_count=input_feature_count,
        width=8,
        dropout=0.0,
    )


def _targets(batch: int, time: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    event = torch.randint(0, 2, (batch, time), dtype=torch.float32)
    boundary = torch.randint(0, 2, (batch, time, 2), dtype=torch.float32)
    anomaly_type = torch.randint(0, 2, (batch, time, 5), dtype=torch.float32)
    return event, boundary, anomaly_type


def test_config_dilations_receptive_fields_and_validation() -> None:
    config = _tiny_config()
    config.validate()
    assert config.generator_dilations == STANDARD_DILATIONS
    assert config.generator_dual_dilations[0] == (1, 512)
    assert config.generator_dual_dilations[-1] == (512, 1)
    assert config.prediction_generator_receptive_field == 3969
    assert config.refinement_stage_receptive_field == 2047
    assert config.final_receptive_field == 10107

    with pytest.raises(ValueError, match="1..512"):
        MSTCNASRFConfig(input_feature_count=2, generator_dilations=(1, 2, 4)).validate()
    with pytest.raises(ValueError, match="exactly three"):
        MSTCNASRFConfig(input_feature_count=2, refinement_stages=2).validate()
    with pytest.raises(ValueError, match="at least four"):
        MSTCNASRFConfig(input_feature_count=2, width=0).validate()
    with pytest.raises(ValueError, match="stage_weights"):
        MSTCNASRFLossConfig(stage_weights=(1.0, -1.0, 1.0, 1.0)).validate()
    with pytest.raises(ValueError, match="at least one"):
        MSTCNASRFLossConfig(auxiliary_positive_weight_cap=0.5).validate()


def test_dual_generator_fuses_raw_convolution_branches_before_relu() -> None:
    model = MSTCNASRF(_tiny_config())
    layer = model.prediction_generator.layers[0]
    with torch.no_grad():
        layer.left.weight.zero_()
        layer.left.bias.fill_(-1.0)
        layer.right.weight.zero_()
        layer.right.bias.fill_(-1.0)
        layer.fuse.weight.fill_(-1.0 / 16.0)
        layer.fuse.bias.zero_()
    values = torch.zeros(1, 8, 5)
    channel_mask = torch.ones(1, 1, 5, dtype=torch.bool)

    output = layer(values, channel_mask)

    # Each of 16 raw branch channels contributes (+1/16) after fusion, so the
    # fused ReLU produces one.  ReLU on each branch before fusion would instead
    # erase the deliberately negative branch values and produce zero.
    torch.testing.assert_close(output, torch.ones_like(output))


def test_output_shapes_bfloat16_compatibility_and_finite_gradients() -> None:
    torch.manual_seed(20260827)
    config = _tiny_config()
    model = MSTCNASRF(config)
    values = torch.randn(2, 19, config.input_feature_count)
    output = model(values)
    assert len(output.stage_logits) == 4
    assert all(logits.shape == (2, 19) for logits in output.stage_logits)
    assert output.final_logits.shape == (2, 19)
    assert output.boundary_logits.shape == (2, 19, 2)
    assert output.type_logits.shape == (2, 19, 5)
    assert model.trainable_parameter_count == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert model.trainable_parameter_count > 0

    event, boundary, anomaly_type = _targets(2, 19)
    valid = torch.ones(2, 19, dtype=torch.bool)
    loss = compute_ms_tcn_asrf_loss(
        output,
        event,
        boundary,
        anomaly_type,
        valid_mask=valid,
    )
    assert torch.isfinite(loss.total)
    loss.total.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)

    bf16_model = MSTCNASRF(config).to(dtype=torch.bfloat16)
    bf16_output = bf16_model(torch.randn(1, 11, config.input_feature_count).bfloat16())
    assert bf16_output.final_logits.dtype is torch.bfloat16
    assert bool(torch.isfinite(bf16_output.final_logits.float()).all())


def test_padding_mask_zeroes_every_stage_and_makes_valid_outputs_pad_invariant() -> None:
    torch.manual_seed(20260828)
    config = _tiny_config()
    model = MSTCNASRF(config).eval()
    valid_length = 7
    padded_length = 15
    prefix = torch.randn(1, valid_length, config.input_feature_count)
    first_padding = torch.randn(1, padded_length - valid_length, config.input_feature_count)
    changed_padding = torch.full_like(first_padding, float("nan"))
    valid = torch.zeros(1, padded_length, dtype=torch.bool)
    valid[:, :valid_length] = True

    first = model(torch.cat((prefix, first_padding), dim=1), valid_mask=valid)
    changed = model(torch.cat((prefix, changed_padding), dim=1), valid_mask=valid)
    cropped = model(prefix, valid_mask=torch.ones(1, valid_length, dtype=torch.bool))

    for first_stage, changed_stage, cropped_stage in zip(
        first.stage_logits,
        changed.stage_logits,
        cropped.stage_logits,
        strict=True,
    ):
        torch.testing.assert_close(first_stage[:, :valid_length], changed_stage[:, :valid_length])
        torch.testing.assert_close(first_stage[:, :valid_length], cropped_stage)
        assert torch.count_nonzero(first_stage[:, valid_length:]) == 0
        assert torch.count_nonzero(changed_stage[:, valid_length:]) == 0
    torch.testing.assert_close(
        first.boundary_logits[:, :valid_length],
        changed.boundary_logits[:, :valid_length],
    )
    torch.testing.assert_close(first.boundary_logits[:, :valid_length], cropped.boundary_logits)
    torch.testing.assert_close(
        first.type_logits[:, :valid_length],
        changed.type_logits[:, :valid_length],
    )
    torch.testing.assert_close(first.type_logits[:, :valid_length], cropped.type_logits)
    assert torch.count_nonzero(first.boundary_logits[:, valid_length:]) == 0
    assert torch.count_nonzero(first.type_logits[:, valid_length:]) == 0


def test_forward_rejects_invalid_padding_masks() -> None:
    model = MSTCNASRF(_tiny_config())
    values = torch.randn(2, 5, 6)
    with pytest.raises(ValueError, match="shape"):
        model(values, valid_mask=torch.ones(2, 4, dtype=torch.bool))
    with pytest.raises(TypeError, match="boolean"):
        model(values, valid_mask=torch.ones(2, 5))


def test_every_padded_or_type_masked_value_is_excluded_from_loss() -> None:
    torch.manual_seed(4)
    batch, time = 2, 7
    stage_logits = tuple(torch.randn(batch, time) for _ in range(4))
    boundary_logits = torch.randn(batch, time, 2)
    type_logits = torch.randn(batch, time, 5)
    output = MSTCNASRFOutput(stage_logits, boundary_logits, type_logits)
    event, boundary, anomaly_type = _targets(batch, time)
    valid = torch.tensor(
        [
            [True, True, True, True, False, False, False],
            [True, True, False, False, False, False, False],
        ]
    )
    type_mask = valid.clone()
    type_mask[0, 2] = False
    baseline = compute_ms_tcn_asrf_loss(
        output,
        event,
        boundary,
        anomaly_type,
        valid_mask=valid,
        type_mask=type_mask,
    )

    altered_stages = tuple(
        logits.masked_fill(~valid, float("nan")) for logits in output.stage_logits
    )
    altered_boundary_logits = boundary_logits.masked_fill(
        ~valid.unsqueeze(-1).expand_as(boundary_logits), float("nan")
    )
    effective_type = valid & type_mask
    altered_type_logits = type_logits.masked_fill(
        ~effective_type.unsqueeze(-1).expand_as(type_logits), float("nan")
    )
    altered_event = event.masked_fill(~valid, float("nan"))
    altered_boundary = boundary.masked_fill(~valid.unsqueeze(-1).expand_as(boundary), float("nan"))
    altered_type = anomaly_type.masked_fill(
        ~effective_type.unsqueeze(-1).expand_as(anomaly_type), float("nan")
    )
    altered = compute_ms_tcn_asrf_loss(
        MSTCNASRFOutput(altered_stages, altered_boundary_logits, altered_type_logits),
        altered_event,
        altered_boundary,
        altered_type,
        valid_mask=valid,
        type_mask=type_mask,
    )
    torch.testing.assert_close(altered.total, baseline.total)
    torch.testing.assert_close(altered.event, baseline.event)
    torch.testing.assert_close(altered.temporal_smoothing, baseline.temporal_smoothing)
    torch.testing.assert_close(altered.boundary, baseline.boundary)
    torch.testing.assert_close(altered.anomaly_type, baseline.anomaly_type)


def test_default_type_supervision_ignores_normal_rows_and_balances_auxiliary_bce() -> None:
    batch, time = 1, 5
    stages = tuple(torch.zeros(batch, time) for _ in range(4))
    boundary_logits = torch.zeros(batch, time, 2)
    type_logits = torch.zeros(batch, time, 5)
    output = MSTCNASRFOutput(stages, boundary_logits, type_logits)
    event = torch.tensor([[0.0, 1.0, 1.0, 0.0, 0.0]])
    boundary = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]])
    anomaly_type = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    valid = torch.ones(batch, time, dtype=torch.bool)
    config = MSTCNASRFLossConfig(
        event_dice_weight=0.0,
        smoothing_weight=0.0,
        auxiliary_positive_weight_cap=20.0,
    )
    baseline = compute_ms_tcn_asrf_loss(
        output,
        event,
        boundary,
        anomaly_type,
        valid_mask=valid,
        config=config,
    )

    changed_type_logits = type_logits.clone()
    changed_type_logits[:, [0, 3, 4]] = float("nan")
    changed_type_targets = anomaly_type.clone()
    changed_type_targets[:, [0, 3, 4]] = float("nan")
    changed = compute_ms_tcn_asrf_loss(
        MSTCNASRFOutput(stages, boundary_logits, changed_type_logits),
        event,
        boundary,
        changed_type_targets,
        valid_mask=valid,
        config=config,
    )
    torch.testing.assert_close(changed.anomaly_type, baseline.anomaly_type)

    unweighted = compute_ms_tcn_asrf_loss(
        output,
        event,
        boundary,
        anomaly_type,
        valid_mask=valid,
        config=MSTCNASRFLossConfig(
            event_dice_weight=0.0,
            smoothing_weight=0.0,
            auxiliary_positive_weight_cap=None,
        ),
    )
    assert baseline.boundary > unweighted.boundary
    assert baseline.anomaly_type > unweighted.anomaly_type


def test_final_stage_weight_controls_final_stage_gradient_and_value() -> None:
    batch, time = 1, 6
    stages = tuple(torch.zeros(batch, time, requires_grad=True) for _ in range(4))
    boundary_logits = torch.zeros(batch, time, 2, requires_grad=True)
    type_logits = torch.zeros(batch, time, 5, requires_grad=True)
    output = MSTCNASRFOutput(stages, boundary_logits, type_logits)
    event = torch.tensor([[0.0, 1.0, 0.0, 1.0, 1.0, 0.0]])
    boundary = torch.zeros(batch, time, 2)
    anomaly_type = torch.zeros(batch, time, 5)
    valid = torch.ones(batch, time, dtype=torch.bool)
    final_only = MSTCNASRFLossConfig(
        stage_weights=(0.0, 0.0, 0.0, 1.0),
        smoothing_weight=0.0,
        boundary_weight=0.0,
        type_weight=0.0,
    )
    loss = compute_ms_tcn_asrf_loss(
        output,
        event,
        boundary,
        anomaly_type,
        valid_mask=valid,
        config=final_only,
    )
    loss.total.backward()
    assert all(logits.grad is not None for logits in stages)
    assert all(torch.count_nonzero(logits.grad) == 0 for logits in stages[:-1])
    assert torch.count_nonzero(stages[-1].grad) > 0

    changed_stages = stages[:-1] + (torch.full((batch, time), 3.0),)
    changed = compute_ms_tcn_asrf_loss(
        MSTCNASRFOutput(changed_stages, boundary_logits.detach(), type_logits.detach()),
        event,
        boundary,
        anomaly_type,
        valid_mask=valid,
        config=final_only,
    )
    assert not torch.isclose(changed.total, loss.total.detach())


def test_loss_requires_boolean_nonempty_valid_mask_and_aligned_stage_weights() -> None:
    output = MSTCNASRFOutput(
        tuple(torch.zeros(1, 3) for _ in range(4)),
        torch.zeros(1, 3, 2),
        torch.zeros(1, 3, 5),
    )
    event, boundary, anomaly_type = _targets(1, 3)
    with pytest.raises(TypeError, match="boolean"):
        compute_ms_tcn_asrf_loss(
            output,
            event,
            boundary,
            anomaly_type,
            valid_mask=torch.ones(1, 3),
        )
    with pytest.raises(ValueError, match="at least one"):
        compute_ms_tcn_asrf_loss(
            output,
            event,
            boundary,
            anomaly_type,
            valid_mask=torch.zeros(1, 3, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="align"):
        compute_ms_tcn_asrf_loss(
            output,
            event,
            boundary,
            anomaly_type,
            valid_mask=torch.ones(1, 3, dtype=torch.bool),
            config=MSTCNASRFLossConfig(stage_weights=(1.0, 1.0)),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        MSTCNASRFLossConfig(event_positive_weight=0.0).validate()


def test_event_positive_weight_increases_positive_target_logit_gradient() -> None:
    def gradient_magnitude(positive_weight: float | None) -> float:
        stages = tuple(torch.zeros(1, 1, requires_grad=True) for _ in range(4))
        output = MSTCNASRFOutput(
            stages,
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 5),
        )
        loss = compute_ms_tcn_asrf_loss(
            output,
            torch.ones(1, 1),
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 5),
            valid_mask=torch.ones(1, 1, dtype=torch.bool),
            config=MSTCNASRFLossConfig(
                stage_weights=(0.0, 0.0, 0.0, 1.0),
                event_bce_weight=1.0,
                event_positive_weight=positive_weight,
                event_dice_weight=0.0,
                smoothing_weight=0.0,
                boundary_weight=0.0,
                type_weight=0.0,
            ),
        )
        loss.total.backward()
        assert stages[-1].grad is not None
        return float(stages[-1].grad.abs().item())

    unweighted = gradient_magnitude(None)
    weighted = gradient_magnitude(4.0)
    assert weighted == pytest.approx(4.0 * unweighted)
