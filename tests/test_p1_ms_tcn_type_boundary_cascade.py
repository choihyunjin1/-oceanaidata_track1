from __future__ import annotations

import torch

from p1_qc.ms_tcn_asrf import MSTCNASRF as BaselineMSTCN
from p1_qc.ms_tcn_asrf import MSTCNASRFConfig
from p1_qc.ms_tcn_type_boundary_cascade import (
    MSTCNASRF as CascadeMSTCN,
)
from p1_qc.ms_tcn_type_boundary_cascade import compute_ms_tcn_asrf_loss


def _tiny_config() -> MSTCNASRFConfig:
    return MSTCNASRFConfig(
        input_feature_count=6,
        width=8,
        generator_dilations=tuple(1 << exponent for exponent in range(10)),
        refinement_dilations=tuple(1 << exponent for exponent in range(10)),
        refinement_stages=3,
        dropout=0.0,
    )


def test_cascade_shapes_and_masks() -> None:
    model = CascadeMSTCN(_tiny_config()).eval()
    values = torch.randn(2, 17, 6)
    valid = torch.ones(2, 17, dtype=torch.bool)
    valid[1, 13:] = False
    output = model(values, valid_mask=valid)

    assert len(output.stage_logits) == 4
    assert len(output.stage_boundary_logits) == 4
    assert len(output.stage_type_logits) == 4
    assert output.final_logits.shape == (2, 17)
    assert output.boundary_logits.shape == (2, 17, 2)
    assert output.type_logits.shape == (2, 17, 5)
    assert torch.equal(output.final_logits[1, 13:], torch.zeros(4))
    assert torch.equal(output.boundary_logits[1, 13:], torch.zeros(4, 2))
    assert torch.equal(output.type_logits[1, 13:], torch.zeros(4, 5))


def test_warm_start_exactly_embeds_baseline_predictions() -> None:
    torch.manual_seed(17)
    config = _tiny_config()
    baseline = BaselineMSTCN(config).eval()
    cascade = CascadeMSTCN(config).eval()
    receipt = cascade.initialize_from_baseline_state_dict(baseline.state_dict())
    values = torch.randn(2, 23, 6)
    valid = torch.ones(2, 23, dtype=torch.bool)
    valid[1, 19:] = False

    expected = baseline(values, valid_mask=valid)
    observed = cascade(values, valid_mask=valid)
    assert receipt["exact_initial_row_predictor"] is True
    for left, right in zip(expected.stage_logits, observed.stage_logits, strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    torch.testing.assert_close(expected.boundary_logits, observed.boundary_logits)
    torch.testing.assert_close(expected.type_logits, observed.type_logits)


def test_new_pathways_receive_gradient_under_deep_supervision() -> None:
    torch.manual_seed(23)
    config = _tiny_config()
    baseline = BaselineMSTCN(config)
    cascade = CascadeMSTCN(config)
    cascade.initialize_from_baseline_state_dict(baseline.state_dict())
    values = torch.randn(2, 31, 6)
    valid = torch.ones(2, 31, dtype=torch.bool)
    events = torch.zeros(2, 31)
    events[:, 8:19] = 1.0
    boundaries = torch.zeros(2, 31, 2)
    boundaries[:, 8, 0] = 1.0
    boundaries[:, 18, 1] = 1.0
    types = torch.zeros(2, 31, 5)
    types[:, 8:19, 3] = 1.0

    output = cascade(values, valid_mask=valid)
    loss = compute_ms_tcn_asrf_loss(
        output,
        events,
        boundaries,
        types,
        valid_mask=valid,
    )
    loss.total.backward()

    stage = cascade.refinement_stages[0]
    assert stage.stem.weight.grad is not None
    assert bool(torch.isfinite(stage.stem.weight.grad).all())
    assert float(stage.stem.weight.grad[:, 1:].abs().sum()) > 0.0
    assert stage.boundary_delta_head.weight.grad is not None
    assert float(stage.boundary_delta_head.weight.grad.abs().sum()) > 0.0
    assert stage.type_delta_head.weight.grad is not None
    assert float(stage.type_delta_head.weight.grad.abs().sum()) > 0.0
    assert cascade.trainable_parameter_count > baseline.trainable_parameter_count
