from __future__ import annotations

import torch

from p1_qc.ms_tcn_asrf import MSTCNASRF as BaselineMSTCN
from p1_qc.ms_tcn_asrf import MSTCNASRFConfig
from p1_qc.ms_tcn_frozen_trust_adapter import MSTCNASRF as TrustAdapterMSTCN
from p1_qc.ms_tcn_frozen_trust_adapter import compute_ms_tcn_asrf_loss


def _tiny_config() -> MSTCNASRFConfig:
    dilations = tuple(1 << exponent for exponent in range(10))
    return MSTCNASRFConfig(
        input_feature_count=6,
        width=8,
        generator_dilations=dilations,
        refinement_dilations=dilations,
        refinement_stages=3,
        dropout=0.0,
    )


def test_adapter_warm_start_is_exact_and_baseline_is_frozen() -> None:
    torch.manual_seed(31)
    config = _tiny_config()
    baseline = BaselineMSTCN(config).eval()
    adapter = TrustAdapterMSTCN(config).eval()
    receipt = adapter.initialize_from_baseline_state_dict(baseline.state_dict())
    values = torch.randn(2, 27, 6)
    valid = torch.ones(2, 27, dtype=torch.bool)
    valid[1, 22:] = False
    expected = baseline(values, valid_mask=valid)
    observed = adapter(values, valid_mask=valid)

    torch.testing.assert_close(expected.final_logits, observed.final_logits)
    torch.testing.assert_close(expected.boundary_logits, observed.boundary_logits)
    torch.testing.assert_close(expected.type_logits, observed.type_logits)
    assert receipt["baseline_frozen"] is True
    assert all(not parameter.requires_grad for parameter in adapter.baseline.parameters())
    assert adapter.trainable_parameter_count > 0


def test_adapter_gradient_and_row_trust_region() -> None:
    torch.manual_seed(37)
    config = _tiny_config()
    baseline = BaselineMSTCN(config).eval()
    adapter = TrustAdapterMSTCN(config)
    adapter.initialize_from_baseline_state_dict(baseline.state_dict())
    values = torch.randn(2, 35, 6)
    valid = torch.ones(2, 35, dtype=torch.bool)
    events = torch.zeros(2, 35)
    events[:, 7:25] = 1.0
    boundaries = torch.zeros(2, 35, 2)
    boundaries[:, 7, 0] = 1.0
    boundaries[:, 24, 1] = 1.0
    types = torch.zeros(2, 35, 5)
    types[:, 7:25, 4] = 1.0

    initial = adapter(values, valid_mask=valid)
    loss = compute_ms_tcn_asrf_loss(
        initial,
        events,
        boundaries,
        types,
        valid_mask=valid,
    )
    loss.total.backward()
    assert adapter.adapter_stages[0].row_delta_head.weight.grad is not None
    assert float(adapter.adapter_stages[0].row_delta_head.weight.grad.abs().sum()) > 0.0
    assert all(parameter.grad is None for parameter in adapter.baseline.parameters())

    with torch.no_grad():
        for stage in adapter.adapter_stages:
            stage.row_delta_head.weight.fill_(100.0)
            stage.row_delta_head.bias.fill_(100.0)
    changed = adapter(values, valid_mask=valid)
    baseline_logits = baseline(values, valid_mask=valid).final_logits
    maximum_delta = (changed.final_logits - baseline_logits).abs().max().detach()
    assert float(maximum_delta) <= 0.750001
