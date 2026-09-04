from __future__ import annotations

import torch

from p1_qc.ms_tcn_asrf import MSTCNASRF as BaselineMSTCN
from p1_qc.ms_tcn_asrf import MSTCNASRFConfig
from p1_qc.ms_tcn_environment_balanced_replay import MSTCNASRF as ReplayMSTCN


def _config() -> MSTCNASRFConfig:
    return MSTCNASRFConfig(
        input_feature_count=5,
        width=8,
        dropout=0.0,
    )


def test_warm_start_is_exact_and_all_parameters_remain_trainable() -> None:
    torch.manual_seed(7)
    baseline = BaselineMSTCN(_config()).eval()
    replay = ReplayMSTCN(_config()).eval()
    receipt = replay.initialize_from_baseline_state_dict(baseline.state_dict())
    values = torch.randn(2, 13, 5)
    valid = torch.ones(2, 13, dtype=torch.bool)
    with torch.no_grad():
        expected = baseline(values, valid_mask=valid)
        observed = replay(values, valid_mask=valid)
    assert torch.equal(expected.final_logits, observed.final_logits)
    assert torch.equal(expected.boundary_logits, observed.boundary_logits)
    assert torch.equal(expected.type_logits, observed.type_logits)
    assert receipt["architecture_changed"] is False
    assert replay.trainable_parameter_count == sum(p.numel() for p in replay.parameters())
