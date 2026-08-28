"""Exact baseline MS-TCN API with an explicit e150 warm-start receipt."""

from __future__ import annotations

from typing import Any

import torch

from p1_qc import ms_tcn_asrf as baseline

ANOMALY_TYPE_COUNT = baseline.ANOMALY_TYPE_COUNT
BOUNDARY_COUNT = baseline.BOUNDARY_COUNT
STANDARD_DILATIONS = baseline.STANDARD_DILATIONS
MSTCNASRFConfig = baseline.MSTCNASRFConfig
MSTCNASRFLossConfig = baseline.MSTCNASRFLossConfig
MSTCNASRFLossOutput = baseline.MSTCNASRFLossOutput
MSTCNASRFOutput = baseline.MSTCNASRFOutput
compute_ms_tcn_asrf_loss = baseline.compute_ms_tcn_asrf_loss


class MSTCNASRF(baseline.MSTCNASRF):
    """Baseline topology used only to isolate the balanced-replay intervention."""

    def initialize_from_baseline_state_dict(
        self, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        self.load_state_dict(state_dict, strict=True)
        return {
            "copied_tensor_count": len(state_dict),
            "trainable_parameter_count": self.trainable_parameter_count,
            "exact_initial_predictor": True,
            "architecture_changed": False,
            "intervention": "training_window_distribution_only",
        }


__all__ = [
    "ANOMALY_TYPE_COUNT",
    "BOUNDARY_COUNT",
    "MSTCNASRF",
    "MSTCNASRFConfig",
    "MSTCNASRFLossConfig",
    "MSTCNASRFLossOutput",
    "MSTCNASRFOutput",
    "STANDARD_DILATIONS",
    "compute_ms_tcn_asrf_loss",
]
