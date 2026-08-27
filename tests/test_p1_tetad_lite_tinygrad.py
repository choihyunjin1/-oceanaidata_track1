from __future__ import annotations

import numpy as np
import pytest
from tinygrad import Tensor
from tinygrad.nn.state import get_parameters

from p1_qc.tetad_lite_tinygrad import (
    TETADLiteConfig,
    TETADLiteTinygrad,
    aggregate_binary_metrics,
    hungarian_assignments,
    interval_set_loss,
    rasterize_intervals,
)


def _small_model() -> TETADLiteTinygrad:
    Tensor.manual_seed(17)
    return TETADLiteTinygrad(
        TETADLiteConfig(
            input_features=3,
            patch_size=4,
            d_model=16,
            num_heads=4,
            ff_multiplier=2,
            num_queries=5,
            max_patches=8,
        )
    )


def test_forward_shapes_and_normalized_ordered_intervals() -> None:
    model = _small_model()
    # Seventeen steps also covers non-divisible patch padding.
    features = Tensor(np.random.default_rng(3).normal(size=(2, 17, 3)).astype(np.float32))
    logits, intervals = model(features)

    assert logits.shape == (2, 5)
    assert intervals.shape == (2, 5, 2)
    interval_values = intervals.numpy()
    assert np.isfinite(logits.numpy()).all()
    assert np.isfinite(interval_values).all()
    assert (interval_values >= 0.0).all()
    assert (interval_values <= 1.0).all()
    assert (interval_values[:, :, 0] <= interval_values[:, :, 1]).all()


def test_hungarian_loss_is_finite_and_backward_reaches_parameters() -> None:
    model = _small_model()
    features = Tensor(np.random.default_rng(5).normal(size=(2, 16, 3)).astype(np.float32))
    logits, intervals = model(features)
    targets = [
        np.asarray([[0.10, 0.30], [0.62, 0.88]], dtype=np.float32),
        np.asarray([[0.35, 0.55]], dtype=np.float32),
    ]

    loss = interval_set_loss(logits, intervals, targets)
    assert np.isfinite(float(loss.total.numpy()))
    assert np.isfinite(float(loss.classification.numpy()))
    assert np.isfinite(float(loss.endpoint_l1.numpy()))
    assert np.isfinite(float(loss.interval_iou.numpy()))
    assert [len(indices[0]) for indices in loss.assignments] == [2, 1]

    loss.total.backward()
    gradients = [parameter.grad for parameter in get_parameters(model) if parameter.grad is not None]
    assert gradients
    assert all(np.isfinite(gradient.numpy()).all() for gradient in gradients)
    assert model.query_embedding.grad is not None


def test_empty_target_batch_has_uniform_backward_path() -> None:
    model = _small_model()
    features = Tensor(np.zeros((1, 8, 3), dtype=np.float32))
    logits, intervals = model(features)
    loss = interval_set_loss(logits, intervals, [np.empty((0, 2), dtype=np.float32)])

    assert float(loss.endpoint_l1.numpy()) == 0.0
    assert float(loss.interval_iou.numpy()) == 0.0
    loss.total.backward()
    assert model.class_head.weight.grad is not None


def test_rasterization_uses_half_open_union_and_one_cell_minimum() -> None:
    mask = rasterize_intervals(
        intervals=np.asarray([[0.10, 0.30], [0.50, 0.50], [0.80, 1.00]]),
        scores=np.asarray([0.9, 0.5, 0.49]),
        length=10,
        threshold=0.5,
    )
    np.testing.assert_array_equal(mask, np.asarray([0, 1, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8))


def test_aggregate_binary_metrics() -> None:
    metrics = aggregate_binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_query_overflow_fails_closed_instead_of_dropping_truth() -> None:
    model = _small_model()
    features = Tensor(np.zeros((1, 16, 3), dtype=np.float32))
    logits, intervals = model(features)
    targets = [
        np.asarray(
            [[0.02 * index, 0.02 * index + 0.01] for index in range(6)],
            dtype=np.float32,
        )
    ]

    with pytest.raises(ValueError, match="exceeds the fixed query budget"):
        hungarian_assignments(logits, intervals, targets)


def test_positive_class_weight_is_supported_and_validated() -> None:
    model = _small_model()
    features = Tensor(np.zeros((1, 16, 3), dtype=np.float32))
    logits, intervals = model(features)
    target = [np.asarray([[0.25, 0.75]], dtype=np.float32)]

    weighted = interval_set_loss(
        logits, intervals, target, positive_class_weight=10.0
    )
    assert np.isfinite(float(weighted.total.numpy()))
    with pytest.raises(ValueError, match="positive_class_weight"):
        interval_set_loss(logits, intervals, target, positive_class_weight=0.0)
