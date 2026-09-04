from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from p1_qc.mstcn_group_dro import (
    changed_row_concentration,
    make_group_ids,
    materialize_group_batch,
    robust_bce_from_rows,
)


@dataclass
class Window:
    row_ids: np.ndarray
    window_size: int

    @property
    def valid_length(self) -> int:
        return len(self.row_ids)


def test_sparse_groups_are_merged_deterministically() -> None:
    keys = pd.DataFrame(
        {
            "station": ["A", "A", "B"],
            "layer": [1, 1, 2],
            "time": ["2025-01-01", "2025-01-02", "2025-07-01"],
        }
    )
    ids, receipt = make_group_ids(keys, minimum_rows=2)
    assert receipt["raw_group_count"] == 2
    assert receipt["effective_group_count"] == 2
    assert ids[0] == ids[1]
    assert receipt["effective_rows"]["__SPARSE__"] == 1


def test_group_batch_padding_never_creates_a_group() -> None:
    groups = np.asarray([0, 1, 2], dtype=np.int16)
    values, valid = materialize_group_batch(groups, [Window(np.asarray([0, 2]), 4)])
    assert values.tolist() == [[0, 2, -1, -1]]
    assert valid.tolist() == [[True, True, False, False]]


def test_robust_bce_is_exact_fixed_convex_combination() -> None:
    loss = torch.tensor([[1.0, 3.0, 9.0]], requires_grad=True)
    groups = torch.tensor([[0, 0, 1]])
    valid = torch.tensor([[True, True, True]])
    robust, receipt = robust_bce_from_rows(
        loss, groups, valid, group_count=2, strength=0.5
    )
    pooled = (1.0 + 3.0 + 9.0) / 3.0
    expected = 0.5 * pooled + 0.5 * 9.0
    assert torch.isclose(robust, torch.tensor(expected))
    assert receipt["present_group_count"] == 2
    robust.backward()
    assert loss.grad is not None


def test_changed_row_concentration_reports_station_dominance() -> None:
    result = changed_row_concentration(
        ["A", "A", "B", "B"], [1, 1, 1, 0], [0, 0, 0, 0]
    )
    assert result["changed_rows"] == 3
    assert result["maximum_station_share"] == 2 / 3
