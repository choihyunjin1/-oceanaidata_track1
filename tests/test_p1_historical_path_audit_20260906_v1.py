import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/p1_historical_path_audit_20260906_v1.py"
SPEC = importlib.util.spec_from_file_location("p1_history_audit", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_router_exhaustive_cells_and_bits():
    for station, layer, o, b in itertools.product(
        ["G-ORS", "I-ORS", "S-ORS", "unknown"], range(1, 9), [0, 1], [0, 1]
    ):
        router, gi = MODULE.compose_tree(
            [station], [layer], [o], [b], add_cells=MODULE.ADD_CELLS,
            remove_cells=MODULE.REMOVE_CELLS,
        )
        add = o == 1 and b == 0 and (station, layer) in MODULE.ADD_CELLS
        remove = o == 0 and b == 1 and (station, layer) in MODULE.REMOVE_CELLS
        assert router.tolist() == [int((b or add) and not remove)]
        assert gi.tolist() == [int(b or add)]


def test_gi_general_not_two_fixed_rows():
    for size in [0, 1, 2, 7]:
        result = MODULE.compose_mstcn_spike([0]*size, [1]*size, [0]*size, ["spike"]*size)
        assert result.tolist() == [1]*size


def test_gi_types_and_existing_predictions():
    result = MODULE.compose_mstcn_spike([1, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0],
                                     ["", "", "noise", "spike"])
    assert result.tolist() == [1, 1, 0, 1]


@pytest.mark.parametrize("bad", [[2], [np.nan], [[0, 1]]])
def test_binary_rejects(bad):
    with pytest.raises(ValueError):
        MODULE.binary(bad)


def test_length_contract():
    with pytest.raises(ValueError):
        MODULE.compose_tree(["G-ORS"], [1], [0, 1], [1], add_cells=set(), remove_cells=set())


def test_metric_manual_counts():
    result = MODULE.metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert (result["tp"], result["fp"], result["fn"], result["f1"]) == (1, 1, 1, 0.5)
