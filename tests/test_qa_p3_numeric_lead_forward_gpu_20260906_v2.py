import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "numeric_qa", ROOT / "scripts/qa_p3_numeric_lead_forward_gpu_20260906_v2.py"
)
qa = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qa)


def test_independent_sse_pooled_not_group_mean():
    result = qa.independent_metrics([0, 0, 0], [0, 3, 4], [0, 0, 0])
    assert result["control_sse_m2"] == 25
    assert result["control_rmse_m"] == np.sqrt(25 / 3)
    assert result["delta_rmse_m"] < 0


@pytest.mark.parametrize("bad", [[], [np.nan], [np.inf]])
def test_invalid_evaluation_rejected(bad):
    with pytest.raises(ValueError):
        qa.independent_metrics(bad, bad, bad)
