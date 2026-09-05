import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "depth_postaudit", ROOT / "scripts/run_p1_depth_contract_postaudit_20260905_v2.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_paired_identical_has_zero_interval():
    frame = pd.DataFrame(
        {
            "time": pd.date_range("2025-04-01", periods=50, freq="D", tz="Asia/Seoul").astype(str),
            "fold": ["q2"] * 50,
            "label": np.arange(50) % 2,
        }
    )
    frame["prediction"] = frame.label
    result = m.paired_block_interval(frame, "prediction", "prediction", replicates=100)
    assert result["ci90"] == [0.0, 0.0]
    assert result["blocks"] == 8


def test_fixed_decoder_seals_and_has_no_official_io():
    source = (ROOT / "scripts/run_p1_depth_contract_postaudit_20260905_v2.py").read_text()
    assert ".to_csv(" not in source
    assert '"test.csv"' not in source
    assert "strength=1.0" in source
    assert "directory.mkdir(exist_ok=False)" in source
