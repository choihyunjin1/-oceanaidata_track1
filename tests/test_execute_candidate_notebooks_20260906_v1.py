import importlib.util
import json
from pathlib import Path

import nbformat
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/execute_candidate_notebooks_20260906_v1.py"
SPEC = importlib.util.spec_from_file_location("executor", SOURCE)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_real_kernel_preserves_source_and_single_attempt(tmp_path):
    (tmp_path / "06_docs").mkdir()
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("assert 2 + 2 == 4")])
    source = tmp_path / "TRAIN.ipynb"
    nbformat.write(nb, source)
    before = source.read_bytes()
    result = m.execute(tmp_path, ["TRAIN"], 60)
    assert result[0]["status"] == "PASS"
    assert source.read_bytes() == before
    receipt = json.loads((tmp_path / "06_docs/executed_notebooks/TRAIN-receipt.json").read_text())
    assert receipt["status"] == "PASS"
    with pytest.raises(FileExistsError):
        m.execute(tmp_path, ["TRAIN"], 60)
