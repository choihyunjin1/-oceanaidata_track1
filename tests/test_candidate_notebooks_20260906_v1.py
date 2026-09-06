import importlib.util
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

SOURCE = Path(__file__).resolve().parents[1] / "scripts/create_candidate_notebooks_20260906_v1.py"
SPEC = importlib.util.spec_from_file_location("notebooks", SOURCE)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_synthetic_notebook_runs_top_to_bottom(tmp_path, monkeypatch):
    (tmp_path / "README.md").write_text("synthetic package")
    (tmp_path / "04_logs").mkdir()
    (tmp_path / "run.py").write_text(
        "from pathlib import Path\n"
        "assert not any(Path('04_logs').iterdir()), 'Cold workdir must remain empty'\n"
        "print('synthetic stage only')"
    )
    monkeypatch.setenv("P2_DATA_DIR", str(tmp_path))
    nb = m.notebook("P2", "TRAIN", [["run.py", "train"]])
    nbformat.validate(nb)
    result = NotebookClient(nb, timeout=60, resources={"metadata": {"path": str(tmp_path)}}).execute()
    assert all(cell.execution_count is not None for cell in result.cells if cell.cell_type == "code")
    assert len(list((tmp_path / "06_docs/notebook_launches").glob("*.log"))) == 1


@pytest.mark.parametrize("commands", [[], [["../run.py"]], [["run.py", "bad\nargument"]]])
def test_invalid_commands(commands):
    with pytest.raises(ValueError):
        m.notebook("P1", "TRAIN", commands)
