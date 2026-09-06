import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("original_builder", REPO / "scripts/build_p1_original_source_package_20260906_v1.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    path = tmp_path_factory.mktemp("original_package") / "candidate"
    BUILDER.build(REPO, path)
    return path


def test_empty_model_and_no_data(package):
    for directory in ["01_data", "03_model", "04_logs", "05_answer"]:
        assert not list((package / directory).iterdir())
    assert not any(p.suffix in {".csv", ".pt", ".joblib", ".parquet", ".npz"}
                   for p in package.rglob("*") if p.is_file())


def test_manifest_and_notebook(package):
    manifest = json.loads((package / "source-manifest.json").read_text())
    for name, sha in manifest["files"].items():
        assert BUILDER.sha(package / name) == sha
    nbformat.validate(nbformat.read(package / "RUN_ALL.ipynb", as_version=4))


def test_original_recipe(package):
    recipe = json.loads((package / "02_code/tree_recipe.json").read_text())
    assert recipe["O_selection"]["iteration_count"] == 700
    assert recipe["O_selection"]["postprocess"] == {
        "close_gap_rows": 0, "high_threshold": .2, "low_threshold": .1, "minimum_positive_run": 12}
    assert recipe["B_parameters"]["subsample_freq"] == 1
    assert recipe["B_parameters"]["num_leaves"] == 63


def test_extracted_weight_is_original(package):
    original = ast.parse((REPO / "scripts/run_p1_meaningful_learning_curve_generation_v1.py").read_text(encoding="utf-8"))
    copied = ast.parse((package / "02_code/event_weight.py").read_text())
    source = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "_event_day_weight")
    actual = next(n for n in copied.body if isinstance(n, ast.FunctionDef))
    assert ast.dump(source) == ast.dump(actual)
    spec = importlib.util.spec_from_file_location("weight_toy", package / "02_code/event_weight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.DataFrame({"station": ["S-ORS"]*4, "layer": [1]*4,
        "time": pd.date_range("2025-01-01", periods=4, freq="10min", tz="Asia/Seoul")})
    assert np.array_equal(module._event_day_weight(frame, np.array([0, 0, 1, 1])), np.ones(4))


def test_fresh_source_import_and_train_only_boundary(package):
    # Tiny contract check only: no real model fits/data/CUDA initialization.
    code = """
import importlib.util, sys
from pathlib import Path
p = Path(sys.argv[1])
sys.path.insert(0, str(p / '02_code/source/src'))
import p1_qc.pipeline, p1_qc.ms_tcn_asrf, p1_qc.ms_tcn_asrf_data
assert Path(p1_qc.pipeline.__file__).is_relative_to(p)
s = importlib.util.spec_from_file_location('guard', p / '02_code/ms_driver.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
m.verify_sources(p / '02_code/source')
m.install_io_guard(p)
try:
    Path(sys.argv[2]).read_bytes()
except PermissionError:
    print('BOUNDARY_PASS')
else:
    raise AssertionError('outside read permitted')
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", code, str(package), str(REPO / "README.md")],
                            env=env, cwd=package, capture_output=True, text=True, check=True)
    assert "BOUNDARY_PASS" in result.stdout


def test_replay_dispatch_uses_owned_snapshot(package):
    script = (package / "02_code/run.py").read_text()
    assert '03_model/mstcn/02_code/mstcn.py' in script
    assert 'historical_answer_exact' in script
    assert '21600' in script
