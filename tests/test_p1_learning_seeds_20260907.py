"""Portable five-seed reduction and source-only training adapter checks."""
import ast
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from package_p1_learning_seeds_20260907 import BASE, adapt  # noqa: E402


def test_five_seed_exact_arithmetic():
    code = adapt((BASE / "02_code/run.py").read_text(encoding="utf-8"), "B5")
    node = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == "mean_b")
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "mean", "exec"), scope)
    x = np.random.default_rng(77).random((5, 100))
    np.testing.assert_array_equal(scope["mean_b"](x), (x[:3].mean(axis=0)*3+x[3]+x[4])/5)


def test_five_seed_fit_counts_and_no_wallcap():
    code = adapt((BASE / "02_code/run.py").read_text(encoding="utf-8"), "B5")
    assert '"fits_max": 6' in code and '"fits": 9' in code
    assert 'receipt["fits"] == 6' in code and "n_jobs=4" in code
    assert "timeout=max(" not in code


def test_o_slow_does_not_modify_b_reduction():
    code = adapt((BASE / "02_code/run.py").read_text(encoding="utf-8"), "O_slow")
    assert "SEEDS = [20260813, 20260829, 20260847]" in code
    assert code.count("np.mean(") == 3 and '"fits": 7' in code
