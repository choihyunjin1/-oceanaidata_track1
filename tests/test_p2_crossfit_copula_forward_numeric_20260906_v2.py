"""Synthetic technical amendment and exact input/reuse guards, no contest inputs."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_p2_crossfit_copula_forward_numeric_20260906_v2 as m  # noqa: E402


def fixture():
    key = np.array(["a", "b", "c", "d"])
    c = np.array([1., 2., 3., 4.])
    x = np.arange(44, dtype=float).reshape(4, 11)
    x[1, 3] = np.nan
    selected = np.array([False, False, True, True])
    return key, c, x, selected


def test_output_roundoff_allowed_without_mutation():
    keys, c, x, selected = fixture()
    changed = c.copy()
    changed[0] = np.nextafter(c[0], np.inf)
    before = changed.copy()
    receipt = m.verify_numeric_invariance(keys, keys, c, c, x, x, selected, c, changed)
    assert receipt["output_different_rows"] == 1
    assert np.array_equal(before, changed)
    assert not receipt["prediction_values_modified"]


@pytest.mark.parametrize("kind", ["key", "baseline", "feature", "nan", "large_output"])
def test_roundoff_does_not_relax_inputs_or_large_errors(kind):
    keys, c, x, selected = fixture()
    ak, ac, ax, prediction = keys.copy(), c.copy(), x.copy(), c.copy()
    if kind == "key":
        ak[[0, 1]] = ak[[1, 0]]
    elif kind == "baseline":
        ac[0] = np.nextafter(ac[0], np.inf)
    elif kind == "feature":
        ax[0, 1] = np.nextafter(ax[0, 1], np.inf)
    elif kind == "nan":
        ax[1, 3] = 0
    else:
        prediction[0] += 1e-9
    with pytest.raises(AssertionError):
        m.verify_numeric_invariance(keys, ak, c, ac, x, ax, selected, c, prediction)


def test_large_missingness_batches_and_analytic_provenance():
    rng = np.random.default_rng(20260906)
    x = rng.normal(size=(1801, 11))
    residual = x[:, 0] * .4 + rng.normal(size=len(x)) * .1
    x[:99, 5] = np.nan
    fitted = m.profile.fit_copula(x, residual)
    assert all(m.covariance_checks(fitted, x, residual).values())
    selected = np.arange(len(x)) >= 877
    ax = x.copy()
    ax[selected, 5] = np.nan
    c = np.full(len(x), 12.)
    natural = c + m.profile.predict_copula(fitted, x)
    altered = c + m.profile.predict_copula(fitted, ax)
    keys = np.arange(len(x)).astype(str)
    m.verify_numeric_invariance(keys, keys, c, c, x, ax, selected, natural, altered)
    assert np.array_equal(m.profile.predict_copula(fitted, x[~selected]),
                          m.profile.predict_copula(fitted, ax[~selected]))
    with pytest.raises(AssertionError):
        m.covariance_checks({**fitted, "covariance": fitted["covariance"] + .01}, x, residual)


def test_budget_and_empty_outage_contract():
    cfg, _, _ = m.settings()
    assert cfg["new_backbone_fits"] + cfg["new_copula_fits"] == 48
    assert cfg["reused_backbone_fits"] + cfg["reused_copula_fit_saves"] == 40
    assert cfg["previous_execution_seconds"] + cfg["remaining_execution_cap_seconds"] == 5400
    assert cfg["outside_outage_output_atol_celsius"] == 1e-12
    assert m.metric(np.array([]), np.array([]))["status"] == "NOT_ESTIMABLE_NO_ROWS"


def test_native_new_writer_and_exact_old_allowlist(tmp_path):
    code = """
import os,sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import run_p2_crossfit_copula_forward_numeric_20260906_v2 as m
root=Path(sys.argv[2]);m.ROOT=root;m.OUT=root/'new';m.OUT.mkdir()
m.old.OUT=root/'old';m.old.OUT.mkdir();m.old.REPORT=root/'oldreports';m.old.REPORT.mkdir()
m.SEAL=root/'seal.json';source=root/'source';source.mkdir()
(source/'observations.csv').write_text('synthetic')
allowed=m.old.OUT/'allowed.pt';denied=m.old.OUT/'denied.pt'
m.torch.save({'x':m.torch.ones(1)},allowed);m.torch.save({'x':m.torch.ones(1)},denied)
m.SEAL.write_text(json.dumps({'parent_inventory':{'old/allowed.pt':'metadata'}}))
os.environ['P2_DATA_DIR']=str(source);m.install_guard()
assert m.torch.load(allowed,weights_only=True)['x'].item()==1
for seed in (1,2,3):
    path=m.OUT/f'{seed}.pt';m.torch.save({'x':m.torch.tensor([seed])},path)
    assert len(m.base.file_hash(path))==64
    assert m.torch.load(path,weights_only=True)['x'].item()==seed
for path,mode in [(denied,'rb'),(allowed,'wb'),(source/'observations.csv','w'),(source/'test_index.csv','rb')]:
    try:open(path,mode)
    except PermissionError:pass
    else:raise AssertionError('guard did not reject')
print(json.dumps({'status':'PASS'}))
"""
    output = subprocess.run([sys.executable, "-c", code, str(ROOT / "scripts"), str(tmp_path)],
                            text=True, capture_output=True, check=True, timeout=40)
    assert json.loads(output.stdout)["status"] == "PASS"
