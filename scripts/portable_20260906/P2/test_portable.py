"""Synthetic-only portability, feature firewall and native writer integration tests."""

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "02_code"))
import core  # noqa: E402
import run as entry  # noqa: E402


def observations():
    rows = []
    for hour in range(96):
        for layer in range(1, 9):
            rows.append(
                {
                    "station": "S-ORS",
                    "year": 2024,
                    "time": pd.Timestamp("2024-07-01", tz="Asia/Seoul") + pd.Timedelta(hours=hour),
                    "layer": layer,
                    "depth": layer * 3.01,
                    "nominal_depth": layer * 3,
                    "temp": 20 - layer * 0.2 + np.sin(hour / 5),
                    "psal": 31 + layer * 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_target_firewall_and_mass():
    obs = observations()
    frame, truth = core.public_frame(obs)
    changed = obs.copy()
    changed.loc[changed.layer.isin((2, 3, 4)), ["temp", "psal"]] = -999
    other, _ = core.public_frame(changed)
    assert all(
        np.array_equal(a, b) for a, b in zip(core.arrays(frame), core.arrays(other), strict=True)
    )
    cfg = json.loads((HERE / "config.json").read_text())
    arrays, receipt = core.training_arrays(frame, truth, "v23_blockmask", cfg)
    assert np.isclose(arrays[-1].sum(), len(truth))
    assert receipt["original_rows"] == len(truth) and receipt["augmented_rows"] > 0
    assert arrays[0].shape[1:] == (5, 8) and arrays[2].shape[1] == 11


def test_core_architecture_and_gradient():
    torch.manual_seed(3)
    model = core.make_model("v23_blockmask", 11)
    assert sum(p.numel() for p in model.parameters()) == 4865
    t = torch.randn(3, 5, 8, requires_grad=True)
    mask = torch.ones(3, 5)
    loss = model(t, mask, torch.randn(3, 11)).square()
    penalty = core.observed_temperature_gradient_penalty(loss, t, mask, torch.ones(3))
    (loss.mean() + penalty).backward()
    assert torch.isfinite(penalty)


def test_schema_and_order():
    sample = pd.DataFrame(
        {"station": ["S-ORS"] * 3, "layer": [2, 3, 4], "time": ["2025-09-01T00:00:00+09:00"] * 3}
    )
    answer = sample.assign(temp=[1.0, 2.0, 3.0])
    assert all(entry.validate_output(answer, sample, sample, 3).values())
    with pytest.raises(ValueError):
        entry.validate_output(answer.iloc[::-1], sample, sample, 3)


def test_packaged_imports_only():
    allowed = {
        "__future__",
        "argparse",
        "hashlib",
        "importlib",
        "json",
        "os",
        "sys",
        "time",
        "pathlib",
        "types",
        "typing",
        "numpy",
        "pandas",
        "torch",
        "threadpoolctl",
        "core",
    }
    for path in (HERE / "02_code").glob("*.py"):
        source = path.read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                assert all(n.name.split(".")[0] in allowed for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.module.split(".")[0] in allowed
        assert not any(
            term in source
            for term in (
                "PycharmProjects",
                "bin17",
                "alpha40",
                "router_anchor",
                "gi_spike",
                "run_p2_",
            )
        )


def test_native_writer_and_actual_negative_reads(tmp_path):
    # The audit hook sees real native saves, byte checksums, prohibited loads/CSV/source writes.
    program = f"""
import sys, pathlib
sys.path.insert(0, {str(HERE / "02_code")!r})
import run as m
p=pathlib.Path({str(tmp_path)!r})
m.MODELS=p/'03_model';m.MODELS.mkdir()
m.ANSWERS=p/'05_answer';m.ANSWERS.mkdir()
source=p/'observations.csv';source.write_text('synthetic')
old=p/'old.pt';old.write_bytes(b'synthetic')
counts=m.install_guard('RUN_TRAINING',source)
for i in range(3):
    out=m.MODELS/f'model{{i}}.pt'
    m.torch.save({{'weight':m.torch.ones(2)}},out)
    assert len(m.sha(out))==64
for action in (lambda:old.read_bytes(),lambda:source.write_text('bad'),lambda:m.torch.load(out,weights_only=True),lambda:(p/'test_index.csv').read_text()):
    try:action()
    except PermissionError:pass
    else:raise AssertionError('forbidden operation succeeded')
print('native-guard-pass')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program], text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert "native-guard-pass" in result.stdout


def test_builder_empty_and_rejects_overwrite(tmp_path):
    spec = importlib.util.spec_from_file_location("builder", HERE / "build_package.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    destination = tmp_path / "P2"
    builder.build(destination)
    assert not list((destination / "03_model").iterdir())
    assert not list((destination / "05_answer").iterdir())
    assert not list(destination.rglob("*.csv"))
    with pytest.raises(FileExistsError):
        builder.build(destination)
