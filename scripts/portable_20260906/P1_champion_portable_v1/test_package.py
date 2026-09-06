"""No-fit, no-real-data tests of portable scope, exact selection, and archive boundaries."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


runtime = module("_champion_portable_test_runtime", HERE / "run.py")
builder = module("_champion_portable_test_builder", HERE / "build_package.py")
tree = module("_champion_portable_test_tree", ROOT / "scripts/p1_champion_reconstruction_20260906_v1/tree.py")
composition = module("_champion_portable_composition", ROOT / "scripts/p1_champion_reconstruction_20260906_v1/composition.py")


def toy_frame():
    time = pd.date_range("2024-01-01", "2025-12-10", freq="D", tz="Asia/Seoul")
    n = len(time)
    frame = pd.DataFrame({"station": ["toy"] * n, "year": time.year, "layer": [1] * n,
                          "time": time.astype(str), "label": (np.arange(n) % 3 == 0).astype(int),
                          "temp": np.sin(np.arange(n) / 10), "psal": np.full(n, 32.0),
                          "depth": np.full(n, 10.0)})
    extra = pd.DataFrame({"station": ["boundary"] * 3, "year": [2025] * 3, "layer": [1] * 3,
        "time": ["2025-06-20 23:50:00+09:00", "2025-06-21 00:00:00+09:00", "2025-06-21 00:10:00+09:00"],
        "label": [1, 1, 1], "temp": [1., 2., 3.], "psal": [32.] * 3, "depth": [10.] * 3})
    frame = pd.concat([frame, extra]).sort_values(["station", "layer", "time"], kind="stable").reset_index(drop=True)
    frame["row_id"] = np.arange(len(frame))
    return frame


def test_q4_only_selection_masks_identical_to_original_all_fold_construction():
    frame = toy_frame()
    original = tree.config()
    short = {**original, "folds": [original["folds"][-1]]}
    full_masks, full_support = tree.split_masks(frame, original)
    short_masks, short_support = tree.split_masks(frame, short)
    for phase in ("inner", "outer"):
        for full, part in zip(full_masks[("2025_q4", phase)], short_masks[("2025_q4", phase)], strict=True):
            np.testing.assert_array_equal(full, part)
        assert next(s for s in full_support if s["fold"] == "2025_q4" and s["stage"] == phase) == next(
            s for s in short_support if s["stage"] == phase)
    tm, vm = full_masks[("2025_q4", "inner")]
    assert not tm[frame.station.eq("boundary")].any()  # whole crossing run is excluded
    assert not (tm & vm).any()


def test_inner_selection_is_same_algorithm_and_ignores_unrelated_partitions():
    frame = toy_frame()
    cfg = tree.config()
    _tm, vm = tree.split_masks(frame, cfg)[0][("2025_q4", "inner")]
    inner = frame.loc[vm].reset_index(drop=True)
    n = len(inner)
    probabilities = {"O": np.linspace(.01, .99, n), "B": np.linspace(.8, .02, n)}
    # Ask the canonical rule function for its exact schema; no observations beyond inner.
    rules = tree.core.rule_masks(inner, tree.core.stats_fit(inner))
    selected = tree.select_policy(inner, probabilities, rules, cfg)
    changed = frame.copy()
    changed.loc[~vm, ["temp", "psal", "label"]] = 99999
    other = tree.select_policy(changed.loc[vm].reset_index(drop=True), probabilities, rules, cfg)
    assert selected == other
    assert tree.predict_policy(inner, probabilities, rules, cfg, selected)["union"].shape == (n,)


def test_active_features_exact80_trainstats_and_label_invariance():
    frame = toy_frame().iloc[:50].reset_index(drop=True)
    cfg = tree.config()
    stats = tree.core.stats_fit(frame)
    actual = tree.base_features(frame, stats, cfg)
    changed = frame.assign(label=1 - frame.label, row_id=-10)
    sentinel = tree.base_features(changed, stats, cfg)
    pd.testing.assert_frame_equal(actual.frame, sentinel.frame)
    assert len(actual.feature_columns) == 80
    assert "label" not in actual.feature_columns


def test_actual_model_budget_and_original_seed_parameters():
    cfg = tree.config()
    assert [("O", cfg["O_seed"])] + [("B", s) for s in cfg["B_seeds"]] == [
        ("O", 20260813), ("B", 20260813), ("B", 20260829), ("B", 20260847)]
    assert tree.parameters(cfg, "O", cfg["O_seed"])["n_estimators"] == 700
    for seed in cfg["B_seeds"]:
        assert tree.parameters(cfg, "B", seed)["n_estimators"] == 700
        assert tree.parameters(cfg, "B", seed)["n_jobs"] == 4
    assert 2 * (1 + len(cfg["B_seeds"])) + 3 == 11


def test_zip_preserves_all_empty_user_folders(tmp_path):
    source = tmp_path / "source"
    for name in ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"):
        (source / name).mkdir(parents=True)
    (source / "02_code/check.txt").write_text("synthetic")
    archive = tmp_path / "source.zip"
    builder.archive(source, archive)
    extracted = tmp_path / "new_external_extract"
    with zipfile.ZipFile(archive) as z:
        z.extractall(extracted)
    assert all((extracted / name).is_dir() for name in ("03_model", "04_logs", "05_answer"))
    assert list((extracted / "03_model").iterdir()) == []


@pytest.mark.parametrize("kind,dirty,consumed,expected", [
    ("cold", False, False, True), ("saved", False, False, False),
    ("cold", True, False, False), ("cold", False, True, False)])
def test_empty_model_and_consumed_attempt_gate(tmp_path, monkeypatch, kind, dirty, consumed, expected):
    models, docs = tmp_path / "03_model", tmp_path / "06_docs"
    models.mkdir()
    docs.mkdir()
    if dirty:
        (models / "old_model.txt").write_text("synthetic")
    if consumed:
        (docs / "cold-start.json").write_text("{}")
    monkeypatch.setattr(runtime, "MODELS", models)
    monkeypatch.setattr(runtime, "DOCS", docs)
    if expected:
        runtime.empty_training_gate({"kind": kind})
    else:
        with pytest.raises((RuntimeError, FileExistsError)):
            runtime.empty_training_gate({"kind": kind})


@pytest.mark.parametrize("kind,original,replay,fresh,exact,expected", [
    ("saved", True, False, True, True, "PASS"), ("saved", False, False, True, True, "FAIL"),
    ("saved", False, True, True, True, "FAIL"), ("saved", True, True, False, True, "FAIL"),
    ("cold", False, True, True, True, "PASS"), ("cold", False, True, True, False, "FAIL")])
def test_saved_mismatch_never_leaves_pass_receipt(kind, original, replay, fresh, exact, expected):
    assert runtime.answer_status(kind, original, replay=replay, fresh_pid=fresh, byte_exact=exact) == expected


def test_multirow_key_safe_union_is_original_implementation():
    left = pd.DataFrame({"station": ["x"] * 3, "year": [2025] * 3, "layer": [1] * 3,
                         "time": ["2025-01-01 00:00", "2025-01-01 00:10", "2025-01-01 00:20"],
                         "tree": [0, 1, 0]})
    right = left[list(composition.KEYS)].copy()
    right["proposal"] = [1, 0, 0]
    result, receipt = composition.combine(left, right.iloc[::-1])
    assert result.label.tolist() == [1, 1, 0]
    assert receipt["tree_positive_removed_rows"] == 0
    with pytest.raises(ValueError, match="key sets differ"):
        composition.combine(left, right.iloc[:-1])


def test_manifest_tamper_fails_before_model_or_official_access(tmp_path, monkeypatch):
    code = tmp_path / "02_code"
    code.mkdir()
    frozen = {"kind": "cold", "tree_arm": "union", "budget": {"total_fits": 11}, "packages": {}}
    (code / "frozen.json").write_text(json.dumps(frozen))
    manifest = {"kind": "cold", "files": {"02_code/frozen.json": runtime.sha(code / "frozen.json")}}
    (tmp_path / "package-manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(runtime, "PACKAGE", tmp_path)
    monkeypatch.setattr(runtime, "CODE", code)
    assert runtime.checked() == frozen
    (code / "frozen.json").write_text("{}")
    with pytest.raises(ValueError, match="package source/model changed"):
        runtime.checked()


def test_isolated_help_and_explicit_data_error_from_unrelated_cwd(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "P1_DATA_DIR"}
    result = subprocess.run([sys.executable, "-I", str(HERE / "run.py"), "--help"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0
    result = subprocess.run([sys.executable, "-I", str(HERE / "run.py"), "infer"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and "set P1_DATA_DIR" in result.stderr


def test_no_old_cli_or_official_training_path_and_frozen_ms_math():
    source = (HERE / "run.py").read_text()
    assert "tree.fit_four(inner, valid" in source and "tree.fit_four(frame, probe" in source
    assert "tree.worker(" not in source and "tree.launch(" not in source
    assert "high_threshold=0.8, low_threshold=0.4" in source
    assert "snap_radius=12, minimum_rows=19, maximum_rows=None" in source
    assert "batch_size=64" in source
    assert "np.mean(np.stack([getattr(p, name) for p in predictions]), axis=0).astype(np.float32)" in source
    assert "predict_policy(frame, predictions, rules" in source
    assert "taskkill" in source and '"/T", "/F"' in source
    assert "empty_training_gate(frozen)" in source
    assert "old_answer_values_read" in source


def test_saved_ms_math_matches_original_predict_frame_three_distinct_seeds(monkeypatch, tmp_path):
    ms = module("_champion_portable_original_ms", ROOT / "scripts/p1_champion_reconstruction_20260906_v1/mstcn.py")
    monkeypatch.setitem(sys.modules, "mstcn", ms)
    bundle = namedtuple("Bundle", "row_probability boundary_probability type_probability")
    keys = pd.DataFrame({"station": ["toy"] * 3, "year": [2025] * 3, "layer": [1] * 3,
                         "time": ["2025-01-01 00:00", "2025-01-01 00:10", "2025-01-01 00:20"]})
    encoded = SimpleNamespace(layout="toy")
    seen = []

    def decode(probability, boundary, layout, **kwargs):
        seen.append(kwargs)
        assert layout == "toy" and probability.shape == boundary.shape == (3,)
        return (probability > .5).astype(np.int8)

    source = SimpleNamespace(_all_windows=lambda *_: [1], PredictionBundle=bundle,
        predict_encoded=lambda seed, *a, **k: bundle(
            np.asarray([.1, .9, .3], dtype=np.float32) + ms.SEEDS.index(seed) * .02,
            np.asarray([.9, .1, .8], dtype=np.float32), np.full((3, 5), .4, np.float32)),
        _decoder_row_probability=lambda mean, _cfg: mean.row_probability * (.9 + mean.type_probability[:, 0] * .1),
        decode_long_event_segments=decode)
    monkeypatch.setattr(ms, "verify_owned", lambda *_: {})
    monkeypatch.setattr(ms, "sha", lambda *_: "synthetic")
    monkeypatch.setattr(ms, "read_json", lambda p: {"status": "PASS", "training_result_sha256": "synthetic"})
    monkeypatch.setattr(ms, "load_source", lambda *_: (source, {}))
    monkeypatch.setattr(ms, "configure_device", lambda *_: (SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)), "fake"))
    monkeypatch.setattr(ms, "fresh_features", lambda *_: (SimpleNamespace(keys=keys), [], {}))
    monkeypatch.setattr(ms, "encoder_from_receipt", lambda *_: object())
    monkeypatch.setattr(ms, "encode_with_saved", lambda *_: encoded)
    monkeypatch.setattr(ms, "load_model", lambda _d, seed, *_: seed)
    monkeypatch.setattr(runtime, "read", lambda *_: {})
    oldkeys, oldbits, _mean = ms.predict_frame(keys, model_dir=tmp_path / "03_model")
    newkeys, newbits = runtime.ms_predict(keys)
    pd.testing.assert_frame_equal(oldkeys, newkeys)
    np.testing.assert_array_equal(oldbits, newbits)
    assert seen[0] == seen[1] == {"high_threshold": .8, "low_threshold": .4,
                                "snap_radius": 12, "minimum_rows": 19, "maximum_rows": None}
