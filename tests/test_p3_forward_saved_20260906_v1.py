"""Synthetic only: no real models, observations, official rows, fitting or GPU."""

import hashlib
import importlib.util
import io
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

BASE = Path(__file__).resolve().parents[1] / "scripts/portable_20260906/P3_forward_saved_v1"


def module(name):
    spec = importlib.util.spec_from_file_location("saved_test_" + name, BASE / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def recipe():
    return {
        "model": {"single_weight": 0.5, "multi_weight": 0.5},
        "router": {
            "alpha": 10.0,
            "temperature_multiplier": 2.0,
            "strength": 0.5,
            "name": "smooth_medium",
            "active_leads": [12, 18, 24],
        },
        "shrink": {"active_leads": [12, 18, 24], "persistence_weight": 0.2},
    }


@pytest.fixture
def fake_cold(tmp_path):
    b = module("build_package")
    root = tmp_path / "synthetic_cold"
    for d in ("02_code", "03_model", "04_logs", "06_docs"):
        (root / d).mkdir(parents=True)
    code, docs = root / "02_code", root / "06_docs"
    (code / "dummy.py").write_text("# synthetic source only\n", encoding="utf-8")
    (code / "requirements.txt").write_text("# synthetic only\n", encoding="utf-8")
    put(docs / "shrink-provenance.json", {"synthetic": True})
    frozen = {
        "source_files": {"train_wave.csv": "a" * 64},
        "recipe": recipe(),
        "shrink_provenance_sha256": b.sha(docs / "shrink-provenance.json"),
    }
    put(code / "frozen.json", frozen)
    manifest = {p.name: b.sha(p) for p in code.iterdir()}
    put(code / "manifest.json", manifest)
    columns = ["hs_synthetic_" + str(i) for i in range(527)] + ["hmax_" + str(i) for i in range(64)]
    put(root / "04_logs/columns.json", {"columns": columns})
    column_sha = b.sha(root / "04_logs/columns.json")
    p = {
        "status": "PREPARED_SOURCE_ONLY",
        "fits": 0,
        "old_cache_reads": 0,
        "empty_03_model": True,
        "manifest_sha256": b.sha(code / "manifest.json"),
        "files": {"columns.json": column_sha},
    }
    put(docs / "prepare.json", p)
    models = {}
    for name in b.MODELS:
        path = root / "03_model" / name
        path.write_bytes(b"synthetic-not-a-real-model:" + name.encode())
        models["03_model/" + name] = b.sha(path)
    # Metadata only: these historical model files deliberately do not exist.
    models.update({f"03_model/historical_{i}.cbm": "d" * 64 for i in range(10)})
    models.update({f"03_model/prior_{i}.joblib": "e" * 64 for i in range(4)})
    r = {
        "status": "TWELVE_BACKBONE_FIVE_ROUTER_COMPLETE",
        "new_backbone_fits": 12,
        "new_router_fits": 5,
        "old_oof_model_answer_reads": 0,
        "official_rows": 0,
        "variant": "hmax",
        "source_sha256": frozen["source_files"],
        "pid": 101,
        "prepare_sha256": b.sha(docs / "prepare.json"),
        "manifest_sha256": b.sha(code / "manifest.json"),
        "models": models,
        "work": {"columns.json": column_sha, "absent_oof.parquet": "f" * 64},
    }
    put(docs / "training-result.json", r)
    q = {
        "status": "PASS",
        "training_result_sha256": b.sha(docs / "training-result.json"),
        "pid": 102,
        "rows": 103602,
        "oof_keys_probabilities_exact": True,
        "full_probe_exact": True,
        "new_fits": 0,
        "official_rows": 0,
    }
    put(docs / "training-qa.json", q)
    a = {
        "status": "LOCAL_ANSWER_NOT_UPLOADED",
        "pid": 103,
        "rows": 1200,
        "training_result_sha256": b.sha(docs / "training-result.json"),
        "training_qa_sha256": b.sha(docs / "training-qa.json"),
        "sha256": "b" * 64,
        "public_input_sha256": {"test_context.parquet": "c" * 64, "test_index.csv": "d" * 64},
        "sample_rows": 0,
        "hidden_rows": 0,
        "uploads": 0,
        "new_fits": 0,
        "elapsed_seconds": 1000,
    }
    put(docs / "answer-qa.json", a)
    put(docs / "answer-replay-qa.json", {**a, "status": "EXACT_ANSWER_REPLAY_PASS", "pid": 104})
    return b, root, {"dummy.py": manifest["dummy.py"]}


def test_builder_needs_only_three_full_models_and_metadata(fake_cold):
    b, root, pins = fake_cold
    accepted = b.validate_cold(root, pins)
    assert len(accepted["selected_columns"]) == 527
    assert set(accepted["full_models_sha256"]) == set(b.MODELS)
    assert accepted["historical_cache_oof_probe_required"] is False
    assert not list(root.rglob("*.parquet"))
    assert not (root / "05_answer").exists()


@pytest.mark.parametrize(
    "file,key,value",
    [
        ("training-result.json", "new_backbone_fits", 11),
        ("training-result.json", "official_rows", 1),
        ("training-qa.json", "status", "FAIL"),
        ("training-qa.json", "new_fits", 1),
        ("training-qa.json", "pid", 101),
        ("answer-replay-qa.json", "pid", 103),
        ("answer-replay-qa.json", "sha256", "z" * 64),
        ("answer-replay-qa.json", "elapsed_seconds", 21600),
    ],
)
def test_rejects_unaccepted_cold_chain_before_packaging(fake_cold, file, key, value):
    b, root, pins = fake_cold
    path = root / "06_docs" / file
    item = b.read(path)
    item[key] = value
    put(path, item)
    with pytest.raises(ValueError):
        b.validate_cold(root, pins)


def test_rejects_changed_model_and_source(fake_cold):
    b, root, pins = fake_cold
    (root / "03_model/full_single.cbm").write_bytes(b"changed synthetic model")
    with pytest.raises(ValueError, match="model bytes"):
        b.validate_cold(root, pins)
    (root / "02_code/dummy.py").write_text("changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="source manifest"):
        b.validate_cold(root, pins)


def test_rejects_manifest_escape(fake_cold):
    b, root, pins = fake_cold
    manifest = b.read(root / "02_code/manifest.json")
    manifest["../../elsewhere.txt"] = "0" * 64
    put(root / "02_code/manifest.json", manifest)
    with pytest.raises(ValueError, match="escapes"):
        b.validate_cold(root, pins)


def test_synthetic_archive_manifest_and_preflight_without_oof(fake_cold, monkeypatch, tmp_path):
    b, root, pins = fake_cold
    helpers = tmp_path / "synthetic_builder"
    helpers.mkdir()
    put(helpers / "source-pins.json", pins)
    for name in ("infer.py", "README.md"):
        (helpers / name).write_bytes((BASE / name).read_bytes())
    monkeypatch.setattr(b, "HERE", helpers)
    destination = tmp_path / "saved_package"
    receipt = b.build(root, destination)
    assert receipt["models"] == 3 and receipt["official_data_reads"] == 0
    with zipfile.ZipFile(receipt["archive_path"]) as archive:
        assert not any(
            Path(name).suffix in {".csv", ".parquet", ".npz", ".npy", ".log"}
            for name in archive.namelist()
        )
        assert len([name for name in archive.namelist() if name.startswith("03_model/")]) == 3
        extraction = tmp_path / "extracted_synthetic"
        archive.extractall(extraction)
    c = module("infer").preflight(extraction)
    assert c["cold_training_qa_status"] == "PASS"
    assert not (extraction / "04_logs").exists()
    assert c["model_fits"] == 0


def test_postprocessing_rejects_changed_shrink():
    m = module("infer")
    r = recipe()
    m.postprocessing_contract(r)
    r["shrink"]["persistence_weight"] = 0.21
    with pytest.raises(ValueError, match="postprocessing"):
        m.postprocessing_contract(r)


def synthetic_inputs(m, monkeypatch):
    keys = ["case_id", "station", "lead_h"]
    leads = [3, 6, 9, 12, 18, 24]
    cases = np.arange(200)
    context = pd.DataFrame(
        {
            "case_id": np.repeat(cases, 289),
            "station": "SYNTHETIC",
            "step_minute": np.tile(np.arange(-2880, 1, 10), 200),
        }
    )
    index = (
        pd.DataFrame(
            {"case_id": np.repeat(cases, 6), "station": "SYNTHETIC", "lead_h": np.tile(leads, 200)}
        )
        .sample(frac=1, random_state=5)
        .reset_index(drop=True)
    )
    expected = index.copy()
    expected["hs_pred"] = expected.case_id / 100 + expected.lead_h / 100
    payload = expected.to_csv(index=False, lineterminator="\n").encode()
    monkeypatch.setattr(pd, "read_parquet", lambda path, columns: context[columns].copy())
    monkeypatch.setattr(pd, "read_csv", lambda path, usecols: index[usecols].copy())
    monkeypatch.setattr(m, "sha", lambda path: "a" * 64)

    def predict(e, cfg, frame, columns, models):
        rows = frame[["case_id", "station"]].loc[frame.index.repeat(6)].reset_index(drop=True)
        rows["lead_h"] = np.tile(leads, len(frame))
        return rows, rows.case_id.to_numpy() / 100 + rows.lead_h.to_numpy() / 100

    e = SimpleNamespace(BASE_COLUMNS=[], DIRECTION_COLUMNS=[], summarize_context=lambda group: {})
    materializer = SimpleNamespace(KEYS=keys, predict_cases=predict)
    c = {
        "public_input_sha256": {name: "a" * 64 for name in m.PUBLIC_NAMES},
        "expected_answer_sha256": hashlib.sha256(payload).hexdigest(),
        "selected_columns": [],
    }
    return c, e, materializer, payload, index, context


def test_shuffled_index_preserved_case_major_keys_and_lf(monkeypatch, tmp_path):
    m = module("infer")
    original_reader = pd.read_csv
    c, e, materializer, expected, index, _ = synthetic_inputs(m, monkeypatch)
    payload, _ = m.make_answer(c, tmp_path, e, materializer, {}, ())
    assert payload == expected and b"\r\n" not in payload
    # Bypass the monkeypatched reader solely for in-memory synthetic bytes.
    parsed = original_reader(io.BytesIO(payload))
    assert parsed[materializer.KEYS].equals(index)


@pytest.mark.parametrize(
    "failure", ["input_hash", "answer_hash", "grid", "duplicate", "nan", "range"]
)
def test_synthetic_answer_fail_closed(failure, monkeypatch, tmp_path):
    m = module("infer")
    c, e, materializer, _, index, context = synthetic_inputs(m, monkeypatch)
    if failure == "input_hash":
        c["public_input_sha256"]["test_index.csv"] = "b" * 64
    elif failure == "answer_hash":
        c["expected_answer_sha256"] = "b" * 64
    elif failure == "grid":
        context.loc[0, "step_minute"] = -2879
    elif failure == "duplicate":
        index.loc[1] = index.loc[0]
    else:
        prior = materializer.predict_cases

        def bad(*args):
            keys, values = prior(*args)
            values[0] = np.nan if failure == "nan" else 31
            return keys, values

        materializer.predict_cases = bad
    with pytest.raises(ValueError):
        m.make_answer(c, tmp_path, e, materializer, {}, ())


def test_post_read_input_hash_change_rejected(monkeypatch, tmp_path):
    m = module("infer")
    c, e, materializer, *_ = synthetic_inputs(m, monkeypatch)
    sequence = iter(["a" * 64, "a" * 64, "b" * 64, "a" * 64])
    monkeypatch.setattr(m, "sha", lambda path: next(sequence))
    with pytest.raises(ValueError, match="changed during inference"):
        m.make_answer(c, tmp_path, e, materializer, {}, ())


def test_access_guard_has_no_historical_or_other_model_exception(monkeypatch, tmp_path):
    m = module("infer")
    callbacks = []
    monkeypatch.setattr(m.sys, "addaudithook", callbacks.append)
    monkeypatch.setattr(m, "ROOT", tmp_path / "package")
    monkeypatch.setattr(m, "CODE", m.ROOT / "02_code")
    source, output = tmp_path / "source", tmp_path / "output"
    deny = tmp_path / "original_repository"
    monkeypatch.setenv("P3_DENY_REPO", str(deny))
    m.access_guard(source, output)
    audit = callbacks[0]
    for path in (
        m.ROOT / "03_model/full_single.cbm",
        source / "test_index.csv",
        output / "submission.csv",
    ):
        audit("open", (str(path), "rb", 0))
    for path in (
        m.ROOT / "04_logs/oof.parquet",
        m.ROOT / "04_logs/probe.npz",
        m.ROOT / "03_model/old.cbm",
        source / "sample_submission.csv",
        deny / "src/code.py",
    ):
        with pytest.raises(PermissionError):
            audit("open", (str(path), "rb", 0))
    for path in (
        source / "test_index.csv",
        m.ROOT / "03_model/full_single.cbm",
        m.CODE / "infer.py",
    ):
        with pytest.raises(PermissionError):
            audit("open", (str(path), "wb", os.O_WRONLY | os.O_TRUNC))
    with pytest.raises(PermissionError, match="network"):
        audit("socket.connect", ())


def test_supervisor_timeout_uses_tree_termination(monkeypatch, tmp_path):
    m = module("infer")
    monkeypatch.setenv("P3_DATA_DIR", str(tmp_path / "data"))
    waits, killed, commands = [], [], []

    class FakeProcess:
        pid = 12345

        def wait(self, timeout):
            waits.append(timeout)
            raise m.subprocess.TimeoutExpired("synthetic", timeout)

    process = FakeProcess()

    def popen(command, **kwargs):
        commands.append(command)
        return process

    monkeypatch.setattr(m.subprocess, "Popen", popen)
    monkeypatch.setattr(m, "terminate_worker", lambda p: killed.append(p.pid))
    assert m.supervise(tmp_path / "timeout_output") == 124
    assert waits == [600] and killed == [12345]
    assert "-I" in commands[0] and "--worker" in commands[0]
    receipt = m.read(tmp_path / "timeout_output/process-receipt.json")
    assert receipt["timeout_seconds"] == 600 and receipt["new_fits"] == 0


def test_supervisor_preserves_existing_output(monkeypatch, tmp_path):
    m = module("infer")
    monkeypatch.setenv("P3_DATA_DIR", str(tmp_path / "data"))
    with pytest.raises(ValueError, match="new output"):
        m.supervise(tmp_path)


def test_reviewed_source_pins_still_match_source_files():
    b = module("build_package")
    pins = b.read(BASE / "source-pins.json")
    assert len(pins) == 12
    repo = BASE.parents[2]
    for name, expected in pins.items():
        assert b.sha(repo / name) == expected


@pytest.mark.parametrize("variant", ["numeric", "hmax"])
@pytest.mark.parametrize("invalid", [None, "categories", "feature_order", "seed", "router"])
def test_native_model_contract_before_public_reads(monkeypatch, variant, invalid):
    m = module("infer")
    columns = ["x_" + str(i) for i in range(527 if variant == "hmax" else 591)]
    single = SimpleNamespace(
        tree_count_=700,
        random_seed_=20260817,
        get_cat_feature_indices=lambda: [0, 1] if variant == "hmax" else [0],
        feature_names_=["station", "lead_h", "current_hs_for_residual", *columns],
    )
    multi = SimpleNamespace(
        tree_count_=1200,
        random_seed_=20260817,
        get_cat_feature_indices=lambda: [0],
        feature_names_=["station", *columns],
    )
    router = SimpleNamespace(
        config=SimpleNamespace(
            alpha=10.0, temperature_multiplier=2.0, strength=0.5, name="smooth_medium"
        ),
        columns=["hs_current"],
    )
    if invalid == "categories":
        single.get_cat_feature_indices = lambda: [1]
    elif invalid == "feature_order":
        single.feature_names_ = list(reversed(single.feature_names_))
    elif invalid == "seed":
        multi.random_seed_ = 0
    elif invalid == "router":
        router.columns = ["target_hs"]

    class FakeCatBoost:
        def load_model(self, path):
            return single if path.name == "full_single.cbm" else multi

    monkeypatch.setitem(m.sys.modules, "catboost", SimpleNamespace(CatBoostRegressor=FakeCatBoost))
    monkeypatch.setitem(m.sys.modules, "joblib", SimpleNamespace(load=lambda path: router))
    monkeypatch.setattr(m.importlib, "import_module", lambda name: SimpleNamespace(name=name))
    c = {"variant": variant, "selected_columns": columns}
    if invalid is not None:
        with pytest.raises(ValueError):
            m.load_policy(c)
    else:
        assert m.load_policy(c)[3] == (single, multi, router)
