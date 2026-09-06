import importlib.util
import json
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "champion_materialize", ROOT / "scripts/p1_champion_reconstruction_20260906_v1/materialize.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return {"path": str(path), "sha256": m.sha(path)}


def authorized(tmp_path):
    started = time.time() - 30
    evidence = {}
    for name, status in (("internal_qa", "PASS"), ("tree_historical", "COMPLETE"),
                         ("tree_training", "COMPLETE"), ("mstcn_training", "TRAINING_COMPLETE_REPLAY_PENDING")):
        evidence[name] = dump(tmp_path / f"{name}.json",
                              {"status": status, "started_unix": started, "started_at": started})
    evidence["internal_qa"] = dump(tmp_path / "internal_qa.json", {
        "status": "PASS", "terminal_result_sha256": evidence["tree_historical"]["sha256"]})
    evidence["tree_training"] = dump(tmp_path / "tree_training.json", {
        "status": "COMPLETE", "historical_result_sha256": evidence["tree_historical"]["sha256"]})
    evidence["tree_replay"] = dump(tmp_path / "tree_replay.json", {
        "status": "PASS", "terminal_result_sha256": evidence["tree_training"]["sha256"]})
    evidence["mstcn_replay"] = dump(tmp_path / "mstcn_replay.json", {
        "status": "PASS", "training_result_sha256": evidence["mstcn_training"]["sha256"]})
    evidence["mstcn_qa"] = dump(tmp_path / "mstcn_qa.json", {
        "status": "PASS", "training_result_sha256": evidence["mstcn_training"]["sha256"],
        "fresh_replay_sha256": evidence["mstcn_replay"]["sha256"]})
    evidence["tree_full_qa"] = dump(tmp_path / "tree_full_qa.json", {
        "status": "PASS", "terminal_result_sha256": evidence["tree_training"]["sha256"]})
    evidence["union_result"] = dump(tmp_path / "union_result.json", {
        "status": "COMPLETE_RETROSPECTIVE_EVALUATION",
        "tree_terminal_sha256": evidence["tree_historical"]["sha256"]})
    for name in ("union_qa", "union_bootstrap_qa"):
        evidence[name] = dump(tmp_path / f"{name}.json", {
            "status": "PASS", "result_sha256": evidence["union_result"]["sha256"]})
    cfg = {"official_materialization_authorized": True, "tree_arm": "router",
           "evidence": evidence, "earliest_started_unix": started}
    path = tmp_path / "decision.json"
    dump(path, cfg)
    return path, cfg


def test_denied_decision_no_official_io_or_output(tmp_path, monkeypatch):
    path = tmp_path / "decision.json"
    dump(path, {"official_materialization_authorized": False})
    monkeypatch.setattr(m, "read_official", lambda *_a, **_k: pytest.fail("early official access"))
    with pytest.raises(PermissionError):
        m.execute("tree", path, tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_complete_exact_evidence_gate(tmp_path):
    path, cfg = authorized(tmp_path)
    assert m.decision(path) == cfg
    (tmp_path / "internal_qa.json").write_text('{"status":"FAIL"}')
    with pytest.raises(ValueError, match="evidence changed"):
        m.decision(path)


def test_replay_cross_run_rejected(tmp_path):
    path, cfg = authorized(tmp_path)
    cfg["evidence"]["tree_replay"] = dump(tmp_path / "tree_replay.json", {
        "status": "PASS", "terminal_result_sha256": "0" * 64})
    dump(path, cfg)
    with pytest.raises(ValueError, match="another training run"):
        m.decision(path)


def test_clock_cannot_be_reset(tmp_path):
    path, cfg = authorized(tmp_path)
    cfg["earliest_started_unix"] += 20
    dump(path, cfg)
    with pytest.raises(ValueError, match="actual training starts"):
        m.decision(path)


def test_unrelated_internal_qa_rejected(tmp_path):
    path, cfg = authorized(tmp_path)
    cfg["evidence"]["internal_qa"] = dump(tmp_path / "internal_qa.json", {
        "status": "PASS", "terminal_result_sha256": "0" * 64})
    dump(path, cfg)
    with pytest.raises(ValueError, match="another historical run"):
        m.decision(path)


@pytest.mark.parametrize("qa", ["union_qa", "union_bootstrap_qa"])
def test_unrelated_union_qa_rejected(tmp_path, qa):
    path, cfg = authorized(tmp_path)
    cfg["evidence"][qa] = dump(tmp_path / f"{qa}.json", {
        "status": "PASS", "result_sha256": "0" * 64})
    dump(path, cfg)
    with pytest.raises(ValueError, match="another evaluation"):
        m.decision(path)


def test_unrelated_mstcn_independent_qa_rejected(tmp_path):
    path, cfg = authorized(tmp_path)
    cfg["evidence"]["mstcn_qa"] = dump(tmp_path / "mstcn_qa.json", {
        "status": "PASS", "training_result_sha256": "0" * 64,
        "fresh_replay_sha256": cfg["evidence"]["mstcn_replay"]["sha256"]})
    dump(path, cfg)
    with pytest.raises(ValueError, match="MS-TCN independent QA"):
        m.decision(path)


def test_component_subdirectories_join_and_csv_roundtrip(tmp_path, monkeypatch):
    original = pd.DataFrame({"station": ["S-ORS"] * 3, "year": [2026] * 3,
                            "layer": [1] * 3, "time": ["2026-01-01 00:00:00+09:00",
                            "2026-01-01 00:10:00+09:00", "2026-01-01 00:20:00+09:00"]})
    stages = tmp_path / "stages"
    for name, column, bits in (("tree", "tree", [1, 0, 0]), ("mstcn", "proposal", [0, 1, 0])):
        directory = stages / name
        directory.mkdir(parents=True)
        frame = original.assign(**{column: bits})
        if name == "mstcn":
            frame = frame.iloc[::-1]
        path = directory / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        dump(directory / f"{name}.json", {"decision_sha256": "synthetic",
                                          "predictions_sha256": m.sha(path)})
    output = tmp_path / "answer"
    output.mkdir()
    monkeypatch.setattr(m, "read_official", lambda *_a, **_k: original.copy())
    report = m.merge_stage({"decision_sha256": "synthetic", "earliest_started_unix": time.time()},
                           tmp_path, stages, output)
    assert report["status"] == "PASS"
    assert report["validator_after"]["positive"] == 2
    assert report["validator_after"]["test_order_match"]
    assert report["composition"]["tree_positive_removed_rows"] == 0
    assert report["csv_sha256"] == m.sha(output / "P1_submission.csv")


def test_official_input_mutation_during_read_rejected(tmp_path, monkeypatch):
    path = tmp_path / "test.csv"
    path.write_text("synthetic input", encoding="utf-8")
    monkeypatch.setattr(m, "TEST_SHA", m.sha(path))

    def mutate(*_args, **_kwargs):
        path.write_text("changed synthetic input", encoding="utf-8")
        return pd.DataFrame()

    monkeypatch.setattr(m.pd, "read_csv", mutate)
    with pytest.raises(ValueError, match="changed during read"):
        m.read_official(tmp_path)
