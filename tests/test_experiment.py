from __future__ import annotations

import json

from p1_qc.experiment import RunRecorder, sha256_file, stable_hash


def test_hashes_are_stable(tmp_path) -> None:
    path = tmp_path / "value.txt"
    path.write_text("ocean", encoding="utf-8")
    assert sha256_file(path) == sha256_file(path)
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})


def test_run_recorder_writes_manifest(tmp_path) -> None:
    recorder = RunRecorder("test", {"seed": 7}, root=tmp_path, run_id="run", seed=7)
    recorder.record_json("metrics.json", {"f1": 0.7})
    recorder.finish(score=0.7)
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["score"] == 0.7
