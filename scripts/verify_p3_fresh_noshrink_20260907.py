"""Verify frozen packaged no-shrink inference using completed fresh-cold models.

No fitting, tuning, uploads, archive edits, or historical-output overwrites.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser()
    for name in ["fresh", "package", "data", "out", "report"]:
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    fresh, package, data, out, report = (
        getattr(args, name).resolve() for name in ["fresh", "package", "data", "out", "report"]
    )
    assert not out.exists() and not report.exists(), "Use a new disjoint audit directory"
    assert not out.is_relative_to(fresh) and not out.is_relative_to(package)
    expected_base = "2015b38750d357630d5b2e9eee32807d2961ce752e35eb2b454ee16e579dda56"
    expected_final = "70761affca4d3fc6f1d24ae53467e5b185b23465926ebb4b851f0300872cddbd"
    source_zip = package / "SOURCE_ONLY.zip"
    source_hash = sha(source_zip)
    assert source_hash == "c3aed055873bed601397073be4610fb06fa085021122f3656d65c1939faa6e78"
    saved_zip_hash = sha(package / "SAVED_MODELS.zip")
    terminal = read(fresh.parent / "terminal_result.json")
    cold = next(item for item in terminal["runs"] if item["run"] == "fresh_cold_2")
    assert cold["run_kind"] == "FRESH_COLD" and cold["new_production_fits"] == 41
    assert cold["reused_backbone_fits"] == 0 and cold["answer_replay"] == "EXACT_REPLAY_PASS"
    assert sha(fresh / "05_answer/submission.csv") == cold["answer_sha256"] == expected_base
    training = read(fresh / "06_docs/cpudet-training.json")
    train_qa = read(fresh / "06_docs/cpudet-training-qa.json")
    assert train_qa["status"] == "PASS"
    assert train_qa["training_sha256"] == sha(fresh / "06_docs/cpudet-training.json")
    assert training["backbone_fits"] == 36 and training["router_fits"] == 5
    original_models = {name: sha(fresh / name) for name in training["models"]}
    assert original_models == training["models"]
    out.mkdir(parents=True)
    with zipfile.ZipFile(source_zip) as archive:
        for member in archive.infolist():
            assert (out / member.filename).resolve().is_relative_to(out)
        archive.extractall(out)
    pins = read(out / "02_code/noshrink-source-manifest.json")
    assert all(sha(out / name) == digest for name, digest in pins.items())
    # Only completed fresh models and their receipts/column metadata are adopted.
    # No historical answer, context parquet, OOF predictions or caches are copied.
    copied = list(original_models) + [
        "06_docs/cpudet-training.json", "06_docs/cpudet-training-qa.json", "04_logs/columns.json"
    ]
    for name in copied:
        target = out / name
        assert target.resolve().is_relative_to(out) and not target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fresh / name, target)
        assert sha(target) == sha(fresh / name)
    env = os.environ.copy()
    env.update(P3_DATA_DIR=str(data), P3_DENY_REPO=str(Path(__file__).resolve().parents[1]),
               PYTHONDONTWRITEBYTECODE="1")
    stages = []
    start = time.monotonic()
    for stage in ["adopt-trained", "infer", "replay"]:
        began = time.monotonic()
        with (out / (stage + ".stdout.log")).open("x", encoding="utf-8") as stdout:
            with (out / (stage + ".stderr.log")).open("x", encoding="utf-8") as stderr:
                run = subprocess.run([sys.executable, str(out / "02_code/noshrink.py"), stage],
                                     cwd=out, env=env, stdout=stdout, stderr=stderr, check=False)
        stages.append({"stage": stage, "exit_code": run.returncode,
                       "seconds": time.monotonic() - began})
        if run.returncode:
            save(report, {"status": "TECHNICAL_FAILURE", "stages": stages,
                          "output_root": str(out), "new_fits": 0, "uploads": 0})
            raise RuntimeError(f"{stage} failed; inspect preserved logs, do not restart")
    answer = read(out / "06_docs/noshrink-answer.json")
    replay = read(out / "06_docs/noshrink-replay.json")
    import numpy as np
    import pandas as pd

    result_path = out / "05_answer/submission_p3_numeric_cpudet_noshrink.csv"
    frame = pd.read_csv(result_path)
    keys = ["case_id", "station", "lead_h"]
    index = pd.read_csv(data / "test_index.csv", usecols=keys)
    checks = {
        "schema": list(frame.columns) == keys + ["hs_pred"],
        "rows": len(frame) == 1200,
        "key_order": frame[keys].equals(index),
        "unique_keys": not frame.duplicated(keys).any(),
        "finite_range": bool(np.isfinite(frame.hs_pred).all() and frame.hs_pred.between(0, 30).all()),
        "separate_pid_replay": answer["pid"] != replay["pid"] and replay["status"] == "EXACT_REPLAY_PASS",
        "candidate_hash": sha(result_path) == answer["candidate_sha256"] == replay["candidate_sha256"],
        "base_equals_2015b387": answer["base_sha256"] == replay["base_sha256"] == expected_base,
        "final_equals_70761aff": sha(result_path) == expected_final,
        "fresh_models_unchanged": all(sha(fresh / n) == h for n, h in original_models.items()),
        "audit_models_unchanged": all(sha(out / n) == h for n, h in original_models.items()),
        "source_pins_unchanged": all(sha(out / n) == h for n, h in pins.items()),
        "archives_unchanged": sha(source_zip) == source_hash and sha(package / "SAVED_MODELS.zip") == saved_zip_hash,
    }
    result = {"status": "PASS" if all(checks.values()) else "MISMATCH_INVESTIGATE_ONLY",
              "checks": checks, "confidence": "Share with caveats",
              "fresh_cold_receipt": cold, "addon_stages": stages,
              "addon_wall_seconds": time.monotonic() - start,
              "answer": {"path": str(result_path), "sha256": sha(result_path),
                         "rows": len(frame), "bytes": result_path.stat().st_size},
              "infer": answer, "replay": replay,
              "fresh_training_sha256": sha(fresh / "06_docs/cpudet-training.json"),
              "packaged_addon_sha256": sha(out / "02_code/noshrink.py"),
              "source_zip_sha256": source_hash, "saved_zip_sha256": saved_zip_hash,
              "new_fits": 0, "uploads": 0, "historical_outputs_modified": False,
              "completed_fresh_colds": 1, "not_two_fresh_colds": True,
              "caveats": ["same Windows PC only; not fresh OS/offline/network-blocked test",
                          "base elapsed exceeds six hours; general official scope unverified",
                          "chained completed base run plus packaged addon stages, not one new training notebook execution",
                          "adopt-trained sets expected_base_answer_sha256 null; independent checks above test exact historical hashes"]}
    save(report, result)
    print(json.dumps({"status": result["status"], "checks": checks, "answer": result["answer"]}))


if __name__ == "__main__":
    main()
