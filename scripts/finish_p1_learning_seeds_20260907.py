"""Actual local and extracted saved notebook replay for each independent P1 arm."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile

import nbformat
import package_p1_learning_seeds_20260907 as b
from build_final_release_20260907_v1 import copy_pins, read, save, sha
from jupyter_client import KernelManager
from nbclient import NotebookClient


def archive(directory, target):
    # This variant replays tree-qa in the saved notebook, unlike the old archive.
    # Explicitly permit only its 256-row TRAIN-derived numerical probe, not general NPZ.
    import numpy as np

    files = sorted(p for p in directory.rglob("*") if p.is_file())
    for path in files:
        relative = path.relative_to(directory)
        assert not path.is_symlink()
        assert not any(x in {"__pycache__", ".git", ".env"} for x in relative.parts)
        assert path.suffix.lower() not in {".csv", ".parquet", ".log", ".pyc"} and "LOCK" not in path.name
        if path.suffix.lower() == ".npz":
            if relative.as_posix() == "03_model/tree/own_probe.npz":
                with np.load(path, allow_pickle=False) as z:
                    assert set(z.files) == {"o_x", "b_x", "o", "b"}
                    assert z["o_x"].shape == z["b_x"].shape == (256, 80)
                    assert z["o"].shape == z["b"].shape == (256,)
            else:
                assert relative.as_posix().startswith("03_model/mstcn/04_validation/")
    pins = {p.relative_to(directory).as_posix(): sha(p) for p in files}
    if not target.exists():
        with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(directory.rglob("*")):
                z.write(p, p.relative_to(directory).as_posix() + ("/" if p.is_dir() else ""))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert set(n for n in z.namelist() if not n.endswith("/")) == set(pins)
        import hashlib
        for name, digest in pins.items():
            with z.open(name) as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == digest
    return {"file": target.name, "sha256": sha(target), "bytes": target.stat().st_size,
            "files": len(pins), "member_sha256": pins, "CRC_and_members": "PASS"}


def execute(root, receipt):
    path = root / "SAVED_PREDICT.ipynb"
    before = sha(path)
    start = time.monotonic()
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    NotebookClient(nb, km=km, timeout=None, resources={"metadata": {"path": str(root)}}).execute(cleanup_kc=True)
    assert sha(path) == before
    qa = read(root / "06_docs/P1_submission.csv.json")
    assert qa["status"] == "SCHEMA_PASS"
    result = {"status": "PASS", "notebook_sha256": before, "seconds": time.monotonic()-start, "answer": qa}
    save(receipt, result)
    print(json.dumps(result), flush=True)
    return result


def main(arm, resume_package=False):
    import numpy as np
    import pandas as pd

    os.environ["P1_DATA_DIR"] = "C:/Users/cedis/Downloads/데이터셋_P1/P1_qc_anomaly"
    out = b.DAY / ("P1_original_" + arm + "_v1")
    package = out / "PACKAGE"
    started = time.monotonic()
    if resume_package:
        first = read(out / "local-notebook.json")
        assert first["status"] == "PASS" and first["answer"]["sha256"] == sha(package / "05_answer/P1_submission.csv")
        assert first["notebook_sha256"] == sha(package / "SAVED_PREDICT.ipynb")
    else:
        first = execute(package, out / "local-notebook.json")
    pins = read(package / "source-manifest.json")["files"]
    pins["source-manifest.json"] = sha(package / "source-manifest.json")
    models = read(package / "06_docs/model-lineage.json")["files"]
    archives = []
    for role in ["SOURCE_ONLY", "SAVED_MODELS"]:
        target = out / role
        for name in ["01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"]:
            (target / name).mkdir(parents=True, exist_ok=resume_package)
        copy_pins(package, target, pins)
        if role == "SAVED_MODELS":
            copy_pins(package, target, models)
        archives.append(archive(target, out / ("P1_original_" + arm + "_" + role + ".zip")))
    save(out / "archives.json", archives)
    replay = out / "EXTRACTED_SAVED"
    replay.mkdir()
    a = next(a for a in archives if "SAVED_MODELS" in a["file"])
    with zipfile.ZipFile(out / a["file"]) as z:
        assert all((replay / n).resolve().is_relative_to(replay.resolve()) for n in z.namelist())
        z.extractall(replay)
    for name in ["01_data", "03_model", "04_logs", "05_answer", "06_docs"]:
        (replay / name).mkdir(exist_ok=True)
    second = execute(replay, out / "extracted-notebook.json")
    assert first["answer"]["sha256"] == second["answer"]["sha256"]
    original = package / "05_answer/P1_submission.csv"
    answer = out / "ANSWER" / ("submission_p1_original_" + arm + ".csv")
    answer.parent.mkdir()
    shutil.copy2(original, answer)
    frame = pd.read_csv(answer)
    keys = ["station", "year", "layer", "time"]
    source = pd.read_csv(os.environ["P1_DATA_DIR"] + "/test.csv", usecols=keys)
    assert len(frame) == 169011 and list(frame) == keys+["label"]
    assert frame[keys].astype(str).equals(source[keys].astype(str)) and not frame.duplicated(keys).any()
    assert np.isin(frame.label, [0, 1]).all()
    assert sha(answer) != "57844ef235f987059d17bddb035cc0a98b92bb0d53263cf878c68b31bbc53687", "distinct candidate required"
    result = {"status": "CANDIDATE_SAVED_ZIP_REPLAY_PASS", "arm": arm,
              "answer": {"path": str(answer), "rows": len(frame), "bytes": answer.stat().st_size, "sha256": sha(answer)},
              "archives": archives, "local_notebook": first, "extracted_notebook": second,
              "schema_key_order_unique_finite_binary": "PASS", "new_whole_variant_cold": False,
              "reproduction_basis": "existing original component cold + new trained full component(s) + two actual notebook saved paths exact SHA",
              "historical_qa": str(b.s.OUT / "independent-qa.json"), "official_score": None,
              "seconds": time.monotonic()-started, "new_fits": 0, "uploads": 0, "final_designation": False}
    save(out / "result.json", result)
    save(b.s.REPORT / (arm + "-candidate-ready.json"), result)
    print(json.dumps({"status": result["status"], "answer": result["answer"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=["B5", "O_slow"])
    parser.add_argument("--resume-package", action="store_true")
    args = parser.parse_args()
    main(args.arm, args.resume_package)
