"""Package reviewed cold sources and owned trained models without mutating attempts."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import nbformat
from build_candidate_upload_parts_20260906_v1 import split, validate_parts


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def owned(root, relative):
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()) or target.is_symlink():
        raise ValueError("unsafe package member")
    return target


def copy_pins(source, target, pins):
    for relative, expected in pins.items():
        old = owned(source, relative)
        if sha(old) != expected:
            raise ValueError("source/model pin mismatch: " + relative)
        new = owned(target, relative)
        new.parent.mkdir(parents=True, exist_ok=True)
        if new.exists():
            if sha(new) != expected:
                raise ValueError("conflicting member")
        else:
            shutil.copy2(old, new)


def archive(directory, target):
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    for path in files:
        rel = path.relative_to(directory)
        if path.is_symlink() or any(x in {"__pycache__", ".git", ".env"} for x in rel.parts):
            raise ValueError("forbidden package member")
        if path.suffix.lower() in {".csv", ".parquet", ".log", ".pyc"} or "LOCK" in path.name:
            raise ValueError("answers/raw tables/logs/locks excluded")
        if path.suffix.lower() == ".npz" and not str(rel).replace("\\", "/").startswith("03_model/mstcn/04_validation/"):
            raise ValueError("only explicitly required P1 owned TRAIN probes allowed")
    pins = {p.relative_to(directory).as_posix(): sha(p) for p in files}
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(p for p in directory.rglob("*") if p.is_dir()):
            z.writestr(path.relative_to(directory).as_posix() + "/", b"")
        for path in files:
            z.write(path, path.relative_to(directory).as_posix())
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP CRC mismatch")
        for name, expected in pins.items():
            with z.open(name) as stream:
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(1 << 20), b""):
                    h.update(block)
            if h.hexdigest() != expected:
                raise ValueError("archived member mismatch")
    return {"file": target.name, "sha256": sha(target), "bytes": target.stat().st_size,
            "files": len(pins), "member_sha256": pins, "CRC_and_members": "PASS"}


def saved_notebook(problem, target, commands):
    # This invokes unchanged inference stages in a fresh extracted saved package,
    # never the consumed training orchestrator or its elapsed-time locks.
    script = (
        "import os, sys, subprocess\nfrom pathlib import Path\n"
        "root = Path.cwd().resolve()\n"
        f"assert os.environ.get('{problem}_DATA_DIR'), 'Set distributed data directory'\n"
        "assert not any((root / '05_answer').iterdir()), 'Use a fresh saved ZIP extraction'\n"
        f"commands = {commands!r}\n"
        "for command in commands:\n"
        "    args = [os.environ.get(arg[1:], '') if arg.startswith('$') else arg for arg in command]\n"
        "    subprocess.run([sys.executable, '-B', *args], cwd=root, check=True, timeout=1800)\n"
    )
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(
            f"# {problem} saved-model inference\n\n"
            "Zero new fits. Extract this saved archive into a new folder. Select the pinned environment. "
            "TRAIN notebooks belong to the separate SOURCE_ONLY archive, not this nonempty model folder. "
            "No existing locks or artifacts are reset. Predictions go to 05_answer. "
            "The receipt outside this frozen archive records actual execution validation."),
        nbformat.v4.new_code_cell(script),
        nbformat.v4.new_markdown_cell(
            "Compare the complete answer SHA to the release manifest; a mismatch must not inherit the recorded score. "
            "No upload or final designation is performed. Preserve a failed extraction."),
    ], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    nbformat.validate(notebook)
    nbformat.write(notebook, target / "SAVED_PREDICT.ipynb")


def build(problem, verified, destination):
    verified, destination = verified.resolve(), destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    manifest_name = "source-manifest.json" if problem == "P1" else "PACKAGE_MANIFEST.json"
    source_pins = read(verified / manifest_name)["files"]
    source_pins = {**source_pins, manifest_name: sha(verified / manifest_name)}
    source = destination / "SOURCE_ONLY"
    saved = destination / "SAVED_MODELS"
    for folder in (source, saved):
        for name in ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"):
            (folder / name).mkdir(parents=True)
        copy_pins(verified, folder, source_pins)
    if problem == "P1":
        terminal = read(verified / "terminal.json")
        if terminal["status"] != "TRAIN_TO_ANSWER_COMPLETE" or not terminal["historical_answer_exact"] or terminal["fits"] != 7:
            raise ValueError("P1 full-cold evidence missing")
        answer_name = "P1_submission.csv"
        expected = "57844ef235f987059d17bddb035cc0a98b92bb0d53263cf878c68b31bbc53687"
        ms = "03_model/mstcn/"
        model_pins = {ms + k: v for k, v in read(verified / (ms + "training-result.json"))["files_sha256"].items()}
        extra = [ms + "training-result.json", ms + "fresh-process-replay.json", ms + "terminal.json",
                 "03_model/tree/training-result.json", "03_model/tree/qa.json"]
        tree = read(verified / "03_model/tree/training-result.json")
        model_pins.update({f"03_model/tree/{name}.joblib": tree[name + "_sha256"] for name in ("O", "B")})
        model_pins.update({name: sha(verified / name) for name in extra})
        commands = [["02_code/run.py", stage, "--data", "$P1_DATA_DIR"] for stage in ("infer-tree", "infer-ms", "combine")]
        cold_seconds, fits = terminal["runtime_seconds"], 7
    else:
        terminal = read(verified / "04_logs/cold-terminal.json")
        if terminal["status"] != "COLD_TRAIN_INFER_REPLAY_COMPLETE" or not terminal["exact_existing_candidate_sha_match"]:
            raise ValueError("P2 full-cold evidence missing")
        answer_name = "submission_p2_L120_3seed.csv"
        expected = "fee6118bb4a4d1d421094aa0174d634cf73e919804ea1b62ab0bab4f1384ce4d"
        train = read(verified / "03_model/MODEL_MANIFEST.json")
        model_pins = {"03_model/" + fit["file"]: fit["sha256"] for fit in train["fits"]}
        extra = ["03_model/MODEL_MANIFEST.json", "04_logs/training-result.json", "04_logs/training-independent-qa.json"]
        model_pins.update({name: sha(verified / name) for name in extra})
        commands = [["-I", "02_code/boot.py", stage] for stage in ("RUN_INFERENCE", "REPLAY", "FINAL_QA")]
        cold_seconds, fits = terminal["runtime_seconds"], 3
    copy_pins(verified, saved, model_pins)
    saved_notebook(problem, saved, commands)
    for folder in (source, saved):
        (folder / "RELEASE_ROLE.md").write_text(
            f"# {problem}: {folder.name}\n\n"
            + ("Start from empty models using the original training notebook/README.\n" if folder == source else
               "Run SAVED_PREDICT.ipynb, NOT TRAIN/RUN_ALL. This is a new saved-model task, not a restarted cold attempt.\n")
            + "Original pinned source README is preserved as historical provenance. Use the top-level current release guide for selected scores and status.\n"
            + "P1 owned TRAIN-derived probes are retained only where the unchanged model integrity guard requires them; no raw source tables are bundled.\n",
            encoding="utf-8")
    archives = [archive(folder, destination / f"{problem}_{folder.name}.zip") for folder in (source, saved)]
    parts = None
    if (destination / archives[1]["file"]).stat().st_size > 45_000_000:
        parts = split(destination / archives[1]["file"], destination / "SAVED_MODEL_PARTS")
        validate_parts(destination / "SAVED_MODEL_PARTS/REASSEMBLY_MANIFEST.json")
    answer = verified / "05_answer" / answer_name
    if sha(answer) != expected:
        raise ValueError("selected answer changed")
    (destination / "ANSWER").mkdir()
    shutil.copy2(answer, destination / "ANSWER" / answer_name)
    result = {"problem": problem, "status": "ARCHIVES_BUILT_SAVED_EXECUTION_PENDING",
              "source_model_pins_checked": len(source_pins) + len(model_pins),
              "original_full_cold_fits": fits, "original_full_cold_seconds": cold_seconds,
              "answer_file": "ANSWER/" + answer_name, "answer_sha256": expected,
              "archives": archives, "split_parts": len(parts["parts"]) if parts else 0,
              "new_fits": 0, "uploads": 0, "source_modified": False,
              "fresh_venv_or_OS_offline_tested": False}
    save(destination / "BUILD_QA.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", choices=("P1", "P2"))
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.problem, args.verified, args.destination)
    print(json.dumps({k: v for k, v in result.items() if k != "archives"}, ensure_ascii=False))
