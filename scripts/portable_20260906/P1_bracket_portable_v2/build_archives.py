"""Create new, explicit code-only cold and saved-model replay ZIPs without changing source."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "artifacts/p1_bracket_candidate_20260906_v1/package"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("mechanical extraction anchor changed")
    return source.replace(old, new)


def cold_changes(run, config, qa, contract):
    run = replace_once(run, 'RUN = "p1_bracket_candidate_20260906_v1"',
                       'RUN = "p1_bracket_portable_cold_cpu2_20260906_v2"')
    run = replace_once(run, "THREADS = 4", "THREADS = 2")
    config = dict(config, experiment_id="p1_bracket_portable_cold_cpu2_20260906_v2", threads=2)
    config["resource_amendment"] = contract["resource_change"]
    qa = replace_once(qa, 'training["threads"] == cfg["threads"] == 4',
                      'training["threads"] == cfg["threads"] == 2')
    qa = replace_once(qa, 'parameters["n_jobs"] == 4', 'parameters["n_jobs"] == 2')
    qa = replace_once(qa, 'os.environ[name] = "4"', 'os.environ[name] = "2"')
    qa = replace_once(qa, "    probabilities = {}\n", "    probabilities = {}\n    baseline_comparisons = []\n")
    old = '''            check(prefix + "_unchanged_baseline_sha", fit["sha256"]
                  == cfg["baseline_original_model_sha_metadata_only"][stage])'''
    new = '''            baseline_comparisons.append({"model": prefix, "same_as_cpu4_model_sha":
                fit["sha256"] == cfg["baseline_original_model_sha_metadata_only"][stage],
                "scope": "CPU2 resource variant; comparison, not a quality gate"})'''
    qa = replace_once(qa, old, new)
    qa = replace_once(qa, '"passed": len(checks), "failed": 0,',
                      '"passed": len(checks), "failed": 0, "cpu4_model_comparisons": baseline_comparisons,')
    return run, config, qa


def validate_source(contract):
    for relative, key in (("02_code/run.py", "source_run_sha256"),
                          ("02_code/source-manifest.json", "source_manifest_sha256"),
                          ("train_result.json", "source_train_sha256"),
                          ("03_model/frozen_recipe.json", "source_recipe_sha256")):
        if sha(SOURCE / relative) != contract[key]:
            raise ValueError("source lineage mismatch: " + relative)
    files = json.loads((SOURCE / "02_code/source-manifest.json").read_text(encoding="utf-8"))
    for relative, record in files.items():
        if sha(SOURCE / "02_code" / relative) != record["sha256"]:
            raise ValueError("source code mismatch")
    recipe = json.loads((SOURCE / "03_model/frozen_recipe.json").read_text(encoding="utf-8"))
    for arm, expected in recipe["model_hashes"].items():
        if sha(SOURCE / "03_model" / ("full_" + arm + ".joblib")) != expected:
            raise ValueError("source model mismatch")
    return files


def archive(package, target):
    paths = sorted(p for p in package.rglob("*") if p.is_file())
    if any(p.suffix in {".csv", ".npy", ".npz", ".parquet", ".pyc"} for p in paths):
        raise ValueError("answer/raw/probe/cache payload forbidden in these ZIPs")
    hashes = {p.relative_to(package).as_posix(): sha(p) for p in paths}
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as stream:
        for path in paths:
            stream.write(path, "P1/" + path.relative_to(package).as_posix())
        for folder in ("01_data", "03_model", "04_logs", "05_answer", "06_docs"):
            stream.writestr("P1/" + folder + "/", "")
    with zipfile.ZipFile(target) as stream:
        if stream.testzip() is not None or any(
            hashlib.sha256(stream.read("P1/" + name)).hexdigest() != digest for name, digest in hashes.items()
        ):
            raise ValueError("ZIP CRC/member hash mismatch")
    return {"sha256": sha(target), "bytes": target.stat().st_size, "files": hashes,
            "source_data_files": 0, "answer_files": 0, "probe_files": 0}


def build(destination):
    if destination.exists():
        raise FileExistsError("new archive build destination required")
    contract = json.loads((HERE / "contract.json").read_text())
    source_files = validate_source(contract)
    destination.mkdir(parents=True)
    receipts = {}
    for mode in ("saved", "cold"):
        package = destination / mode / "P1"
        for directory in ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"):
            (package / directory).mkdir(parents=True)
        for name in list(source_files) + ["source-manifest.json"]:
            target = package / "02_code" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCE / "02_code" / name, target)
        shutil.copy2(HERE / "README.md", package / "README.md")
        if mode == "saved":
            for name in ("full_original.joblib", "full_balanced.joblib", "frozen_recipe.json"):
                shutil.copy2(SOURCE / "03_model" / name, package / "03_model" / name)
            for name in ("model-replay-qa.json", "independent-training-qa.json", "root-answer-qa.json"):
                shutil.copy2(SOURCE / "06_docs" / name, package / "06_docs" / name)
            shutil.copy2(SOURCE / "train_result.json", package / "06_docs/source-train-result.json")
            shutil.copy2(HERE / "archive_infer.py", package / "02_code/archive_infer.py")
        else:
            run_path, cfg_path = package / "02_code/run.py", package / "02_code/configs/candidate.json"
            run, cfg, qa = cold_changes(run_path.read_text(encoding="utf-8"),
                json.loads(cfg_path.read_text()), (SOURCE / "06_docs/qa_training_independent.py").read_text(encoding="utf-8"), contract)
            run_path.write_text(run, encoding="utf-8", newline="\n")
            write(cfg_path, cfg)
            (package / "06_docs/qa_training_independent.py").write_text(qa, encoding="utf-8", newline="\n")
            manifest = json.loads((package / "02_code/source-manifest.json").read_text())
            for name in ("run.py", "configs/candidate.json"):
                manifest[name]["sha256"] = sha(package / "02_code" / name)
                manifest[name]["resource_amendment_source"] = {"build_archives.py": sha(Path(__file__)),
                                                               "contract.json": sha(HERE / "contract.json")}
            write(package / "02_code/source-manifest.json", manifest)
        members = {p.relative_to(package).as_posix(): sha(p) for p in sorted(package.rglob("*")) if p.is_file()}
        write(package / "portable-manifest.json", {"mode": mode, "contract": contract, "files": members,
              "builder_sha256": sha(Path(__file__)), "source_snapshot_unchanged": mode == "saved"})
        receipts[mode] = archive(package, destination / ("P1_bracket_" + mode + "_v2.zip"))
    write(destination / "archive-build-qa.json", {"status": "CRC_MEMBER_HASH_PASS", "archives": receipts,
          "new_fits": 0, "official_rows": 0, "uploads": 0, "contract": contract})
    print(json.dumps({k: {field: v[field] for field in ("bytes", "sha256")} for k, v in receipts.items()}))
    return receipts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    build(parser.parse_args().destination.resolve())
