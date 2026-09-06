"""Build separate source-only and saved P1 archives without reading answer values."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ID = "p1_champion_portable_20260906_v1"
CHAMPION = "p1_champion_reconstruction_20260906_v1"
EXPECTED = "b2f17f5cda8030cb3d97fbb504e6babb6aef8ba7fe555092901479677af0625e"
RECON = "scripts/" + CHAMPION
VERSIONS = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm",
            "torch", "joblib", "pyarrow", "psutil")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def source_inventory():
    seal = read(ROOT / "reports" / CHAMPION / "tree-seal.json")
    pins = dict(seal["sources"])
    pins.update(read(ROOT / "artifacts" / CHAMPION / "mstcn_full3/training-result.json")["source_pins"])
    for name in ("mstcn.py", "composition.py"):
        pins[RECON + "/" + name] = sha(ROOT / RECON / name)
    for name in ("src/ocean_goal/__init__.py", "src/ocean_goal/meaningful_score.py"):
        pins[name] = sha(ROOT / name)
    for name, expected in pins.items():
        if sha(ROOT / name) != expected:
            raise ValueError("original frozen source changed: " + name)
    return pins


def archive_evidence():
    decision = read(ROOT / "reports" / CHAMPION / "post-qa-decision.json")
    if decision["tree_arm"] != "union":
        raise ValueError("frozen tree composition changed")
    evidence = {}
    for name, pin in decision["evidence"].items():
        if sha(pin["path"]) != pin["sha256"]:
            raise ValueError("source evidence changed: " + name)
        value = read(pin["path"])
        if value["status"] not in ("PASS", "COMPLETE", "TRAINING_COMPLETE_REPLAY_PENDING",
                                   "COMPLETE_RETROSPECTIVE_EVALUATION"):
            raise ValueError("source evidence did not pass")
        evidence[name] = {"sha256": pin["sha256"], "status": value["status"]}
    for name in ("candidate", "candidate_replay"):
        path = ROOT / "artifacts" / CHAMPION / name / "05_answer/terminal.json"
        value = read(path)
        if value["status"] != "PASS" or value["csv_sha256"] != EXPECTED:
            raise ValueError("original answer/replay metadata mismatch")
        evidence[name] = {"sha256": sha(path), "answer_sha256": value["csv_sha256"]}
    return evidence


def copy_saved(destination):
    original = ROOT / "artifacts" / CHAMPION
    tree = read(original / "tree_full/terminal_result.json")
    ms = read(original / "mstcn_full3/training-result.json")
    inventory = {}
    selected = [(original / "tree_full/selector.json", "tree/selector.json",
                 tree["files"]["selector.json"])]
    for relative, digest in tree["files"].items():
        if relative.startswith("full/models/"):
            selected.append((original / "tree_full" / relative, "tree/" + relative, digest))
    for relative, digest in ms["files_sha256"].items():
        if relative.startswith("03_model/"):
            selected.append((original / "mstcn_full3" / relative, "mstcn/" + relative, digest))
    for source, relative, digest in selected:
        if sha(source) != digest:
            raise ValueError("saved source model changed")
        target = destination / "03_model" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        inventory["03_model/" + relative] = digest
    if sum(p.endswith(".pt") for p in inventory) != 3 or sum(
            "/models/" in p and not p.endswith("preprocess.joblib") for p in inventory) != 4:
        raise ValueError("saved model count changed")
    return inventory


def build(destination, kind):
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError("new destination required")
    pins, evidence = source_inventory(), archive_evidence()
    if not all((HERE / name).is_file() for name in ("TRAIN.ipynb", "PREDICT.ipynb")):
        raise FileNotFoundError("root notebooks must be added before package sealing")
    for name in ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"):
        (destination / name).mkdir(parents=True)
    for relative in pins:
        target = destination / "02_code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for name in ("README.md", "TRAIN.ipynb", "PREDICT.ipynb"):
        shutil.copy2(HERE / name, destination / name)
    shutil.copy2(HERE / "run.py", destination / "02_code/run.py")
    requirements = {v: importlib.metadata.version(v) for v in VERSIONS}
    (destination / "02_code/requirements.txt").write_text(
        "\n".join(f"{name}=={value}" for name, value in requirements.items()) + "\n", encoding="utf-8")
    (destination / "01_data/README.md").write_text(
        "Set P1_DATA_DIR to the original organizer P1_qc_anomaly folder. Data are not redistributed.\n",
        encoding="utf-8")
    write(destination / "06_docs/provenance.json", {"evidence": evidence, "sources": pins,
        "old_answers_read": 0, "old_ooof_inputs": 0, "old_clock_not_reset": True})
    frozen = {"id": ID, "kind": kind, "tree_arm": "union", "expected_answer_sha256": EXPECTED,
              "tree_config": read(ROOT / RECON / "tree-contract.json"),
              "budget": {"tree_cpu_threads": 4, "MS_cpu_threads": 2, "exclusive_cuda_device": 0,
                         "tree_inner_fits": 4, "tree_full_fits": 4, "MS_full_fits": 3,
                         "total_fits": 11, "cold_seconds": 21600, "saved_command_seconds": 1800},
              "packages": requirements, "source_snapshot": pins,
              "deployment": "exact original Q4 earlier-inner; union fixed, cells unused",
              "forbidden_training_inputs": ["old_models", "OOF", "answers", "official", "external"],
              "cold_contract_is_new_not_old31fit_restart": True}
    write(destination / "02_code/frozen.json", frozen)
    models = copy_saved(destination) if kind == "saved" else {}
    files = {p.relative_to(destination).as_posix(): sha(p)
             for folder in ("02_code", "06_docs", "01_data")
             for p in sorted((destination / folder).rglob("*")) if p.is_file()}
    files.update({name: sha(destination / name) for name in ("README.md", "TRAIN.ipynb", "PREDICT.ipynb")})
    files.update(models)
    write(destination / "package-manifest.json", {"id": ID, "kind": kind, "files": files})
    return {"status": "SEALED_NOT_EXECUTED", "kind": kind, "path": str(destination),
            "manifest_sha256": sha(destination / "package-manifest.json"), "model_files": len(models)}


def archive(package, target):
    if target.exists():
        raise FileExistsError("archive already exists")
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for path in sorted(package.rglob("*")):
            if path.is_dir() and "__pycache__" not in path.parts:
                z.writestr(path.relative_to(package).as_posix() + "/", b"")
            elif path.is_file() and "__pycache__" not in path.parts:
                z.write(path, path.relative_to(package).as_posix())
    return {"zip": str(target), "sha256": sha(target), "bytes": target.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--kind", choices=("cold", "saved"), required=True)
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    result = build(args.destination, args.kind)
    if args.zip:
        result["archive"] = archive(args.destination, args.zip)
    print(json.dumps(result))
