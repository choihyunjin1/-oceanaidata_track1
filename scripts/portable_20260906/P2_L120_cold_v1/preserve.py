"""Read-only artifact verification plus a new preservation archive; no GPU or official I/O."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

EXPECTED = "fee6118bb4a4d1d421094aa0174d634cf73e919804ea1b62ab0bab4f1384ce4d"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(origin, package, source_zip, saved_zip, receipt):
    origin, package, source_zip, saved_zip, receipt = map(Path, (origin, package, source_zip, saved_zip, receipt))
    checks = {}

    def check(label, value):
        checks[label] = bool(value)
        if not value:
            raise AssertionError(label)

    origin_files = {p.relative_to(origin).as_posix(): sha(p) for p in origin.rglob("*") if p.is_file()}
    for relative, digest in origin_files.items():
        check("copy:" + relative, sha(package / relative) == digest)
    manifest = read(package / "PACKAGE_MANIFEST.json")
    for relative, digest in manifest["files"].items():
        check("sealed:" + relative, sha(package / relative) == digest)
    train = read(package / "04_logs/training-result.json")
    training_qa = read(package / "04_logs/training-independent-qa.json")
    infer = read(package / "04_logs/inference-result.json")
    replay = read(package / "04_logs/replay-result.json")
    qa = read(package / "04_logs/independent-qa.json")
    cold = read(package / "04_logs/cold-terminal.json")
    check("scratch3", train["new_full_fits"] == 3 and train["empty_model_directory_before_training"])
    check("cold_training_only3", cold["new_fits"] == 3)
    check("3seeds", [f["seed"] for f in train["fits"]] == [20260901, 20260902, 20260903])
    for fit in train["fits"]:
        check("model:" + str(fit["seed"]), sha(package / "03_model" / fit["file"]) == fit["sha256"])
    check("training_qa19", training_qa["status"] == "PASS" and training_qa["checks_passed"] == 19 and all(training_qa["checks"].values()))
    check("training_qa_link", training_qa["training_result_sha256"] == sha(package / "04_logs/training-result.json"))
    check("final_qa16", qa["status"] == "PASS" and qa["checks_passed"] == 16 and all(qa["checks"].values()))
    check("5uniquePIDs", len({r["pid"] for r in (train, training_qa, infer, replay, qa)}) == 5)
    check("answer_exact", sha(package / infer["answer"]) == infer["answer_sha256"] == replay["answer_sha256"] == cold["answer_sha256"] == EXPECTED)
    check("replay_exact", replay["exact_first_inference_sha_match"] and sha(package / replay["answer"]) == EXPECTED)
    check("cold_qa_link", cold["qa_sha256"] == sha(package / "04_logs/independent-qa.json"))
    check("scope", train["official_access_rows"] == training_qa["official_rows"] == train["csv_written"] == qa["hidden_rows"] == qa["upload"] == 0)
    check("post_training_keys_only", infer["access"]["sample_value_rows"] == replay["access"]["sample_value_rows"] == 0)
    check("runtime_cap", cold["runtime_seconds"] < 1800)
    check("repo_guard_scope", cold["original_repository_denial_requested"] and not cold["fresh_venv_tested"])
    for role in ("TRAIN", "PREDICT"):
        nr = read(package / "06_docs/executed_notebooks" / (role + "-receipt.json"))
        executed = read(package / "06_docs/executed_notebooks" / (role + ".ipynb"))
        source = read(package / (role + ".ipynb"))
        check(role + ":actualPASS", nr["status"] == "PASS" and nr["source_sha256"] == sha(package / (role + ".ipynb")))
        check(role + ":source_unchanged", [c["source"] for c in source["cells"]] == [c["source"] for c in executed["cells"]])
        cells = [c for c in executed["cells"] if c["cell_type"] == "code"]
        check(role + ":executed_all_cells", all(c["execution_count"] is not None for c in cells))
        check(role + ":no_error_output", not any(o.get("output_type") == "error" for c in cells for o in c["outputs"]))
    with zipfile.ZipFile(source_zip) as zipped:
        check("source_empty_dirs", all(name + "/" in zipped.namelist() for name in ("03_model", "04_logs", "05_answer")))
        check("source_no_models_answers", not any(n.endswith((".pt", ".csv", ".npz", ".parquet")) for n in zipped.namelist()))
        for relative, digest in manifest["files"].items():
            check("sourceZIP:" + relative, hashlib.sha256(zipped.read(relative)).hexdigest() == digest)
    if saved_zip.exists() or receipt.exists():
        raise ValueError("new archive/receipt only")
    selected, excluded = [], []
    for path in sorted(p for p in package.rglob("*") if p.is_file()):
        relative = path.relative_to(package).as_posix()
        if (path.suffix in {".log", ".npz", ".npy", ".parquet", ".pyc"} or path.name.startswith("replay_p2_")
                or "lock" in path.name.lower()
                or "probe" in relative.lower() or "__pycache__" in path.parts):
            excluded.append(relative)
        else:
            selected.append(path)
    check("no_distributed_data", not any(p.name in {"observations.csv", "test_index.csv", "sample_submission.csv"} for p in selected))
    check("archive_has_no_locks", not any("lock" in p.name.lower() for p in selected))
    with zipfile.ZipFile(saved_zip, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in selected:
            zipped.write(path, path.relative_to(package).as_posix())
    with zipfile.ZipFile(saved_zip) as zipped:
        for path in selected:
            check("savedZIP:" + path.relative_to(package).as_posix(), hashlib.sha256(zipped.read(path.relative_to(package).as_posix())).hexdigest() == sha(path))
    result = {"status": "PASS", "checks_passed": len(checks), "checks": checks,
        "source_zip": {"file": source_zip.name, "sha256": sha(source_zip), "bytes": source_zip.stat().st_size,
                       "role": "empty-model cold training and inference via TRAIN/PREDICT notebooks"},
        "saved_zip": {"file": saved_zip.name, "sha256": sha(saved_zip), "bytes": saved_zip.stat().st_size,
                      "role": "preservation only; consumed stages are not a long-term inference interface",
                      "files": len(selected), "excluded": excluded},
        "cold_result_sha256": sha(package / "04_logs/cold-terminal.json"), "answer_sha256": EXPECTED,
        "permanent_directory": str(package.resolve()), "copied_file_hashes": origin_files,
        "new_fits": 0, "gpu": False, "official_input_rows": 0, "upload": 0}
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps({"status": "PASS", "checks": len(checks), "saved_zip": result["saved_zip"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("origin", "package", "source-zip", "saved-zip", "receipt"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    run(args.origin, args.package, args.source_zip, args.saved_zip, args.receipt)
