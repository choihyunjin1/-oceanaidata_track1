"""Build an isolated code-only P3 forward-candidate cold package; zero data/fits."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MATERIALIZER_SHA = "de74d271e45a7dcc520ee553ca1bc86c915ccc30d7c3f3250a77cb8b35fcd5ea"
NUMERIC = "run_p3_numeric_lead_forward_gpu_20260906_v2.py"
HMAX = "run_p3_hmax_removed_forward_20260906_v1.py"
HELPERS = [NUMERIC, HMAX, "p3_forward_candidate_materialize_20260906_v1.py",
           "ocean_evaluation_contract_v5.py", "audit_ocean_forward_support_20260906_v1.py"]
MODULES = ["__init__.py", "data.py", "features.py", "models.py", "validation.py",
           "loss_router.py", "persistence_shrink.py"]
SHRINK_PROVENANCE = "scripts/portable_20260906/P3/recovery_v3/06_docs/shrink-provenance.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build(destination):
    if destination.exists():
        raise FileExistsError("new empty package path required")
    candidate = json.loads((REPO / "configs/experiments/p3_numeric_lead_forward_gpu_20260906_v2.json").read_text())
    recipe_path = REPO / candidate["recipe"]
    if sha(recipe_path) != candidate["inputs"][candidate["recipe"]]:
        raise ValueError("candidate's frozen source recipe changed")
    recipe = json.loads(recipe_path.read_text())
    if sha(REPO / "scripts/p3_forward_candidate_materialize_20260906_v1.py") != MATERIALIZER_SHA:
        raise ValueError("reviewed materializer changed; re-review before new snapshot")
    sources = [*("scripts/" + name for name in HELPERS), *("src/p3_wave/" + name for name in MODULES),
               "configs/evaluation/ocean_forward_v5.json"]
    for name in sources:
        if name in candidate["inputs"] and sha(REPO / name) != candidate["inputs"][name]:
            raise ValueError("frozen source dependency changed: " + name)
    for directory in ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"):
        (destination / directory).mkdir(parents=True)
    code = destination / "02_code"
    provenance = {}
    for name in sources:
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
        provenance[name] = sha(target)
    shutil.copy2(HERE / "run.py", code / "run.py")
    shutil.copy2(HERE / "README.md", destination / "README.md")
    shutil.copy2(REPO / SHRINK_PROVENANCE, destination / "06_docs/shrink-provenance.json")
    frozen = {
        "package_id": "p3_forward_candidate_cold_20260906_v1",
        "scope": "one explicit candidate, not a new search; prepare common before choice",
        "source_files": candidate["source_files"],
        "source_recipe_sha256": sha(recipe_path),
        "shrink_provenance_sha256": sha(destination / "06_docs/shrink-provenance.json"),
        "recipe": {key: recipe[key] for key in ("model", "router", "shrink")},
        "fold_seeds": candidate["fold_seeds"],
        "expected_train": candidate["expected_train"],
        "expected_validation": candidate["expected_validation"],
        "expected_anchors": 24360, "expected_oof_rows": 103602,
        "variants": {"numeric": {"features": 591, "single_kind": "numeric_single"},
                     "hmax": {"features": 527, "single_kind": "categorical_single"}},
        "budget": {"cpu_threads": 2, "exclusive_gpu_device": 0,
                   "backbone_fits": 12, "router_fits": 5, "total_seconds": 21600},
        "source_snapshot": provenance,
        "old_cache_oof_models_answers": "FORBIDDEN",
        "selection": "variant explicitly sealed before fit; no score/grid/threshold reselection",
        "whole_cold_execution": "NOT_RUN", "official_input_execution": "NOT_RUN",
    }
    write(code / "frozen.json", frozen)
    (code / "requirements.txt").write_text(
        "numpy==2.3.5\npandas==3.0.1\nscipy==1.18.0\nscikit-learn==1.9.0\n"
        "catboost==1.2.10\njoblib==1.5.3\npyarrow==25.0.1\n", encoding="utf-8")
    write(code / "manifest.json", {p.relative_to(code).as_posix(): sha(p)
                                  for p in sorted(code.rglob("*")) if p.is_file()})
    return {"status": "CODE_ONLY_PREPARED_NOT_EXECUTED", "path": str(destination),
            "manifest_sha256": sha(code / "manifest.json"), "fits": 0, "official_rows": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    print(json.dumps(build(parser.parse_args().destination.resolve())))
