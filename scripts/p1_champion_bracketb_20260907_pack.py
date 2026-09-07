"""Build a new original-component P1 package replacing only B with trained B107."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
ID = "p1_champion_bracketb_20260907_v1"
OUT = ROOT / "artifacts" / ID


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def adapt_runner(runner):
    replacements = [
        (
            "encoder = TabularEncoder().fit(bundle, np.arange(len(frame)))",
            'extend = module("owned_bracket", CODE / "bracket_features.py").extend\n    bundle = extend(bundle, frame)\n    encoder = TabularEncoder().fit(bundle, np.arange(len(frame)))',
        ),
        ("random_state=seed, n_jobs=8,", "random_state=seed, n_jobs=4,"),
        ('os.environ[key] = "8"', 'os.environ[key] = "4"'),
        (
            'x = b_model["encoder"].transform(bundle)',
            'extend = module("owned_bracket", CODE / "bracket_features.py").extend\n    x = b_model["encoder"].transform(extend(bundle, frame))',
        ),
        (', "cap_seconds": 21600', ', "cap_seconds": None'),
        (",\n                               timeout=max(1, 21600-(time.time()-started))", ""),
    ]
    for before, after in replacements:
        if runner.count(before) != 1:
            raise ValueError("Original source layout drift; adapter substitution ambiguous")
        runner = runner.replace(before, after)
    compile(runner, "new_P1_run.py", "exec")
    return runner


def build(destination):
    result = read(OUT / "terminal_result.json")
    qa = read(OUT / "independent-qa.json")
    if result["status"] != "TRAIN_AND_INTERNAL_COMPLETE_QA_PENDING" or qa["status"] != "PASS":
        raise ValueError("Completed internal training and independent QA required")
    if sha(OUT / "models/full_bracket.joblib") != qa["full_B_aggregate_sha256"]:
        raise ValueError("Completed B artifact differs from independent QA")
    base = Path("C:/Users/cedis/Documents/OceanFinalRelease_20260907/P1/SAVED_MODELS")
    pins = read(base / "source-manifest.json")["files"]
    original_training = read(base / "03_model/tree/training-result.json")
    if sha(base / "03_model/tree/O.joblib") != original_training["O_sha256"]:
        raise ValueError("Original full O model receipt mismatch")
    for name, expected in pins.items():
        if sha(base / name) != expected:
            raise ValueError("Preserved original source pin mismatch: " + name)
    destination.mkdir(parents=True, exist_ok=False)
    for folder in ("01_data", "02_code", "03_model", "04_logs", "05_answer", "06_docs"):
        (destination / folder).mkdir()
    shutil.copytree(
        base / "02_code",
        destination / "02_code",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(
        base / "03_model",
        destination / "03_model",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "ATTEMPT_LOCK.json", "*.log"),
    )
    shutil.copy2(base / "contract.json", destination / "contract.json")
    # Model replacement is confined to this newly created package. Old source/weights stay intact.
    shutil.copy2(OUT / "models/full_bracket.joblib", destination / "03_model/tree/B.joblib")
    cfg = read(ROOT / "configs/experiments" / f"{ID}.json")
    code = destination / "02_code"
    bracket_source = ROOT / "scripts/run_p1_bracket_forward_20260906_v1.py"
    tree = ast.parse(bracket_source.read_text(encoding="utf-8"))
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "bracket_features"
    )
    helper = """from types import SimpleNamespace
import numpy as np
import pandas as pd
from p1_qc.features import FeatureBundle
KEYS = ["station", "year", "layer", "time"]
RAW = KEYS + ["temp", "psal", "depth"]
def segments(frame):
    t = pd.to_datetime(frame.time, utc=True)
    return (frame.station.ne(frame.station.shift()) | frame.layer.ne(frame.layer.shift()) | t.diff().ne(pd.Timedelta(minutes=10))).cumsum()
old = SimpleNamespace(KEYS=KEYS, RAW=RAW, segments=segments)
"""
    helper += ast.unparse(node) + "\n\ndef extend(base, frame):\n"
    helper += """    raw = frame[RAW].reset_index(drop=True)
    order = raw.sort_values(["station", "layer", "time"], kind="stable").index.to_numpy()
    extra = bracket_features(raw.iloc[order].reset_index(drop=True), {"windows_hours": [6,24,72], "flank_hours": 1})
    extra.index = order
    features = pd.concat([base.frame, extra.reindex(raw.index)], axis=1)
    if len(base.feature_columns) != 80 or features.shape[1] != 107:
        raise ValueError("Original 80 + bracket 27 feature contract")
    return FeatureBundle(features, tuple(features), base.categorical_columns)
"""
    (code / "bracket_features.py").write_text(helper, encoding="utf-8")
    runner = (code / "run.py").read_text(encoding="utf-8")
    runner = adapt_runner(runner)
    (code / "run.py").write_text(runner, encoding="utf-8")
    receipt = read(destination / "03_model/tree/training-result.json")
    receipt.update(
        B_sha256=sha(destination / "03_model/tree/B.joblib"),
        new_B_fits=3,
        inherited_O_fits=1,
        features_B=107,
        source_experiment=ID,
        changed_component="B only",
    )
    (destination / "03_model/tree/training-result.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    (destination / "03_model/tree/qa.json").write_text(
        json.dumps({"status": "PASS", "basis": qa}), encoding="utf-8"
    )
    (destination / "06_docs/B_internal_qa.json").write_text(
        json.dumps(qa, indent=2), encoding="utf-8"
    )
    (destination / "06_docs/B_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    modelpins = {
        str(p.relative_to(destination)): sha(p)
        for p in (destination / "03_model").rglob("*")
        if p.is_file()
    }
    (destination / "06_docs/model-lineage.json").write_text(
        json.dumps(
            {
                "files": modelpins,
                "original_model_source": str(base),
                "replacement": "full_bracket.joblib",
                "model_fit_count_new": 3,
                "inherited_O_and_MS": "exact preserved source-retrained models, no inherited answer inputs",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    sourcepins = {
        str(p.relative_to(destination)).replace("\\", "/"): sha(p)
        for p in code.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    sourcepins["contract.json"] = sha(destination / "contract.json")
    for filename, stages in (
        ("TRAIN.ipynb", ["all"]),
        ("SAVED_PREDICT.ipynb", ["infer-tree", "infer-ms", "combine"]),
    ):
        notebook = nbformat.v4.new_notebook()
        notebook.cells = [
            nbformat.v4.new_markdown_cell(
                "# P1 B107 replacement\nRun in a new extracted package. TRAIN performs seven fresh fits; "
                "SAVED_PREDICT performs zero fits. Existing cold O/MS evidence is not a new full-cold run. "
                "Set P1_DATA_DIR before starting. Never run both entrypoints in the same used folder."
            ),
            nbformat.v4.new_code_cell(
                "import os, sys, subprocess\nfrom pathlib import Path\n"
                "package = Path.cwd().resolve()\ndata = Path(os.environ['P1_DATA_DIR']).resolve()\n"
                "assert (package/'02_code/run.py').is_file()\n"
            ),
        ]
        for stage in stages:
            notebook.cells.append(
                nbformat.v4.new_code_cell(
                    f"with (package/'04_logs/{stage}.notebook.log').open('x', encoding='utf-8') as log:\n"
                    f"    subprocess.run([sys.executable, '-I', '-B', str(package/'02_code/run.py'), {stage!r}, '--data', str(data)], "
                    "cwd=package, check=True, stdout=log, stderr=subprocess.STDOUT)\n"
                    f"print({stage!r} + ' completed')"
                )
            )
        notebook.metadata.kernelspec = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
        nbformat.validate(notebook)
        nbformat.write(notebook, destination / filename)
        sourcepins[filename] = sha(destination / filename)
    (destination / "source-manifest.json").write_text(
        json.dumps({"files": sourcepins}, indent=2), encoding="utf-8"
    )
    (destination / "README.md").write_text(
        "# P1 original composition with B107 replacement\n\n"
        "No current official score. Only B changes: original 80 features + 27 bracket features, original three seeds, event/day weights and decoder. "
        "O, MS, six historical cell literals and GI source remain unchanged. Historical cell selection algorithm remains unrecovered; no new selection claim.\n\n"
        "Set P1_DATA_DIR to distributed P1 folder. Source-only: python 02_code/run.py all --data <folder> (7 fresh fits). "
        "Saved: infer-tree, infer-ms, combine stages. Source-only training has not been rerun for this composed package; existing original O/MS cold evidence plus new B training and independent replay are separate claims. "
        "General official six-hour applicability unverified; no new wall-time cutoff. No external or hidden inputs.\n",
        encoding="utf-8",
    )
    sourcepins["README.md"] = sha(destination / "README.md")
    (destination / "source-manifest.json").write_text(
        json.dumps({"files": sourcepins}, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "BUILT_INFERENCE_PENDING",
                "package": str(destination),
                "source_manifest_sha256": sha(destination / "source-manifest.json"),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    build(parser.parse_args().destination.resolve())
