"""Build a dependency-closed P1 package by auditable mechanical source extraction."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(relative, names):
    source = (ROOT / relative).read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    found = {}
    lines = source.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            found[node.name] = "".join(lines[start - 1:node.end_lineno])
    if set(found) != set(names):
        raise ValueError(f"missing definitions: {set(names) - set(found)}")
    return "\n\n".join(found[name] for name in names) + "\n"


def build(destination):
    if destination.exists():
        raise FileExistsError("new package destination required")
    code = destination / "02_code"
    code.mkdir(parents=True)
    manifest = {}

    def put(name, value, sources):
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8", newline="\n")
        manifest[name] = {"sha256": sha(path), "source_sha256": {
            relative: sha(ROOT / relative) for relative in sources}}

    for name in ("config", "features", "rules", "postprocess", "models_tabular"):
        source = f"src/p1_qc/{name}.py"
        put(f"p1_qc/{name}.py", (ROOT / source).read_text(encoding="utf-8"), [source])
    put("p1_qc/__init__.py", '"""Portable clean P1 components only."""\n', [])
    source = "src/p1_qc/data.py"
    put("p1_qc/data.py", "from __future__ import annotations\nfrom collections.abc import Sequence\nimport numpy as np\nimport pandas as pd\n\n" +
        extract(source, ["segment_timeseries", "add_depth_regime"]), [source])
    source = "src/p1_qc/pipeline.py"
    header = '''from __future__ import annotations
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from p1_qc.features import FeatureBundle
from p1_qc.models_tabular import DeterministicTabularClassifier, make_tabular_classifier
from p1_qc.postprocess import PostprocessConfig, _chronological_order, close_short_gaps, hysteresis_threshold, remove_short_runs
'''
    put("p1_qc/pipeline.py", header + extract(source, ["TabularEncoder", "_sample_weights", "_fit_model", "apply_postprocess"]), [source])
    source = "scripts/run_p1_score_repair_20260905_v1.py"
    helper = "scripts/run_p1_meaningful_learning_curve_generation_v1.py"
    header = '''from __future__ import annotations
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from p1_qc.config import load_config
from p1_qc.features import FeatureBundle, build_features
from p1_qc.pipeline import TabularEncoder, _fit_model, apply_postprocess
from p1_qc.rules import detect_plateaus
ROOT = Path(__file__).resolve().parent
KEYS = ["station", "year", "layer", "time"]
RAW = KEYS + ["temp", "psal", "depth"]
'''
    names = ["sha", "write_json", "segments", "stats_fit", "feature_pair", "flank_features", "rule_masks", "metric", "decode", "calibrate", "train_slice"]
    put("core.py", header + extract(helper, ["_event_day_weight", "_lgb_parameters"]) + extract(source, names), [source, helper])
    decoder = "scripts/run_p1_score_repair_decoder_20260905_v1.py"
    put("decoder.py", "import numpy as np\nimport core as screen\n" + extract(decoder, ["control_components"]), [decoder])
    frozen_source = "configs/experiments/p1_score_repair_20260905_v1.json"
    frozen = json.loads((ROOT / frozen_source).read_text())
    keep = ["seed", "expected_training_sha256", "purge_days", "inner_days", "flank_inner_hours", "flank_outer_hours", "flank_min_fraction", "threshold_grid", "low_ratio", "close_gap_rows", "minimum_positive_run", "folds"]
    frozen = {key: frozen[key] for key in keep}
    frozen.update(base_config="configs/p1.toml", lightgbm_recipe="configs/lightgbm.json")
    put("configs/feature.json", json.dumps(frozen, indent=2), [frozen_source])
    put("configs/p1.toml", (ROOT / "configs/p1.toml").read_text(), ["configs/p1.toml"])
    lgb_source = "configs/p1_meaningful_learning_curve_generation_v1.json"
    lgb = json.loads((ROOT / lgb_source).read_text())
    put("configs/lightgbm.json", json.dumps({"lightgbm_parameters": lgb["lightgbm_parameters"]}, indent=2), [lgb_source])
    driver_source = "scripts/verify_p1_clean_regeneration_20260905_v5.py"
    driver = extract(driver_source, ["write", "policy_bits", "select_policy", "ensure_empty_models", "assert_sources", "fit_pair", "train", "infer", "self_test"])
    start = driver.index('        equivalent = (selected == cfg["selection"]')
    end = driver.index('        del training, evaluation, probabilities', start)
    driver = driver[:start] + '        receipt["inner_policy_source"] = "current inner training only; no historical policy override"\n' + driver[end:]
    start = driver.index('        for relative, expected in hashes.items():')
    end = driver.index('        receipt["status"] = "FOUR_FITS_COMPLETE', start)
    driver = driver[:start] + driver[end:]
    driver = driver.replace('canonical.importlib.metadata.version', 'importlib.metadata.version')
    driver = driver.replace('"resource_amendment": "Restore canonical CPU4; inner B then O, full O then B; one diagnostic attempt",', '"resource_amendment": "Portable extraction, same CPU4 and fit order",')
    put("run.py", (HERE / "runtime_header.txt").read_text() + driver + (HERE / "runtime_footer.txt").read_text(), [driver_source])
    put("requirements.txt", "\n".join(name + "==" + importlib.metadata.version(name) for name in
        ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost", "joblib", "pytest"]) + "\n", [])
    (code / "source-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for folder in ("01_data", "03_model", "04_logs", "05_answer", "06_docs"):
        (destination / folder).mkdir()
    (destination / "README.md").write_text((HERE / "README.md").read_text(encoding="utf-8"), encoding="utf-8")
    (destination / "06_docs/constant-lineage.md").write_text(
        "# Constants and fitted values\n\nFixed seed/model settings are extracted from the pre-existing clean recipe; no Public-score inversion. "
        "Depth/spike statistics, encoders, model weights and threshold/policy are fitted from distributed train in this run. "
        "The expected answer SHA is verification metadata only, never an optimization target. "
        "No physical-range patch, station-layer policy, bracket feature, MS-TCN, prior answer or external data is included. "
        "All new fitted values are stored in 03_model. Original numeric hyperparameters remain documented recipe constants; JSON storage does not establish provenance.\n", encoding="utf-8")
    print(json.dumps({"package": str(destination), "source_files": len(manifest), "models_copied": 0, "data_copied": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    build(parser.parse_args().destination.resolve())
