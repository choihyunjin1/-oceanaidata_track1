"""Development-only AST extraction; print a patch, never execute legacy modules."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
DEST = Path(__file__).resolve().parent
SELECTIONS = [
    ("src/p2_restore/features.py", ["_wide", "_common_features"]),
    ("src/p2_restore/normalized_curvature_residual.py", ["_numeric", "compute_profile_scale"]),
    (
        "scripts/final_submission_20260905/P2/p2_pipeline.py",
        [
            "build_arrays",
            "domain_balanced_weights",
            "observed_temperature_gradient_penalty",
            "predict_model",
        ],
    ),
    (
        "scripts/run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12.py",
        ["VerticalDeepSet"],
    ),
    (
        "scripts/run_p2_score_repair_20260905_v1.py",
        [
            "nominal_baseline",
            "refresh_public",
            "public_frame",
            "block_selection",
            "training_arrays",
            "fit_model",
        ],
    ),
]
parts = [
    '''"""Standalone frozen C3 numerical primitives, no research-repository imports."""
from __future__ import annotations
import time
from types import SimpleNamespace
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
PUBLIC_LAYERS = (1, 5, 6, 7, 8)
TARGET_LAYERS = (2, 3, 4)
TEMPORAL_FEATURES = ("doy_sin", "doy_cos", "hour_sin", "hour_cos", "m2_sin", "m2_cos")
def arrays(frame, actualdepth=False):
    if actualdepth:
        raise ValueError("Only frozen C3 context is packaged")
    result = build_arrays(frame)
    if not all(np.isfinite(value).all() for value in result):
        raise ValueError("nonfinite C3 inputs")
    return result
def make_model(arm, context_features):
    if arm != "v23_blockmask":
        raise ValueError("Only frozen C3 architecture is packaged")
    return VerticalDeepSet(8, context_features, hidden=32)
'''
]
provenance = []
for relative, names in SELECTIONS:
    path = ROOT / relative
    source = path.read_text(encoding="utf-8")
    nodes = {
        node.name: node
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    for name in names:
        node = nodes[name]
        code = ast.get_source_segment(source, node)
        if name == "build_arrays":
            code = code.replace(
                "    from p2_restore.normalized_curvature_residual import build_normalized_curvature_design\n\n",
                "",
            )
            code = code.replace(
                "design = build_normalized_curvature_design(frame)",
                'design = SimpleNamespace(baseline=_numeric(frame, "baseline"), profile_scale=compute_profile_scale(frame))',
            )
        if name == "fit_model":
            code = code.replace(
                'device = torch.device("cuda")', 'device = torch.device(config["device"])'
            )
            code = code.replace(
                '"device": "cuda", "cpu_threads": 1',
                '"device": str(device), "cpu_threads": config["cpu_threads"]',
            )
        parts.append(code)
        provenance.append(
            {
                "source": relative,
                "name": name,
                "line": node.lineno,
                "source_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "original_function_sha256": hashlib.sha256(
                    ast.get_source_segment(source, node).encode()
                ).hexdigest(),
            }
        )
content = "\n\n".join(parts) + "\n"
assert "p2_restore" not in content and "sys.path" not in content
print("*** Begin Patch")
for path, data in [
    (DEST / "02_code/core.py", content),
    (
        DEST / "06_docs/SOURCE_PROVENANCE.json",
        json.dumps(
            {
                "source_functions": provenance,
                "mechanical_changes": [
                    "Remove repository imports and unused branches",
                    "Replace unused normalized-design assembly with identical baseline/scale extraction",
                    "Explicit device and truthful CPU thread receipt; no numerical recipe change",
                ],
            },
            indent=2,
        )
        + "\n",
    ),
]:
    print("*** Add File: " + path.as_posix())
    print("\n".join("+" + line for line in data.splitlines()))
print("*** End Patch")
