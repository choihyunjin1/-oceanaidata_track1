"""Fast local-only strength audit after the first OAS official probe.

The script does not read official scores, test files, submissions, or answers.
It compares a small predeclared family of scalar and layerwise OAS blend
strengths on the already-existing exposed-block OOF artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from p2_restore.profile_projection import project_profiles_vectorized, public_endpoint_frame


OOF = Path("artifacts/p2_oas_conditional_profile_20260827_v3/oof.parquet")
OBS = Path(r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore\observations.csv")
OUTPUT = Path("artifacts/p2_oas_strength_followup_20260827_v1/result.json")

RECIPES = {
    "scalar_0.10": {2: 0.10, 3: 0.10, 4: 0.10},
    "scalar_0.15": {2: 0.15, 3: 0.15, 4: 0.15},
    "scalar_0.17": {2: 0.17, 3: 0.17, 4: 0.17},
    "scalar_0.20": {2: 0.20, 3: 0.20, 4: 0.20},
    "scalar_0.25": {2: 0.25, 3: 0.25, 4: 0.25},
    "layerwise_0.10_0.20_0.15": {2: 0.10, 3: 0.20, 4: 0.15},
    "layerwise_0.10_0.25_0.15": {2: 0.10, 3: 0.25, 4: 0.15},
}


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def main():
    frame = pd.read_parquet(OOF)
    observations = pd.read_csv(OBS)
    endpoints = public_endpoint_frame(observations)
    truth = frame["truth"].to_numpy(float)
    reference = frame["reference"].to_numpy(float)
    direction = frame["candidate"].to_numpy(float) - reference
    layer = frame["layer"].to_numpy(int)
    baseline = rmse(truth, reference)
    results = {}
    for name, recipe in RECIPES.items():
        weight = np.array([recipe[int(value)] for value in layer], dtype=float)
        raw = reference + weight * direction
        projected = project_profiles_vectorized(frame[["time", "layer"]], raw, endpoints).prediction
        fold_metrics = {}
        for fold, part in frame.groupby("fold", sort=True):
            ids = part.index.to_numpy(int)
            fold_metrics[str(fold)] = {
                "reference_rmse": rmse(truth[ids], reference[ids]),
                "projected_rmse": rmse(truth[ids], projected[ids]),
                "delta_rmse": rmse(truth[ids], projected[ids]) - rmse(truth[ids], reference[ids]),
            }
        results[name] = {
            "weights": recipe,
            "projected_rmse": rmse(truth, projected),
            "delta_rmse": rmse(truth, projected) - baseline,
            "folds": fold_metrics,
        }
    result = {
        "schema_version": "p2.oas_strength_followup.research.20260827.v1",
        "status": "LOCAL_ONLY_EXPOSED_BLOCKS_NO_OFFICIAL_INPUTS",
        "reference_rmse": baseline,
        "recipes": results,
        "decision_rule": (
            "Prefer a prespecified conservative strength that improves aggregate and the "
            "same-season Sep-Oct block; disclose the Jul-Aug reversal."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
