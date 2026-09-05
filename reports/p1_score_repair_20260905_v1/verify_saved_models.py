"""Read-only model reload/OOF audit; never calls fit or accesses official input."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "scripts/run_p1_score_repair_20260905_v1.py"
SPEC = importlib.util.spec_from_file_location("p1_repair", PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def main():
    artifacts = ROOT / "artifacts" / RUNNER.RUN
    result = json.loads((artifacts / "terminal_result.json").read_text(encoding="utf-8"))
    if result["status"] == "TERMINAL_TECHNICAL_FAILURE":
        raise RuntimeError("training is not successfully terminal")
    cfg = json.loads((artifacts / "contract.json").read_text(encoding="utf-8"))
    train_path = Path(os.environ["P1_DATA_DIR"]) / "train.csv"
    assert RUNNER.sha(train_path) == result["train_sha256"]
    assert RUNNER.sha(PATH) == result["runner_sha256"]
    frame = pd.read_csv(train_path, usecols=RUNNER.RAW + ["label", "anomaly_type"])
    frame["row_id"] = np.arange(len(frame))
    frame.sort_values(["station", "layer", "time"], kind="stable", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    output = {
        "status": "PASS",
        "training_fits": 0,
        "official_rows": 0,
        "method": "reload all 9 outer saved models, rebuild isolated features, compare exact saved OOF probability and repeat inference",
        "models": [],
        "corrected_probability_diagnostics": {},
    }
    oof_parts = []
    for fold in cfg["folds"]:
        name = fold["name"]
        oof = pd.read_parquet(artifacts / f"{name}_intact_oof.parquet")
        oof_parts.append(oof)
        times = pd.to_datetime(frame.time, utc=True)
        evaluation = frame.loc[
            (times >= pd.Timestamp(fold["start"])) & (times < pd.Timestamp(fold["end"]))
        ].reset_index(drop=True)
        assert evaluation.row_id.to_list() == oof.row_id.to_list()
        assert evaluation.label.to_list() == oof.label.to_list()
        balanced_package = joblib.load(artifacts / f"{name}_outer_balanced.joblib")
        bundles = RUNNER.feature_pair(evaluation, balanced_package["train_stats"], cfg)
        output["corrected_probability_diagnostics"][name] = {}
        for model_name in ["original", "balanced", "flank"]:
            model_path = artifacts / f"{name}_outer_{model_name}.joblib"
            expected = next(
                x
                for x in result["fit_receipts"]
                if x["fold"] == name and x["stage"] == "outer" and x["model"] == model_name
            )
            assert RUNNER.sha(model_path) == expected["model_sha256"]
            package = joblib.load(model_path)
            assert joblib.hash(package["train_stats"]) == joblib.hash(
                balanced_package["train_stats"]
            )
            matrix = package["encoder"].transform(bundles[1 if model_name == "flank" else 0])
            prediction = package["model"].predict_proba(matrix)[:, 1]
            repeated = package["model"].predict_proba(matrix)[:, 1]
            saved = oof[model_name + "_probability"].to_numpy()
            exact = np.array_equal(prediction, saved)
            repeat_exact = np.array_equal(prediction, repeated)
            if not exact or not repeat_exact:
                output["status"] = "FAIL"
            output["models"].append(
                {
                    "fold": name,
                    "model": model_name,
                    "rows": len(evaluation),
                    "saved_oof_exact": exact,
                    "repeat_exact": repeat_exact,
                    "max_abs_diff": float(np.max(np.abs(prediction - saved))),
                    "sha256": RUNNER.sha(model_path),
                }
            )
            output["corrected_probability_diagnostics"][name][model_name] = RUNNER.diagnostic(
                evaluation, oof[model_name].to_numpy(), oof.selected_control.to_numpy(), saved
            )
            output["corrected_probability_diagnostics"][name][model_name]["average_precision"] = (
                float(average_precision_score(evaluation.label, saved))
            )
    pooled = pd.concat(oof_parts, ignore_index=True)
    output["pooled_average_precision"] = {
        name: float(average_precision_score(pooled.label, pooled[name + "_probability"]))
        for name in ["original", "balanced", "flank"]
    }
    output["probability_diagnostic_note"] = (
        "Use these original/balanced/flank probability summaries. Runner control/union long_probability fields used balanced probabilities and are NOT representative of XGB or OR. F1/counts/selection are unaffected. OR has no defined single raw probability."
    )
    output["qa_npz_sha256"] = RUNNER.sha(artifacts / "qa_oof.npz")
    output["result_sha256"] = RUNNER.sha(artifacts / "terminal_result.json")
    RUNNER.write_json(Path(__file__).parent / "saved-model-reload-qa.json", output)
    print(
        json.dumps(
            {
                "status": output["status"],
                "outer_models_checked": len(output["models"]),
                "training_fits": 0,
                "official_rows": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
