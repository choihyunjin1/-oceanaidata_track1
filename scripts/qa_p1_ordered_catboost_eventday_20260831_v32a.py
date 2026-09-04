"""Independent read-only QA for the terminal v32a historical experiment."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_p1_mstcn_segment_precision_router_retroaudit_20260829_v1 as e150  # noqa: E402

EXPERIMENT_ID = "p1_ordered_catboost_eventday_20260831_v32a"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
KEYS = ["station", "year", "layer", "time"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def summary(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(prediction, dtype=np.int8)
    return {
        "rows": int(len(y)),
        "positives": int(p.sum()),
        "f1": float(f1_score(y, p)),
        "precision": float(precision_score(y, p, zero_division=1)),
        "recall": float(recall_score(y, p, zero_division=1)),
    }


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(abs(float(left) - float(right)) <= tolerance)


def main() -> int:
    result_path = ARTIFACT / "result.json"
    oof_path = ARTIFACT / "historical_oof.parquet"
    config_path = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
    runner_path = ROOT / "scripts" / f"run_{EXPERIMENT_ID}.py"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frame = pd.read_parquet(oof_path)
    truth = frame["label"].to_numpy(np.int8)
    candidate = frame["candidate_prediction"].to_numpy(np.int8)
    tabular = frame["deployment_prediction"].to_numpy(np.int8)
    checks: dict[str, bool] = {
        "terminal_status_no_go": result["status"] == "NO_GO_INTERNAL_GATE",
        "rows_421032": len(frame) == 421_032,
        "keys_unique": not frame.duplicated(KEYS).any(),
        "folds_exact": set(frame["fold"].unique()) == {"2025_q2", "2025_q3", "2025_q4"},
        "candidate_binary": bool(np.isin(candidate, [0, 1]).all()),
        "candidate_probability_finite": bool(np.isfinite(frame["candidate_probability"]).all()),
        "oof_hash_matches": sha256_file(oof_path) == result["historical_oof"]["sha256"],
        "config_hash_matches": sha256_file(config_path) == result["input_hashes"]["config"],
        "fit_count_three": result["fit_count"] == 3,
        "runtime_within_cap": result["runtime_seconds"] <= config["maximum_runtime_seconds"],
        "official_access_zero": set(result["official_access"].values()) == {0},
        "no_submission_artifacts": not any(
            "submission" in path.name.lower() for path in ARTIFACT.glob("**/*") if path.is_file()
        ),
    }
    pooled_candidate = summary(truth, candidate)
    pooled_tabular = summary(truth, tabular)
    checks.update(
        {
            "pooled_candidate_f1_recomputed": close(
                pooled_candidate["f1"], result["pooled_vs_tabular"]["candidate"]["f1"]
            ),
            "pooled_tabular_f1_recomputed": close(
                pooled_tabular["f1"], result["pooled_vs_tabular"]["reference"]["f1"]
            ),
        }
    )
    fold_recomputed: dict[str, Any] = {}
    for fold in ("2025_q2", "2025_q3", "2025_q4"):
        mask = frame["fold"].eq(fold).to_numpy()
        candidate_summary = summary(truth[mask], candidate[mask])
        tabular_summary = summary(truth[mask], tabular[mask])
        fold_recomputed[fold] = {
            "candidate": candidate_summary,
            "tabular": tabular_summary,
            "delta_f1": candidate_summary["f1"] - tabular_summary["f1"],
        }
        checks[f"{fold}_candidate_f1_recomputed"] = close(
            candidate_summary["f1"], result["by_fold"][fold]["candidate"]["f1"]
        )
        checks[f"{fold}_tabular_f1_recomputed"] = close(
            tabular_summary["f1"], result["by_fold"][fold]["tabular_reference"]["f1"]
        )
    bundles = e150.load_bundles()
    e150_prediction = np.full(len(frame), -1, dtype=np.int8)
    for fold, bundle in bundles.items():
        reference = bundle.frame[[*KEYS, "fold"]].copy()
        reference["prediction"] = bundle.raw_candidate
        mask = frame["fold"].eq(fold).to_numpy()
        aligned = frame.loc[mask, [*KEYS, "fold"]].merge(
            reference, on=[*KEYS, "fold"], how="left", validate="one_to_one", sort=False
        )
        if aligned["prediction"].isna().any():
            raise RuntimeError(f"E150 alignment failed: {fold}")
        e150_prediction[mask] = aligned["prediction"].to_numpy(np.int8)
    q34 = frame["fold"].isin(["2025_q3", "2025_q4"]).to_numpy()
    q34_candidate = summary(truth[q34], candidate[q34])
    q34_e150 = summary(truth[q34], e150_prediction[q34])
    checks["q34_candidate_f1_recomputed"] = close(
        q34_candidate["f1"], result["q3_q4_vs_e150"]["candidate"]["f1"]
    )
    checks["q34_e150_f1_recomputed"] = close(
        q34_e150["f1"], result["q3_q4_vs_e150"]["reference"]["f1"]
    )
    expected_points = 28.909341 + 26.578120867377286 * (
        q34_candidate["f1"] - q34_e150["f1"]
    )
    checks["expected_points_recomputed"] = close(
        expected_points, result["public_score_translation"]["expected_points_center"]
    )
    gates_recomputed = {
        "all_q2_q3_q4_delta_f1_vs_tabular_nonnegative": all(
            item["delta_f1"] >= 0.0 for item in fold_recomputed.values()
        ),
        "pooled_delta_f1_vs_tabular_positive": pooled_candidate["f1"] > pooled_tabular["f1"],
        "pooled_bootstrap_ci90_low_vs_tabular_positive": (
            result["pooled_vs_tabular"]["bootstrap"]["difference_ci90"][0] > 0.0
        ),
        "q3_q4_delta_f1_vs_e150_positive": q34_candidate["f1"] > q34_e150["f1"],
        "q3_q4_bootstrap_ci90_low_vs_e150_positive": (
            result["q3_q4_vs_e150"]["bootstrap"]["difference_ci90"][0] > 0.0
        ),
        "runtime_at_most_seconds": result["runtime_seconds"] <= config["maximum_runtime_seconds"],
        "official_accesses_equal_zero": set(result["official_access"].values()) == {0},
    }
    checks["gates_recomputed_match"] = gates_recomputed == result["gates"]
    checks["runner_has_no_official_materializer"] = all(
        token not in runner_path.read_text(encoding="utf-8")
        for token in ("predict_submission", "write_submission", "validate_submission")
    )
    qa = {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "recomputed": {
            "pooled_candidate": pooled_candidate,
            "pooled_tabular": pooled_tabular,
            "folds": fold_recomputed,
            "q3_q4_candidate": q34_candidate,
            "q3_q4_e150": q34_e150,
            "expected_points_center": expected_points,
            "gates": gates_recomputed,
        },
        "source_hashes": {
            "result": sha256_file(result_path),
            "historical_oof": sha256_file(oof_path),
            "config": sha256_file(config_path),
            "runner": sha256_file(runner_path),
        },
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    output = REPORT / "independent-qa.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": qa["status"], "checks": f"{qa['checks_passed']}/{qa['checks_total']}"}))
    return 0 if qa["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
