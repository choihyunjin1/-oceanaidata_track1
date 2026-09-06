"""Independent count and day-cluster arithmetic; no training or prediction code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NAME = "p1_trainfit_postpolicy_20260906_v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(y, p):
    return np.array([np.count_nonzero((y == 1) & (p == 1)),
                     np.count_nonzero((y == 0) & (p == 1)),
                     np.count_nonzero((y == 1) & (p == 0))], dtype=np.int64)


def f1(stat):
    denominator = 2 * stat[0] + stat[1] + stat[2]
    return float(2 * stat[0] / denominator) if denominator else 0.0


def main():
    report = ROOT / "reports" / NAME
    result_path = report / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config_path = ROOT / "configs/experiments" / (NAME + ".json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    old_cfg = json.loads((ROOT / "configs/experiments/p1_tuning_twosided_20260906_v1.json").read_text())
    # Find the unchanged evaluation config through its source runner's base config.
    base_cfg = json.loads((ROOT / "configs/experiments/p1_bracket_forward_20260906_v1.json").read_text())
    contract = json.loads((ROOT / base_cfg["evaluation_contract"]).read_text(encoding="utf-8"))
    frame_path = ROOT / "artifacts" / NAME / "oof.parquet"
    frame = pd.read_parquet(frame_path)
    checks = {
        "terminal_complete": result["status"] == "COMPLETE",
        "runner_pin": digest(ROOT / "scripts" / ("run_" + NAME + ".py")) == result["runner_sha256"],
        "config_pin": digest(config_path) == result["config_sha256"],
        "oof_pin": digest(frame_path) == result["oof_sha256"],
        "unique_rows": not frame.row_id.duplicated().any(),
        "binary_labels_predictions": all(frame[c].isin([0, 1]).all() for c in ["label", "control", "range", "cell", "combined"]),
        "range_never_removes_anchor": bool(frame.range.ge(frame.control).all()),
        "combination_never_removes_cell": bool(frame.combined.ge(frame.cell).all()),
        "no_new_backbone_fits": result["new_backbone_fits"] == 0,
        "zero_official_hidden_csv_upload": all(result[k] == 0 for k in ("official_rows", "hidden_rows", "csv_written", "uploads")),
        "normal_rows_preserved": result["normal_rows_removed"] == 0 and not result["normal_weights_changed"],
        "source_config_pinned": old_cfg["source_v1_config_sha256"] == digest(ROOT / "configs/experiments/p1_bracket_forward_20260906_v1.json"),
    }
    for path, expected in config["source_pins"].items():
        checks["source:" + path] = digest(ROOT / path) == expected
    previous = pd.read_parquet(ROOT / "artifacts/p1_tuning_twosided_20260906_v1/oof.parquet")
    checks["baseline_keys_exact"] = frame[["row_id", "station", "layer", "time", "label", "fold"]].equals(previous[["row_id", "station", "layer", "time", "label", "fold"]])
    checks["baseline_prediction_exact"] = np.array_equal(frame.control, previous.candidate)
    dates = pd.to_datetime(frame.time, utc=True)
    primary_spec = contract["P1"]["primary"]
    primary = frame.loc[(dates >= pd.Timestamp(primary_spec["start"])) & (dates < pd.Timestamp(primary_spec["end"]))]
    settings = contract["common"]["bootstrap"]
    for variant in ("range", "cell", "combined"):
        ref = result["comparisons"][variant]
        for name, rows in (("primary_H1_2025", primary), ("all_twosided", frame)):
            for arm, column in (("control", "control"), ("candidate", variant)):
                stat = counts(rows.label.to_numpy(), rows[column].to_numpy())
                saved = ref["metrics"][name][arm]
                checks[f"{variant}:{name}:{arm}:counts"] = stat.tolist() == [saved[k] for k in ("tp", "fp", "fn")]
                checks[f"{variant}:{name}:{arm}:f1"] = abs(f1(stat) - saved["f1"]) < 1e-14
        for i, row in enumerate(ref["slices"]):
            subset = frame
            for key in ("fold", "station", "layer", "known_station_layer"):
                if key in row:
                    subset = subset.loc[subset[key].eq(row[key])]
            delta = f1(counts(subset.label.to_numpy(), subset[variant].to_numpy())) - f1(counts(subset.label.to_numpy(), subset.control.to_numpy()))
            checks[f"{variant}:slice{i}"] = abs(delta - row["delta_f1"]) < 1e-14
        # Independently group by local calendar day, preserving first appearance.
        day = pd.to_datetime(primary.time, utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
        blocks = [np.flatnonzero(day.to_numpy() == value) for value in pd.unique(day)]
        y, b, c = (primary[name].to_numpy() for name in ("label", "control", variant))
        bs = np.array([counts(y[ix], b[ix]) for ix in blocks])
        cs = np.array([counts(y[ix], c[ix]) for ix in blocks])
        rng = np.random.default_rng(settings["seed"])
        differences = []
        for _ in range(settings["resamples"]):
            draws = rng.integers(0, len(blocks), len(blocks))
            multiplicity = np.bincount(draws, minlength=len(blocks))
            differences.append(f1(multiplicity @ cs) - f1(multiplicity @ bs))
        differences = np.array(differences)
        saved = ref["paired_day_bootstrap"]
        checks[variant + ":bootstrap_clusters"] = len(blocks) == saved["n_clusters"]
        checks[variant + ":ci90"] = bool(np.allclose(np.quantile(differences, settings["ci_quantiles"]), saved["ci90"], rtol=0, atol=1e-14))
        checks[variant + ":p_improve"] = float((differences > 0).mean()) == saved["p_improve"]
        improved = f1(counts(y, c)) > f1(counts(y, b))
        checks[variant + ":decision"] = (ref["decision"] == "MEAN_GAIN_CANDIDATE_RETAINED") == improved
    checks["combined_eligibility"] = result["comparisons"]["combined"]["eligible_after_separate_checks"] == all(result["comparisons"][v]["paired_day_bootstrap"]["delta_candidate_minus_control"] > 0 for v in ("range", "cell"))
    failed = [name for name, value in checks.items() if not value]
    output = {"status": "PASS" if not failed else "FAIL", "passed": len(checks) - len(failed),
              "total": len(checks), "failed_checks": failed, "checks": checks,
              "result_sha256": digest(result_path), "new_backbone_fits": 0,
              "scope": "Independent TP/FP/FN, slice and day-cluster bootstrap arithmetic, pins and baseline alignment. Fitting provenance covered by source QA and separate replay; not an OS-wide access audit."}
    with (report / "independent-qa.json").open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, default=bool)
    print(json.dumps({k: v for k, v in output.items() if k != "checks"}))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
