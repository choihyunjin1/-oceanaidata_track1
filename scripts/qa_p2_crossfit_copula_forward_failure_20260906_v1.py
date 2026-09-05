"""Independent QA of complete C3 baseline and preserved incomplete candidate attempt."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import run_p2_crossfit_copula_forward_20260906_v1 as m


def main():
    started = time.monotonic()
    cfg, contract, _ = m.settings()
    seal = json.loads(m.SEAL.read_text())
    source = m.install_guard()
    failure = json.loads((m.OUT / "terminal_failure.json").read_text())
    baseline = json.loads((m.REPORT / "baseline-result.json").read_text())
    diagnostic = json.loads((m.REPORT / "technical-diagnostic.json").read_text())
    fits = json.loads((m.OUT / "fit-receipts.json").read_text())
    lock = json.loads((m.OUT / "ATTEMPT_LOCK.json").read_text())
    raw = np.load(m.OUT / "baseline_oof.npz", allow_pickle=False)
    checks = []

    def check(name, condition):
        checks.append({"check": name, "pass": bool(condition)})

    m.torch.set_num_threads(2)
    with m.threadpool_limits(2):
        frame, truth = m.load_population(cfg)
        check(
            "source_and_seals_unchanged",
            m.base.file_hash(source) == cfg["source_sha256"] and m.fingerprints() == seal["hashes"],
        )
        check("new_process", os.getpid() != lock["pid"])
        check(
            "baseline_before_candidates",
            baseline["new_fits"] == 24 and baseline["candidate_fits_before_this_receipt"] == 0,
        )
        check(
            "baseline_oof_hash",
            m.base.file_hash(m.OUT / "baseline_oof.npz") == baseline["oof_sha256"],
        )
        check(
            "full_same_keys_labels",
            len(raw["key"]) == len(set(raw["key"])) == 166268
            and np.array_equal(m.key_array(frame), raw["key"])
            and np.array_equal(truth, raw["truth"]),
        )
        check("backbone_count", len(fits) == failure["completed_backbone_fits"] == 36)
        check(
            "physical_model_count",
            len(list((m.OUT / "03_model").glob("*.pt"))) == 36
            and len(list((m.OUT / "03_model").glob("*.npz"))) == 4,
        )
        check(
            "cpu2_only",
            all(r["device"] == "cpu" and r["cpu_threads"] == 2 and r["epochs"] == 60 for r in fits),
        )
        check(
            "no_official_csv_upload",
            failure["official_access_rows"] == failure["csv_written"] == failure["upload"] == 0,
        )
        check(
            "failure_not_scientific_nogo",
            failure["status"] == "TERMINAL_TECHNICAL_FAILURE"
            and failure["scientific_decision"] == "NOT_ESTIMATED_INCOMPLETE",
        )
        check(
            "roundoff_diagnosis",
            diagnostic["outside_physical_features_exact_equal_including_nan"]
            and diagnostic["outside_baseline_exact_equal"]
            and diagnostic["same_shape_subset_predictions_exact_equal"]
            and diagnostic["outside_prediction_different_rows"] == 1
            and diagnostic["outside_prediction_max_abs_difference"] == 3.552713678800501e-15,
        )
        folds, outage = raw["fold"], raw["outage_mask"]
        scopes = {
            "pooled": np.ones(len(frame), bool),
            "primary_B3": folds == "B3",
            "natural_T5_missing": ~np.isfinite(frame.temp_5),
            "natural_T5_present": np.isfinite(frame.temp_5),
            "outage_interval": outage,
        }
        for fid in [f"B{i}" for i in range(1, 9)]:
            scopes[f"fold_{fid}"] = folds == fid
            scopes[f"outage_{fid}"] = (folds == fid) & outage
        for layer in (2, 3, 4):
            scopes[f"layer_{layer}"] = frame.layer.to_numpy() == layer
            scopes[f"primary_layer_{layer}"] = (folds == "B3") & (frame.layer.to_numpy() == layer)
        for surface, panel in (
            ("natural", baseline["natural"]),
            ("outage", baseline["testmatched"]),
        ):
            pred = raw[f"{surface}_C3"]
            check(f"{surface}/finite", np.isfinite(pred).all())
            for scope, selected in scopes.items():
                entry = panel[scope]["C3"]
                n = int(selected.sum())
                check(f"{surface}/{scope}/n", entry["n"] == n)
                if not n:
                    check(
                        f"{surface}/{scope}/empty",
                        entry["status"] == "NOT_ESTIMABLE_NO_ROWS"
                        and entry["rmse"] is None
                        and entry["sse"] is None,
                    )
                else:
                    errors = pred[selected] - truth[selected]
                    sse = float(np.sum(errors**2))
                    check(
                        f"{surface}/{scope}/sse",
                        np.isclose(sse, entry["sse"], rtol=1e-12, atol=1e-9),
                    )
                    check(
                        f"{surface}/{scope}/rmse",
                        np.isclose(np.sqrt(sse / n), entry["rmse"], rtol=1e-12),
                    )
        check(
            "outside_baseline_exact",
            np.array_equal(raw["natural_C3"][~outage], raw["outage_C3"][~outage]),
        )
        times = pd.DatetimeIndex(frame.time)
        replay_checks = []
        for spec in contract["P2"]["folds"]:
            left, right = pd.Timestamp(spec["start"]), pd.Timestamp(spec["end"])
            valid = np.asarray((times >= left) & (times < right))
            train = np.asarray(
                (times < left - pd.Timedelta(days=7)) | (times >= right + pd.Timedelta(days=7))
            )
            check(f"{spec['id']}/scope", np.array_equal(valid, folds == spec["id"]))
            support = next(r for r in seal["support"]["folds"] if r["fold"] == spec["id"])
            stages = {"outer": train}
            for inner in support["inner"]:
                a, b = pd.Timestamp(inner["start"]), pd.Timestamp(inner["end"])
                iv = np.asarray((times >= a) & (times < b))
                it = (
                    np.asarray(
                        (times < a - pd.Timedelta(days=7)) | (times >= b + pd.Timedelta(days=7))
                    )
                    & train
                )
                stages[inner["id"]] = it
                check(
                    f"{spec['id']}/{inner['id']}/nested_purge",
                    np.all(train[iv]) and not np.any(valid & it) and not np.any(iv & it),
                )
            for row in [r for r in fits if r["fold"] == spec["id"]]:
                selection = stages[row["stage"]]
                check(
                    f"{row['fit_id']}/training_keys",
                    m.key_hash(frame.loc[selection]) == row["train_key_sha256"],
                )
                check(
                    f"{row['fit_id']}/model_hash",
                    m.base.file_hash(m.OUT / "03_model" / f"{row['fit_id']}.pt")
                    == row["model_sha256"],
                )
                check(
                    f"{row['fit_id']}/mass",
                    row["training"]["original_rows"] == int(selection.sum())
                    and np.isclose(row["training"]["training_weight_sum"], selection.sum()),
                )
            local = frame.loc[valid].reset_index(drop=True)
            altered, selected = m.outage_frame(local, spec)
            models = m.load_models(spec["id"], "outer", cfg["seeds"])
            for surface, data in (("natural", local), ("outage", altered)):
                predicted = m.cmean(models, data)
                exact = np.array_equal(predicted, raw[f"{surface}_C3"][valid])
                replay_checks.append(
                    {
                        "fold": spec["id"],
                        "surface": surface,
                        "rows": int(valid.sum()),
                        "exact": exact,
                        "max_error": float(np.max(np.abs(predicted - raw[f"{surface}_C3"][valid]))),
                    }
                )
                check(f"{spec['id']}/{surface}/fresh_replay", exact)
        check("source_unchanged_after_replay", m.base.file_hash(source) == cfg["source_sha256"])
    payload = {
        "status": "PASS_BASELINE_AND_FAILURE_ACCOUNTING_CANDIDATE_INCOMPLETE"
        if all(c["pass"] for c in checks)
        else "FAIL",
        "checks_passed": sum(c["pass"] for c in checks),
        "checks_total": len(checks),
        "checks": checks,
        "baseline_rows": len(frame),
        "baseline_new_backbone_fits": 24,
        "completed_inner_backbone_fits": 12,
        "completed_copula_fit_calls": 4,
        "candidate_final_comparison": "NOT_ESTIMATED_INCOMPLETE",
        "candidate_replay": "NOT_COMPLETED",
        "ci90": "NOT_ESTIMATED_INCOMPLETE_COMPARISON",
        "original_failure_unchanged": failure,
        "new_process_pid": os.getpid(),
        "baseline_replay": replay_checks,
        "new_fits": 0,
        "official_access_rows": 0,
        "csv_written": 0,
        "upload": 0,
        "runtime_seconds": time.monotonic() - started,
        "qa_runner_sha256": m.base.file_hash(Path(__file__)),
        "baseline_result_sha256": m.base.file_hash(m.REPORT / "baseline-result.json"),
        "diagnostic_sha256": m.base.file_hash(m.REPORT / "technical-diagnostic.json"),
    }
    m.save(m.REPORT / "failure-independent-qa.json", payload)
    print(
        json.dumps(
            {k: payload[k] for k in ("status", "checks_passed", "checks_total", "runtime_seconds")}
        )
    )
    if not all(c["pass"] for c in checks):
        raise AssertionError([c for c in checks if not c["pass"]])


if __name__ == "__main__":
    main()
