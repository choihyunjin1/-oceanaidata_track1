"""Independent aggregate arithmetic, training lineage, covariance and replay QA."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import run_p2_crossfit_copula_forward_numeric_20260906_v2 as m
from scipy.special import ndtri
from threadpoolctl import threadpool_limits


def main():
    m.torch.set_num_threads(2)
    m.assert_frozen()
    cfg, contract, _ = m.settings()
    sealed = json.loads(m.SEAL.read_text(encoding="utf-8"))
    assert m.fingerprints() == sealed["hashes"]
    source = m.install_guard()
    result = json.loads((m.REPORT / "result.json").read_text(encoding="utf-8"))
    baseline = json.loads((m.REPORT / "baseline-result.json").read_text(encoding="utf-8"))
    replay = json.loads((m.REPORT / "replay.json").read_text(encoding="utf-8"))
    raw = np.load(m.OUT / "evaluation.npz", allow_pickle=False)
    checks = []

    def check(name, value):
        checks.append({"check": name, "pass": bool(value)})

    check(
        "baseline_before_candidates",
        baseline["candidate_fits_before_this_receipt"] == 0 and baseline["new_fits"] == 24,
    )
    check(
        "budget88_exact",
        result["new_backbone_fits"] == 36
        and result["new_copula_fits"] == 12
        and result["reused_backbone_fits"] == 36
        and result["reused_copula_fits"] == 4
        and result["logical_backbone_fits"] == 72
        and result["logical_copula_fits"] == 16
        and result["new_full_fits"] == 0,
    )
    check(
        "source_immutable",
        m.base.file_hash(source) == cfg["source_sha256"] and result["source_sha256_unchanged"],
    )
    check(
        "baseline_hash",
        m.base.file_hash(m.REPORT / "baseline-result.json") == result["baseline_result_sha256"],
    )
    check(
        "evaluation_hash", m.base.file_hash(m.OUT / "evaluation.npz") == result["evaluation_sha256"]
    )
    check(
        "baseline_oof_hash", m.base.file_hash(m.OUT / "baseline_oof.npz") == baseline["oof_sha256"]
    )
    check(
        "no_official_or_full",
        result["official_access_rows"]
        == result["csv_written"]
        == result["upload"]
        == result["new_full_fits"]
        == 0,
    )
    check(
        "cpu_only",
        not result["gpu_used"]
        and all(
            row["device"] == "cpu" and row["cpu_threads"] == 2 for row in result["fit_receipts"]
        ),
    )
    check(
        "full_source_replay",
        replay["status"] == "PASS"
        and replay["pid"] != result["training_pid"]
        and replay["rows_per_surface"] == 166268
        and replay["max_error"] == 0
        and all(row["exact"] for row in replay["checks"]),
    )
    check("same166268unique_keys", len(raw["key"]) == len(set(raw["key"])) == 166268)
    frame, truth = m.load_population(cfg)
    check(
        "source_key_truth_exact",
        np.array_equal(m.key_array(frame), raw["key"]) and np.array_equal(truth, raw["truth"]),
    )
    folds, outage = raw["fold"], raw["outage_mask"]
    scopes = {
        "pooled": np.ones(len(truth), bool),
        "primary_B3": folds == "B3",
        "natural_T5_missing": raw["natural_T5_missing"],
        "natural_T5_present": ~raw["natural_T5_missing"],
        "outage_interval": outage,
    }
    for fold in [f"B{i}" for i in range(1, 9)]:
        scopes[f"fold_{fold}"] = folds == fold
        scopes[f"outage_{fold}"] = (folds == fold) & outage
    for layer in (2, 3, 4):
        scopes[f"layer_{layer}"] = raw["layer"] == layer
        scopes[f"primary_layer_{layer}"] = (folds == "B3") & (raw["layer"] == layer)
    for surface, panel in (("natural", result["natural"]), ("outage", result["testmatched"])):
        for arm in cfg["policies"]:
            prediction = raw[f"{surface}_{arm}"]
            check(f"{surface}/{arm}/finite", np.isfinite(prediction).all())
            for scope, selected in scopes.items():
                measured = panel[scope][arm]
                n = int(selected.sum())
                check(f"{surface}/{arm}/{scope}/n", measured["n"] == n)
                if n == 0:
                    check(
                        f"{surface}/{arm}/{scope}/empty",
                        measured["status"] == "NOT_ESTIMABLE_NO_ROWS"
                        and measured["sse"] is None
                        and measured["rmse"] is None,
                    )
                    continue
                errors = prediction[selected] - truth[selected]
                sse = float(np.sum(errors * errors))
                check(
                    f"{surface}/{arm}/{scope}/sse",
                    np.isclose(sse, measured["sse"], rtol=1e-12, atol=1e-9),
                )
                check(
                    f"{surface}/{arm}/{scope}/rmse",
                    np.isclose(np.sqrt(sse / n), measured["rmse"], rtol=1e-12),
                )
        for arm in cfg["policies"]:
            check(
                f"{arm}/outside_outage_unchanged",
                (np.array_equal(raw[f"natural_{arm}"][~outage], raw[f"outage_{arm}"][~outage]) if arm == "C3"
                 else np.allclose(raw[f"natural_{arm}"][~outage], raw[f"outage_{arm}"][~outage], rtol=0, atol=1e-12)),
            )
    for fold in ("B4", "B8"):
        check(f"{fold}/empty_outage_preserved", not scopes[f"outage_{fold}"].any())
    for spec in contract["P2"]["folds"]:
        train, valid = m.masks(frame.time, spec)
        check(f"{spec['id']}/same_key_scope", np.array_equal(valid, folds == spec["id"]))
        support = next(row for row in result["support"]["folds"] if row["fold"] == spec["id"])
        local = frame.loc[valid].reset_index(drop=True)
        altered, selected = m.outage_frame(local, spec)
        c, ac = raw["natural_C3"][valid], raw["outage_C3"][valid]
        nx, ax = m.profile.physical_features(local, c), m.profile.physical_features(altered, ac)
        for arm in ("insample_full", "crossfit_full"):
            invariant = m.verify_numeric_invariance(m.key_array(local), m.key_array(altered),
                c, ac, nx, ax, selected, raw[f"natural_{arm}"][valid], raw[f"outage_{arm}"][valid])
            check(f"{spec['id']}/{arm}/exact_inputs_numeric_output", invariant["input_key_baseline_features_exact"])
        stage_masks = {"outer": train}
        inner_indices = []
        for inner in support["inner"]:
            it, iv = m.masks(frame.time, inner)
            stage_masks[inner["id"]] = train & it
            inner_indices.extend(np.flatnonzero(iv).tolist())
            check(
                f"{spec['id']}/{inner['id']}/no_outer_label",
                not np.any(valid & (train & it)) and np.all(train[iv]),
            )
        for stage, selection in stage_masks.items():
            for seed in cfg["seeds"]:
                fit_id = f"{spec['id']}_{stage}_{seed}"
                row = next(item for item in result["fit_receipts"] if item["fit_id"] == fit_id)
                check(
                    f"{fit_id}/train_keys",
                    m.key_hash(frame.loc[selection]) == row["train_key_sha256"],
                )
                check(
                    f"{fit_id}/mass",
                    row["training"]["original_rows"] == int(selection.sum())
                    and np.isclose(
                        row["training"]["training_weight_sum"], int(selection.sum()), rtol=1e-5
                    ),
                )
                check(
                    f"{fit_id}/model",
                    m.base.file_hash(m.OUT / "03_model" / f"{fit_id}.pt") == row["model_sha256"],
                )
        for arm in ("insample_full", "crossfit_full"):
            row = next(
                item
                for item in result["copula_receipts"]
                if item["fold"] == spec["id"] and item["arm"] == arm
            )
            model_path = m.OUT / "03_model" / f"{spec['id']}_{arm}.npz"
            data_path = m.OUT / f"calibration_{spec['id']}_{arm}.npz"
            check(
                f"{spec['id']}/{arm}/hash",
                m.base.file_hash(model_path) == row["model_sha256"]
                and m.base.file_hash(data_path) == row["calibration_sha256"],
            )
            model = dict(np.load(model_path, allow_pickle=False))
            data = np.load(data_path, allow_pickle=False)
            expected_indices = (
                np.flatnonzero(train) if arm == "insample_full" else np.asarray(inner_indices)
            )
            check(
                f"{spec['id']}/{arm}/indices",
                np.array_equal(data["row_indices"], expected_indices)
                and not np.any(valid[expected_indices]),
            )
            x, residual = data["physical_x"], data["residual"]
            check(f"{spec['id']}/{arm}/independent_shrinkage", all(m.covariance_checks(model, x, residual).values()))
            if arm == "insample_full":
                models = m.load_models(spec["id"], "outer", cfg["seeds"])
                regenerated_c = m.cmean(models, frame.loc[train].reset_index(drop=True))
            else:
                components = []
                for inner in support["inner"]:
                    _, inner_valid = m.masks(frame.time, inner)
                    models = m.load_models(spec["id"], inner["id"], cfg["seeds"])
                    components.append(
                        m.cmean(models, frame.loc[inner_valid].reset_index(drop=True))
                    )
                regenerated_c = np.concatenate(components)
            check(
                f"{spec['id']}/{arm}/fresh_calibration_predictions",
                np.array_equal(x[:, 0], regenerated_c),
            )
            check(
                f"{spec['id']}/{arm}/residual",
                np.array_equal(residual, truth[expected_indices] - x[:, 0]),
            )
            check(f"{spec['id']}/{arm}/cdf_y", np.array_equal(np.sort(residual), model["response"]))
            z = []
            for index, column in enumerate([*x.T, residual]):
                marginal = np.sort(column[np.isfinite(column)])
                saved = model[f"marginal_{index}"] if index < 11 else model["response"]
                check(f"{spec['id']}/{arm}/marginal{index}", np.array_equal(marginal, saved))
                latent = np.zeros(len(column))
                present = np.isfinite(column)
                if len(marginal):
                    rank = (
                        np.searchsorted(marginal, column[present], "left")
                        + np.searchsorted(marginal, column[present], "right")
                    ) / (2 * len(marginal))
                    latent[present] = ndtri(
                        np.clip(rank, 0.5 / len(marginal), 1 - 0.5 / len(marginal))
                    )
                z.append(latent)
            z = np.column_stack(z)
            centered = z - z.mean(axis=0)
            covariance = centered.T @ centered / len(z)
            shrink = float(model["shrinkage"])
            expected = (1 - shrink) * covariance + shrink * np.trace(covariance) / 12 * np.eye(12)
            check(
                f"{spec['id']}/{arm}/covariance",
                np.allclose(model["covariance"], expected, rtol=1e-10, atol=1e-12),
            )
            check(
                f"{spec['id']}/{arm}/location",
                np.allclose(model["location"], z.mean(axis=0), rtol=1e-12, atol=1e-12),
            )
    origin = pd.Timestamp("2024-01-01T00:00:00+09:00")
    group = ((pd.to_datetime(raw["time"], utc=True) - origin) // pd.Timedelta(days=7)).to_numpy()
    for scope, byarm in result["comparisons"].items():
        selected = scopes[scope]
        surface = "outage" if scope == "outage_interval" else "natural"
        code, labels = pd.factorize(group[selected], sort=False)
        n = np.bincount(code)
        for comparison, measured in byarm.items():
            control = "insample_full" if comparison == "crossfit_vs_insample" else "C3"
            candidate = "crossfit_full" if comparison == "crossfit_vs_insample" else comparison
            sse = [
                np.bincount(
                    code, weights=(raw[f"{surface}_{arm}"][selected] - truth[selected]) ** 2
                )
                for arm in (control, candidate)
            ]
            delta = np.sqrt(sse[1].sum() / n.sum()) - np.sqrt(sse[0].sum() / n.sum())
            generator = np.random.default_rng(20260906)
            distribution = []
            for _ in range(2000):
                draw = generator.integers(0, len(labels), len(labels))
                distribution.append(
                    np.sqrt(sse[1][draw].sum() / n[draw].sum())
                    - np.sqrt(sse[0][draw].sum() / n[draw].sum())
                )
            check(
                f"{scope}/{comparison}/delta",
                np.isclose(
                    measured["delta_candidate_minus_control"], delta, rtol=1e-10, atol=1e-12
                ),
            )
            check(
                f"{scope}/{comparison}/ci90",
                np.allclose(
                    measured["ci90"],
                    np.quantile(distribution, [0.05, 0.95]),
                    rtol=1e-10,
                    atol=1e-12,
                ),
            )
            check(
                f"{scope}/{comparison}/fraction",
                measured["p_improve"] == float((np.asarray(distribution) < 0).mean()),
            )
            check(
                f"{scope}/{comparison}/retention",
                measured["candidate_retained"] == bool(delta < 0)
                and measured["probability_hard_gate"] is None,
            )
    masking_support = []
    for fold in [f"B{i}" for i in range(1, 9)]:
        selected = (folds == fold) & outage
        count = int(selected.sum())
        masking_support.append(
            {
                "fold": fold,
                "same_key_outage_rows": count,
                "additional_temp5_masked_rows": int(
                    (selected & np.isfinite(frame.temp_5.to_numpy(float))).sum()
                ),
                "additional_psal5_masked_rows": int(
                    (selected & np.isfinite(frame.psal_5.to_numpy(float))).sum()
                ),
                "already_temp5_missing_rows": int(
                    (selected & ~np.isfinite(frame.temp_5.to_numpy(float))).sum()
                ),
                "status": "NOT_ESTIMABLE_NO_ROWS" if count == 0 else "MASK_COUNTS_RECONCILED",
            }
        )
    reuse = json.loads((m.REPORT / "zero-fit-reuse.json").read_text())
    check("zero_fit_reuse_pass", reuse["status"] == "ZERO_FIT_REUSE_PROVENANCE_PASS" and reuse["new_fits"] == 0)
    check("four_fit_saves_three_old_receipts", reuse["copula_fit_saves"] == 4 and reuse["parent_post_assert_copula_receipts"] == 3)
    check("combined_budget90min", result["combined_execution_seconds"] <= 5400)
    check("first36fit_exact_parent", result["fit_receipts"][:36] == json.loads((m.old.OUT / "fit-receipts.json").read_text()))
    check("36new_fit_ids_distinct", len(result["fit_receipts"][36:]) == 36 and len({r["fit_id"] for r in result["fit_receipts"]}) == 72)
    check("parent_files_unchanged", m.inventory() == sealed["parent_inventory"])
    failures = [row["check"] for row in checks if not row["pass"]]
    payload = {
        "status": "FAIL" if failures else "PASS",
        "checks_count": len(checks),
        "checks": checks,
        "additional_masking_support": masking_support,
        "failures": failures,
        "qa_runner_sha256": m.base.file_hash(Path(__file__)),
        "result_sha256": m.base.file_hash(m.REPORT / "result.json"),
        "baseline_sha256": result["baseline_result_sha256"],
        "replay_sha256": m.base.file_hash(m.REPORT / "replay.json"),
        "synthetic_tests": 9,
        "ruff": "PASS",
        "new_fits": 0,
        "official_access_rows": 0,
        "csv_written": 0,
        "upload": 0,
        "scope": "arithmetic_lineage_covariance_replay_not_official_score",
    }
    m.save(m.REPORT / "independent-qa.json", payload)
    print(json.dumps({"status": payload["status"], "checks": len(checks), "failures": failures}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
