"""Independent full4 metadata/probe gate after a fresh saved-model replay; zero fits."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import importlib.util
import os
import time
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("tree_full_qa_source", Path(__file__).with_name("tree.py"))
t = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(t)
import numpy as np


def expected_probe_ids(data):
    return data.row_id.iloc[np.linspace(0, len(data) - 1, 2048, dtype=int)].to_numpy()


def verify(output, historical, seal_path):
    started = time.monotonic()
    result = t.read_json(output / "terminal_result.json")
    if result.get("status") != "COMPLETE" or result.get("phase") != "full":
        raise ValueError("full terminal COMPLETE required before any source/model reads")
    if (output / "independent-qa.json").exists():
        raise FileExistsError("full QA receipt already exists")
    seal = t.check_seal(seal_path)
    t.verify_artifacts(output, result)
    cfg = seal["config"]
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    historical_result = t.read_json(historical / "terminal_result.json")
    t.verify_artifacts(historical, historical_result)
    historical_qa = t.read_json(historical / "independent-qa.json")
    check("historical_24fit_QA_PASS", historical_result["completed_fits"] == 24 and
          historical_qa["status"] == "PASS" and historical_qa["terminal_result_sha256"] ==
          result["historical_result_sha256"] == t.core.sha(historical / "terminal_result.json"))
    check("same_source_seal", result["seal_sha256"] == historical_result["seal_sha256"] == t.core.sha(seal_path))
    check("four_attempted_completed_models", result["fit_cap"] == result["attempted_fits"] ==
          result["completed_fits"] == len(result["fit_receipts"]) == 4)
    replay = t.read_json(output / "fresh-replay-qa.json")
    check("fresh_PID_full_probe_replay", replay["status"] == "PASS" and replay["check_count"] == 2 and
          replay["fits"] == 0 and replay["pid"] != result["worker_pid"] and
          replay["terminal_result_sha256"] == t.core.sha(output / "terminal_result.json"))
    check("selector_exact_Q4_inner_bytes", t.core.sha(output / "selector.json") ==
          t.core.sha(historical / "2025_q4/selector.json"))
    check("official_hidden_CSV_upload_zero", all(result[k] == 0 for k in
                                               ("official_rows", "hidden_rows", "csv_written", "upload")))
    check("combined_workflow_under_six_hours", time.time() - historical_result["started_unix"] < 21600)
    data = t.training_source(cfg)
    check("all_train_population", len(data) == 776706 and int(data.label.sum()) == 32126)
    bundle = t.core.joblib.load(output / "full/models/preprocess.joblib")
    expected = [f"O_{cfg['O_seed']}.joblib"] + [f"B_{seed}.joblib" for seed in cfg["B_seeds"]]
    check("exact_O1_B3_order", bundle["model_files"] == expected)
    check("exact_model_file_count", sorted(p.name for p in (output / "full/models").glob("[OB]_*.joblib")) == sorted(expected))
    check("80_features", len(bundle["feature_columns"]) == len(bundle["encoder"].feature_columns) == 80)
    records = {Path(r["path"].replace("\\", "/")).name: r for r in result["fit_receipts"]}
    for filename in expected:
        path = output / "full/models" / filename
        arm, seed = filename[0], int(filename[2:-7])
        model = t.core.joblib.load(path)
        actual = model.model if arm == "O" else model
        params = actual.get_params()
        trees = actual.get_booster().num_boosted_rounds() if arm == "O" else actual.booster_.num_trees()
        check(filename + "/700trees_CPU4_seed_80features", trees == 700 and params["n_jobs"] == 4 and
              params["random_state"] == seed and actual.n_features_in_ == 80)
        fixed = cfg["xgboost_parameters"] if arm == "O" else cfg["lightgbm_parameters"]
        check(filename + "/all_fixed_parameters", all(params[k] == v for k, v in fixed.items()))
        if arm == "B":
            check(filename + "/all_rng_seeds", all(params[k] == seed for k in
                  ("feature_fraction_seed", "bagging_seed", "data_random_seed", "extra_seed")))
        record = records[filename]
        check(filename + "/full_train_model_receipt", record["model_sha256"] == t.core.sha(path) and
              record["train_rows"] == 776706 and record["train_positive"] == 32126 and
              record["train_keys_sha256"] == t.key_digest(data) and
              record["target_sha256"] == t.digest(data.label.to_numpy(dtype=np.int8)))
        del model, actual
    with np.load(output / "full_probe.npz", allow_pickle=False) as probe:
        check("probe_schema_no_labels", set(probe.files) == {"row_id", "O", "B"})
        check("in_sample_full_context_probe_IDs", np.array_equal(probe["row_id"], expected_probe_ids(data)))
        for arm in ("O", "B"):
            check(arm + "/probe_finite_probability", probe[arm].shape == (2048,) and
                  np.isfinite(probe[arm]).all() and (probe[arm] >= 0).all() and (probe[arm] <= 1).all())
    t.verify_artifacts(output, result)
    t.check_seal(seal_path)
    check("source_immutable", t.core.sha(Path(os.environ["P1_DATA_DIR"]) / "train.csv") == cfg["train_sha256"])
    qa = {"status": "PASS", "pid": os.getpid(), "checks": checks, "check_count": len(checks),
          "terminal_result_sha256": t.core.sha(output / "terminal_result.json"),
          "historical_QA_sha256": t.core.sha(historical / "independent-qa.json"),
          "fresh_replay_sha256": t.core.sha(output / "fresh-replay-qa.json"),
          "selector_sha256": t.core.sha(output / "selector.json"), "qa_source_sha256": t.core.sha(__file__),
          "runtime_seconds": time.monotonic() - started, "fits": 0,
          "official_rows": 0, "hidden_rows": 0, "csv_written": 0, "upload": 0,
          "scope": "one full all-train O1+B3 fit path + fresh full-context in-sample probe replay; not two full retrainings",
          "official_materialization": "NOT_RUN_NOT_APPROVED_BY_THIS_RECEIPT",
          "past_28_9_answer_exact_restoration": "NOT_CLAIMED"}
    t.core.write_json(output / "independent-qa.json", qa)
    return qa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    args = parser.parse_args()
    qa = verify(args.output.resolve(), args.historical.resolve(), args.seal.resolve())
    print({"status": qa["status"], "check_count": qa["check_count"], "fits": 0})


if __name__ == "__main__":
    main()
