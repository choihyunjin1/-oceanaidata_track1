"""Independent saved-OOF arithmetic and input-hash QA for P1 A/B/C."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
RUN = "p1_depth_contract_repair_20260905_v2"
REPORT = ROOT / "reports" / RUN
ARTIFACT = ROOT / "artifacts" / RUN
KEYS = ["station", "year", "layer", "time"]


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run():
    results = {
        key: json.loads((REPORT / name).read_text(encoding="utf-8"))
        for key, name in {
            "a": "result.json",
            "a_qa": "independent-qa.json",
            "b": "provenance-audit.json",
            "c": "decoder-result.json",
        }.items()
    }
    a, a_qa, b, c = (results[k] for k in ("a", "a_qa", "b", "c"))
    checks = {
        "A_exact_16fits": (a["screen_fits"], a["final_inner_fits"], a["full_fits"], len(a["fits"]))
        == (12, 2, 2, 16),
        "A_reused_control_12": a["reused_control_models"] == 12,
        "A_fresh_process_replay": a_qa["status"] == "PASS"
        and all(x["max_abs_diff"] == 0 for x in a_qa["replay"]),
        "A_result_hash": a_qa["result_sha256"] == sha(REPORT / "result.json"),
        "A_code_config_hash": a["runner_sha256"] == sha(ROOT / "scripts" / ("run_" + RUN + ".py"))
        and a["config_sha256"] == sha(ROOT / "configs/experiments" / (RUN + ".json")),
        "A_all_model_hashes": all(sha(ROOT / x["path"]) == x["sha256"] for x in a["fits"]),
        "B_not_combined_wrong_split": b["same_global_key_set"]
        and b["changed_fold_rows"] == 119
        and not b["zero_fit_combination_executed"]
        and not b["gpu_fit_started"],
        "C_A_hash": c["source_a_result_sha256"] == sha(REPORT / "result.json"),
        "C_B_hash": c["source_b_result_sha256"] == sha(REPORT / "provenance-audit.json"),
        "C_code_hash": c["runner_sha256"]
        == sha(ROOT / "scripts/run_p1_depth_contract_postaudit_20260905_v2.py"),
        "C_decoder_hash": c["decoder_source_sha256"]
        == sha(ROOT / "scripts/run_p1_score_repair_decoder_20260905_v1.py"),
        "C_only_fixed_policy": c["new_backbone_fits"] == 0
        and c["transition_estimates"] == 4
        and (c["lambda"], c["laplace"], c["probability_clip"]) == (1.0, 1.0, 1e-6),
        "C_oof_hash": c["oof_sha256"] == sha(ARTIFACT / "p1_c_fixed_decoder/oof.parquet"),
        "C_full_transition_hash": c["full_transition_sha256"]
        == sha(ARTIFACT / "p1_c_fixed_decoder/full_transition.json"),
        "official_hidden_csv_upload_zero": all(
            x[k] == 0
            for x in (a, a_qa, b, c)
            for k in ("official_rows", "hidden_rows", "csv_written", "upload")
        ),
    }
    pooled = pd.read_parquet(ARTIFACT / "p1_c_fixed_decoder/oof.parquet")
    base = pd.read_parquet(ARTIFACT / "03_training/oof.parquet")
    checks["same_exact_421032_keys"] = (
        len(pooled) == 421032
        and not pooled.duplicated(KEYS).any()
        and pooled[KEYS].equals(base[KEYS])
    )
    checks["same_truth_and_OFF"] = np.array_equal(pooled.label, base.label) and np.array_equal(
        pooled.candidate, base.candidate
    )
    independent = {}
    for name in ("control", "candidate", "candidate_decoder_on"):
        matrix = confusion_matrix(pooled.label, pooled[name], labels=[0, 1])
        _tn, fp, fn, tp = matrix.ravel()
        score = float(f1_score(pooled.label, pooled[name]))
        independent[name] = {"f1": score, "tp": int(tp), "fp": int(fp), "fn": int(fn)}
        checks[name + "_metrics"] = all(
            independent[name][k] == c["pooled"][name][k] for k in independent[name]
        )
        y, p, r = (
            pooled.label.to_numpy(bool),
            pooled[name].to_numpy(bool),
            pooled.control.to_numpy(bool),
        )
        changes = {
            "added_tp": int((y & p & ~r).sum()),
            "added_fp": int((~y & p & ~r).sum()),
            "removed_tp": int((y & ~p & r).sum()),
            "removed_fp": int((~y & ~p & r).sum()),
        }
        checks[name + "_change_counts"] = all(changes[k] == c["pooled"][name][k] for k in changes)
    chosen = max(independent, key=lambda k: independent[k]["f1"])
    checks["selection_rule_recalculated"] = chosen == c["chosen_development_policy"]
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "check_count": len(checks),
        "passed": int(sum(checks.values())),
        "independent_metrics": independent,
        "chosen_development_policy": chosen,
        "official_rows": 0,
        "hidden_rows": 0,
        "csv_written": 0,
        "upload": 0,
        "qa_scope": "independent sklearn/confusion and change-count arithmetic, exact keys, artifact hashes; A verifier separately reruns saved models in a fresh process",
        "not_claimed": "repeated training determinism, fresh holdout significance or official score uplift",
        "source_hashes": {
            name: sha(REPORT / name)
            for name in (
                "result.json",
                "independent-qa.json",
                "provenance-audit.json",
                "decoder-result.json",
            )
        },
    }
    path = REPORT / "cycle-independent-qa.json"
    if path.exists():
        raise FileExistsError("cycle QA receipt already exists")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(run())
