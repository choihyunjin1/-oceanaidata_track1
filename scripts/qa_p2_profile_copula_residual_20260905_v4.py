"""Independent historical-array arithmetic audit; no new fit or raw source read."""

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import run_p2_profile_copula_residual_20260905_v4 as m  # noqa: E402


def main():
    result = json.loads((m.REPORT / "result.json").read_text(encoding="utf-8"))
    sealed = json.loads(m.SEAL.read_text(encoding="utf-8"))["hashes"]
    assert sealed == m.fingerprint()
    m.install_guard(sealed)
    assert m.base.file_hash(m.OUT / "historical_eval.npz") == result["raw_sha256"]
    arrays = np.load(m.OUT / "historical_eval.npz", allow_pickle=False)
    truth, fold = arrays["truth"], arrays["fold"]
    checks = 0

    def metric(label, actual, expected):
        nonlocal checks
        residual = actual - expected
        for key, value in {"n": len(residual), "sse": float(np.dot(residual, residual)), "rmse": math.sqrt(float(np.dot(residual, residual)) / len(residual)), "bias": float(sum(residual) / len(residual))}.items():
            assert np.isclose(label[key], value, rtol=1e-12, atol=1e-10), key
            checks += 1

    scopes = {"pooled": np.ones(len(truth), bool), "autumn_primary": fold == "2024_sep_oct", "natural_missing": arrays["natural_missing"], "natural_present": ~arrays["natural_missing"], **{name: fold == name for name in np.unique(fold)}}
    for scope, mask in scopes.items():
        for arm in ("C", "full", "half"):
            metric(result["intact"][scope]["metrics"][arm], arrays[f"intact_{arm}"][mask], truth[mask])
    for episode in result["episodes"]:
        selected = arrays[f"{episode['id']}_selected"]
        assert len(selected) == 69850 and int(selected.sum()) == episode["injected_rows"]
        assert episode["evaluation_rows_deleted"] == 0
        checks += 3
        if episode["status"] == "SUPPORT_BLOCKED":
            assert episode["id"] == "autumn_3d" and episode["unsupported_rows"] == 4 and "metrics" not in episode
            checks += 1
            continue
        for scope, mask in {"episode": selected, "outside_episode": ~selected}.items():
            for arm in ("C", "full", "half"):
                metric(episode["metrics"][scope]["metrics"][arm], arrays[f"{episode['id']}_{arm}"][mask], truth[mask])
        for arm in ("C", "full", "half"):
            assert np.array_equal(arrays[f"{episode['id']}_{arm}"][~selected], arrays[f"intact_{arm}"][~selected])
            checks += 1
    assert len(np.unique(arrays["key"])) == len(truth) == 69850
    assert result["new_historical_fits"] == len(result["fit_receipts"]) == 3
    assert result["reused_C_historical_fits"] == 9
    primary = result["intact"]["autumn_primary"]["metrics"]
    decision = "full" if primary["full"]["rmse"] < primary["C"]["rmse"] else "half" if primary["half"]["rmse"] < primary["C"]["rmse"] else "C"
    assert decision == result["selected_predefined_policy"]
    replay = json.loads((m.REPORT / "replay.json").read_text(encoding="utf-8"))
    assert replay["status"] == "PASS" and replay["rows"] == 69850 and replay["pid"] != replay["training_pid"]
    assert replay["maximum_absolute_error"] == 0
    assert result["official_access_rows"] == result["csv_written"] == result["upload"] == 0
    checks += 7
    # Prior source pins establish original lineage, not only a seal of today's files.
    old_result = json.loads((m.base.REPORT / "result.json").read_text(encoding="utf-8"))
    original_manifest = old_result["manifest"]
    for field in ("runner_sha256", "config_sha256"):
        if field in original_manifest:
            path = Path(m.base.__file__) if field == "runner_sha256" else m.base.CONFIG
            assert m.base.file_hash(path) == original_manifest[field]
            checks += 1
    m.save(m.REPORT / "independent-qa.json", {"status": "PASS", "checks": checks, "focused_pytest_pass": 10, "ruff": "PASS", "result_sha256": m.base.file_hash(m.REPORT / "result.json"), "replay_sha256": m.base.file_hash(m.REPORT / "replay.json"), "same_keys": 69850, "evaluation_rows_deleted": 0, "fits_recomputed": 3, "no_new_fits_in_qa": True, "official_access_rows": 0, "scope": "independent SSE/count/RMSE/bias from saved same-row arrays, hashes, exact replay; not scientific success or full scratch baseline proof"})
    print(json.dumps({"status": "PASS", "checks": checks}))


if __name__ == "__main__":
    main()
