"""Independent aggregate-only QA for the P2 BayOTIDE-style bounded pilot."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_bayotide_dynamic_factor_20260828_v1"
ARTIFACT = REPO / "artifacts" / EXPERIMENT_ID


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def metric(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def main() -> None:
    manifest_path = ARTIFACT / "manifest.json"
    result_path = ARTIFACT / "result.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    require(manifest["experiment_id"] == EXPERIMENT_ID, "manifest experiment changed")
    require(manifest["outputs"]["result"]["sha256"] == sha256(result_path), "result hash changed")
    for relative, expected in manifest["sources"].items():
        require(sha256(REPO / relative) == expected, f"source hash changed: {relative}")
    frames: list[pd.DataFrame] = []
    for block, output in manifest["outputs"]["predictions"].items():
        path = REPO / output["path"]
        require(sha256(path) == output["sha256"], f"prediction hash changed: {block}")
        with np.load(path, allow_pickle=False) as payload:
            frames.append(
                pd.DataFrame(
                    {
                        "time": pd.to_datetime(payload["time_ns"], utc=True),
                        "layer": payload["layer"].astype(int),
                        "block": block,
                        "reference": payload["reference"].astype(float),
                        "candidate": payload["candidate"].astype(float),
                        "posterior_sd_c": payload["posterior_sd_c"].astype(float),
                        "active": payload["active"].astype(bool),
                    }
                )
            )
    predictions = pd.concat(frames, ignore_index=True)
    require(len(predictions) == 69_850, "prediction rows changed")
    require(not predictions.duplicated(["time", "layer"]).any(), "prediction keys duplicate")
    require(np.isfinite(predictions[["reference", "candidate", "posterior_sd_c"]]).all().all(), "prediction non-finite")
    inactive = ~predictions["active"].to_numpy(bool)
    require(np.array_equal(predictions.loc[inactive, "candidate"].to_numpy(), predictions.loc[inactive, "reference"].to_numpy()), "fallback is not exact")
    comparator_spec = manifest["inputs"]["comparator"]
    comparator_path = REPO / comparator_spec["path"]
    require(sha256(comparator_path) == comparator_spec["sha256"], "comparator hash changed")
    truth = pd.read_parquet(comparator_path, columns=["time", "layer", "block", "truth"])
    truth["time"] = pd.to_datetime(truth["time"], utc=True)
    scored = predictions.merge(truth, on=["time", "layer", "block"], validate="one_to_one")
    incumbent_rmse = metric(scored["truth"].to_numpy(), scored["reference"].to_numpy())
    candidate_rmse = metric(scored["truth"].to_numpy(), scored["candidate"].to_numpy())
    reported = result["metrics"]["aggregate"]
    require(np.isclose(incumbent_rmse, reported["incumbent_rmse_c"], rtol=0, atol=1e-12), "incumbent metric changed")
    require(np.isclose(candidate_rmse, reported["candidate_rmse_c"], rtol=0, atol=1e-12), "candidate metric changed")
    require(not any(ARTIFACT.rglob("*.csv")), "candidate CSV exists")
    checks = {
        "source_and_output_hashes": True,
        "rows_keys_finite": True,
        "exact_incumbent_fallback": True,
        "aggregate_metric_reproduced": True,
        "no_candidate_csv": True,
        "official_upload_performed": False,
    }
    qa = {
        "schema_version": "p2.bayotide_dynamic_factor.independent_qa.20260828.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "verdict": "PASS",
        "checks": checks,
        "manifest_sha256": sha256(manifest_path),
        "result_sha256": sha256(result_path),
        "aggregate": {
            "rows": int(len(scored)),
            "incumbent_rmse_c": incumbent_rmse,
            "candidate_rmse_c": candidate_rmse,
            "delta_rmse_c": candidate_rmse - incumbent_rmse,
        },
    }
    output = ARTIFACT / "independent_qa.json"
    output.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(qa, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
