"""Independent aggregate-only QA for the P3 ERA5 Hs2 official candidate."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p3_wave.submission import validate_submission  # noqa: E402

EXPERIMENT_ID = "p3_champion_matched_era5_hs2_official_deploy_20260828_v1"
CONFIG = ROOT / f"configs/experiments/{EXPERIMENT_ID}.json"
EXPECTED_CHAMPION_SHA = "ea65370a5c9291868769ad9e54a54707035dc93a01ffa4772d9fd26342f357aa"
KEYS = ["case_id", "station", "lead_h"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    config = read_json(CONFIG)
    artifact = ROOT / config["outputs"]["artifact_dir"]
    candidate_dir = Path(config["outputs"]["candidate_dir"])
    result_path = artifact / "result.json"
    manifest_path = artifact / "manifest.json"
    candidate_path = candidate_dir / config["outputs"]["candidate_filename"]
    candidate_manifest_path = candidate_dir / "MANIFEST.json"
    for path in (result_path, manifest_path, candidate_path, candidate_manifest_path):
        if not path.is_file():
            raise RuntimeError(f"QA input is missing: {path}")
    result = read_json(result_path)
    manifest = read_json(manifest_path)
    candidate_manifest = read_json(candidate_manifest_path)
    if result.get("status") != "READY_NOT_UPLOADED" or candidate_manifest.get("uploaded") is not False:
        raise RuntimeError("terminal/upload status drifted")
    if result["operations"] != {
        "scientific_fits": 2,
        "environment_smoke_fits": 0,
        "parameter_searches": 0,
        "serialization_only_repairs": 1,
        "official_truth_rows": 0,
        "anonymous_absolute_time_reconstructions": 0,
        "external_evaluation_period_matches": 0,
        "uploads": 0,
    }:
        raise RuntimeError("operation counters drifted")
    if sha256_file(candidate_path) != result["candidate_sha256"]:
        raise RuntimeError("candidate hash differs from result")
    if manifest["candidate_sha256"] != result["candidate_sha256"]:
        raise RuntimeError("candidate hash differs from artifact manifest")
    if candidate_manifest["submission_sha256"] != result["candidate_sha256"]:
        raise RuntimeError("candidate hash differs from candidate manifest")

    data_value = os.environ.get("P3_DATA_DIR")
    if not data_value:
        raise RuntimeError("P3_DATA_DIR is required")
    data_dir = Path(data_value).resolve()
    test_index = pd.read_csv(data_dir / "test_index.csv")
    candidate = pd.read_csv(candidate_path)
    champion_path = Path(config["immutable_inputs"]["champion_submission"]["path"])
    champion = pd.read_csv(champion_path)
    if sha256_file(champion_path) != EXPECTED_CHAMPION_SHA:
        raise RuntimeError("champion hash drifted")
    validate_submission(candidate, test_index)
    if not champion[KEYS].equals(test_index[KEYS]):
        raise RuntimeError("champion/test index keys drifted")
    active = test_index["lead_h"].isin([18, 24]).to_numpy()
    inactive = ~active
    if int(active.sum()) != 400 or int(inactive.sum()) != 800:
        raise RuntimeError("active/inactive support drifted")
    if not np.array_equal(
        candidate.loc[inactive, "hs_pred"].to_numpy(dtype=np.float64),
        champion.loc[inactive, "hs_pred"].to_numpy(dtype=np.float64),
    ):
        raise RuntimeError("inactive candidate rows differ from champion")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/validate_p3_submission.py"),
            str(candidate_path),
            "--data-dir",
            str(data_dir),
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("independent validator failed during QA")
    validator = json.loads(completed.stdout)
    if validator.get("submission_sha256") != sha256_file(candidate_path):
        raise RuntimeError("independent validator hash drifted during QA")

    receipt = {
        "schema_version": "p3.champion_matched_era5_hs2_official_deploy.qa.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS",
        "checks": {
            "candidate_hash_bound": True,
            "rows_keys_order_range": True,
            "active_rows_18_24": 400,
            "inactive_rows_3_6_9_12": 800,
            "inactive_champion_bit_exact": True,
            "independent_validator_passed": True,
            "official_truth_rows": 0,
            "upload_count": 0,
        },
        "candidate_sha256": sha256_file(candidate_path),
        "official_values_logged_or_reported": False,
    }
    atomic_json(artifact / "independent_qa.json", receipt)
    print(json.dumps({"status": "PASS", "official_values_logged": False}, indent=2))


if __name__ == "__main__":
    main()
