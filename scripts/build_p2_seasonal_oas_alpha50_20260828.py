"""Build the preregistered P2 OAS alpha=0.50 candidate without uploading it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy
import pandas
import sklearn


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "experiments" / "p2_seasonal_oas_alpha50_deploy_20260828.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_legacy_builder(source: Path, alpha: float, tag: str, ready_dir: Path) -> None:
    environment = os.environ.copy()
    source_root = str(REPO / "src")
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_root if not prior_pythonpath else source_root + os.pathsep + prior_pythonpath
    environment["P2_OAS_ALPHA"] = format(alpha, ".2f")
    environment["P2_OAS_DEPLOY_TAG"] = tag
    environment["P2_OAS_READY_DIR"] = str(ready_dir)
    completed = subprocess.run(
        [sys.executable, str(source)],
        cwd=REPO,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout[-4000:] + completed.stderr[-4000:])


def git_snapshot() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "changed_entry_count": len(status)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config["problem"] == "P2", "config is not P2")
    require(float(config["alpha"]) == 0.5, "alpha is not 0.50")
    require(config["leakage_contract"]["official_upload_authorized"] is False, "upload flag")
    data_dir_value = os.environ.get("P2_DATA_DIR")
    base_value = os.environ.get("P2_OAS_BASE_U")
    require(bool(data_dir_value), "set P2_DATA_DIR")
    require(bool(base_value), "set P2_OAS_BASE_U")
    data_dir = Path(str(data_dir_value)).expanduser().resolve()
    base = Path(str(base_value)).expanduser().resolve()

    pins = config["input_pins"]
    require(sha256(data_dir / "observations.csv") == pins["observations_sha256"], "observations hash")
    require(sha256(data_dir / "test_index.csv") == pins["test_index_sha256"], "test index hash")
    require(sha256(data_dir / "sample_submission.csv") == pins["sample_submission_sha256"], "sample hash")
    require(sha256(base) == pins["base_u_sha256"], "base U hash")

    lineage = config["lineage"]
    source = REPO / lineage["builder"]
    require(sha256(source) == lineage["builder_sha256"], "legacy builder hash changed")
    for relative, expected in lineage["module_sha256"].items():
        require(sha256(REPO / relative) == expected, f"lineage module changed: {relative}")
    for label in ("alpha10", "alpha20", "alpha40"):
        stored = REPO / lineage[label]["stored"]
        require(sha256(stored) == lineage[label]["sha256"], f"stored {label} hash changed")

    geometry = config["metric_geometry"]
    geometry_path = REPO / geometry["artifact"]
    require(sha256(geometry_path) == geometry["artifact_sha256"], "metric geometry artifact hash changed")
    geometry_result = json.loads(geometry_path.read_text(encoding="utf-8"))
    selected = geometry_result["decision"]["selected_next_probe"]
    require(abs(float(selected["alpha"]) - 0.5) <= 1e-12, "metric geometry no longer selects alpha50")
    require(
        float(selected["guaranteed_improvement_vs_alpha40"])
        >= float(config["gates"]["minimum_geometry_improvement_c"]),
        "metric geometry improvement gate failed",
    )

    ready_dir = Path(
        os.environ.get(
            "P2_OAS_READY_DIR",
            str(Path.home() / "Downloads" / "해양 해커톤 제출용" / config["ready_directory_name"]),
        )
    ).expanduser().resolve()
    run_legacy_builder(source, float(config["alpha"]), config["deploy_tag"], ready_dir)

    artifact_dir = REPO / "artifacts" / config["deploy_tag"]
    output = artifact_dir / "P2_submission.csv"
    ready = ready_dir / "P2_submission.csv"
    receipt_path = artifact_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(output.read_bytes() == ready.read_bytes(), "ready copy differs")
    require(receipt["alpha"] == 0.5, "legacy receipt alpha differs")
    require(receipt["inputs"]["base_u"]["sha256"] == pins["base_u_sha256"], "receipt base hash")
    require(receipt["inputs"]["observations"]["sha256"] == pins["observations_sha256"], "receipt observations hash")

    receipt["schema_version"] = "p2.seasonal_oas_alpha50.deployment_receipt.20260828.v1"
    receipt["status"] = "BUILT_PENDING_INDEPENDENT_QA_AND_EXPLICIT_UPLOAD_APPROVAL"
    receipt["deployment_config"] = {
        "path": str(config_path),
        "sha256": sha256(config_path),
        "experiment_id": config["experiment_id"],
    }
    receipt["metric_geometry_gate"] = {
        "artifact": str(geometry_path),
        "sha256": sha256(geometry_path),
        "selected": selected,
        "max_official_probes_before_reassessment": config["gates"]["max_official_probes_before_reassessment"],
    }
    receipt["runtime"] = {
        "completed_at_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "seed": None,
        "deterministic_model": "sklearn.covariance.OAS closed-form fit",
        "git": git_snapshot(),
    }
    receipt["outputs"]["canonical"]["bytes"] = output.stat().st_size
    receipt["outputs"]["ready"]["bytes"] = ready.stat().st_size
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
