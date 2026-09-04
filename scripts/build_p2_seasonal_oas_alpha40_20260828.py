"""Reproduce the pinned OAS lineage and build the P2 alpha=0.40 probe.

The legacy alpha10/20 builder is SHA-pinned and is not modified. This wrapper
first regenerates both historical submissions in isolated ignored directories;
only byte-identical lineage is allowed to proceed to the alpha40 build. It
never reads an answer file and never uploads a submission.
"""

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
DEFAULT_CONFIG = (
    REPO / "configs" / "experiments" / "p2_seasonal_oas_alpha40_deploy_20260828.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run_legacy_builder(*, source: Path, alpha: float, tag: str, ready_dir: Path) -> None:
    environment = os.environ.copy()
    source_root = str(REPO / "src")
    prior_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not prior_pythonpath else source_root + os.pathsep + prior_pythonpath
    )
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
        raise RuntimeError(
            f"legacy OAS builder failed for alpha={alpha}:\n"
            + completed.stdout[-4000:]
            + completed.stderr[-4000:]
        )


def git_snapshot() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return {"commit": commit, "dirty": bool(status), "status_short": status}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(config["problem"] == "P2", "config is not a P2 deployment")
    require(float(config["alpha"]) == 0.4, "this runner is pinned to alpha=0.40")
    require(config["leakage_contract"]["official_upload_authorized"] is False, "upload flag")

    lineage = config["lineage"]
    source = REPO / lineage["builder"]
    require(sha256(source) == lineage["builder_sha256"], "legacy builder hash changed")
    for relative, expected in lineage["module_sha256"].items():
        require(sha256(REPO / relative) == expected, f"lineage module hash changed: {relative}")

    lineage_results: dict[str, object] = {}
    for label, alpha in (("alpha10", 0.10), ("alpha20", 0.20)):
        pin = lineage[label]
        stored = REPO / pin["stored"]
        require(sha256(stored) == pin["sha256"], f"stored {label} hash changed")
        reproduction_dir = REPO / "artifacts" / pin["reproduction_tag"]
        reproduction_ready = REPO / "tmp" / pin["reproduction_tag"]
        run_legacy_builder(
            source=source,
            alpha=alpha,
            tag=pin["reproduction_tag"],
            ready_dir=reproduction_ready,
        )
        reproduction = reproduction_dir / "P2_submission.csv"
        require(stored.read_bytes() == reproduction.read_bytes(), f"{label} byte reproduction failed")
        stored_receipt = json.loads((stored.parent / "receipt.json").read_text(encoding="utf-8"))
        repro_receipt = json.loads(
            (reproduction_dir / "receipt.json").read_text(encoding="utf-8")
        )
        require(
            stored_receipt["fit_receipts"] == repro_receipt["fit_receipts"],
            f"{label} OAS fit receipt changed",
        )
        require(
            stored_receipt["projection"] == repro_receipt["projection"],
            f"{label} PAVA diagnostics changed",
        )
        lineage_results[label] = {
            "stored": str(stored.resolve()),
            "reproduction": str(reproduction.resolve()),
            "sha256": sha256(reproduction),
            "bytes": reproduction.stat().st_size,
            "byte_identical": True,
            "fit_receipts_identical": True,
            "projection_identical": True,
            "projection": repro_receipt["projection"],
        }

    ready_dir = Path(
        os.environ.get(
            "P2_OAS_READY_DIR",
            str(
                Path.home()
                / "Downloads"
                / "해양 해커톤 제출용"
                / config["ready_directory_name"]
            ),
        )
    ).expanduser().resolve()
    run_legacy_builder(
        source=source,
        alpha=float(config["alpha"]),
        tag=config["deploy_tag"],
        ready_dir=ready_dir,
    )

    artifact_dir = REPO / "artifacts" / config["deploy_tag"]
    output = artifact_dir / "P2_submission.csv"
    receipt_path = artifact_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    input_pins = config["input_pins"]
    require(receipt["alpha"] == 0.4, "candidate receipt alpha mismatch")
    require(
        receipt["inputs"]["observations"]["sha256"] == input_pins["observations_sha256"],
        "observations hash mismatch",
    )
    require(
        receipt["inputs"]["test_index"]["sha256"] == input_pins["test_index_sha256"],
        "test index hash mismatch",
    )
    require(
        receipt["inputs"]["sample_submission"]["sha256"]
        == input_pins["sample_submission_sha256"],
        "sample submission hash mismatch",
    )
    require(
        receipt["inputs"]["base_u"]["sha256"] == input_pins["base_u_sha256"],
        "base U hash mismatch",
    )
    require(output.read_bytes() == (ready_dir / "P2_submission.csv").read_bytes(), "ready copy")

    receipt["schema_version"] = "p2.seasonal_oas_alpha40.deployment_receipt.20260828.v1"
    receipt["status"] = "BUILT_PENDING_INDEPENDENT_QA_AND_EXPLICIT_UPLOAD_APPROVAL"
    receipt["deployment_config"] = {
        "path": str(config_path),
        "sha256": sha256(config_path),
        "experiment_id": config["experiment_id"],
    }
    receipt["lineage_reproduction_gate"] = lineage_results
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
    receipt["official_probe_rationale"] = config["official_evidence"]
    receipt["outputs"]["canonical"]["bytes"] = output.stat().st_size
    receipt["outputs"]["ready"]["bytes"] = (ready_dir / "P2_submission.csv").stat().st_size
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
