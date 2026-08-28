"""Fail closed when the fixed P2 bridge would require official hidden targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_boundary_residual_bridge_20260829_v1"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"


class ContractError(RuntimeError):
    """Raised when a preregistered immutable or leakage boundary drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_timestamp(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None:
        raise ContractError("all contract timestamps must be timezone-aware")
    return timestamp


def interval_overlap(
    left_start: datetime,
    left_stop: datetime,
    right_start: datetime,
    right_stop: datetime,
) -> tuple[datetime, datetime] | None:
    start = max(left_start, right_start)
    stop = min(left_stop, right_stop)
    return (start, stop) if start < stop else None


def boundary_windows(
    start: datetime, stop: datetime, *, flank_hours: int
) -> dict[str, tuple[datetime, datetime]]:
    if stop <= start or flank_hours <= 0:
        raise ContractError("invalid block or flank width")
    width = timedelta(hours=flank_hours)
    return {
        "left": (start - width, start),
        "right": (stop, stop + width),
    }


def _validate_record(record: dict[str, Any]) -> None:
    path = ROOT / str(record["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or sha256_file(path) != record["sha256"]
    ):
        raise ContractError(f"immutable contract input changed: {path}")


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment ID drifted")
    _validate_record(config["contract_source"])
    for name in ("config", "result", "commitment"):
        _validate_record(config["frozen_anchor"][name])
    bridge = config["bridge"]
    if (
        int(bridge["flank_hours"]) != 72
        or bridge["interpolation"] != "cubic_smoothstep_3u2_minus_2u3"
        or int(bridge["projector_applications"]) != 1
        or any(
            int(bridge[key]) != 0
            for key in (
                "window_grid_size",
                "taper_grid_size",
                "cap_grid_size",
                "blend_grid_size",
            )
        )
    ):
        raise ContractError("fixed bridge surface drifted")
    policy = config["execution_policy"]
    forbidden = (
        "official_hidden_target_read_allowed",
        "official_test_index_read_allowed",
        "official_sample_submission_read_allowed",
        "submission_csv_generation_allowed",
        "official_upload_authorized",
        "prediction_or_model_artifact_generation_allowed",
        "result_based_retry",
        "contract_reinterpretation_allowed",
        "internal_block_flanks_allowed",
    )
    if any(bool(policy[key]) for key in forbidden) or int(policy["maximum_executions"]) != 1:
        raise ContractError("fail-closed execution policy drifted")
    return config


def current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def audit_contract(config: dict[str, Any]) -> dict[str, Any]:
    commit = current_commit()
    if commit != config["baseline_commit"]:
        raise ContractError(f"baseline commit drifted: {commit}")
    hidden = config["official_hidden_target_interval"]
    hidden_start = parse_timestamp(hidden["start"])
    hidden_stop = parse_timestamp(hidden["stop"])
    flank_hours = int(config["bridge"]["flank_hours"])
    receipts: dict[str, Any] = {}
    collisions: list[dict[str, str]] = []
    for block_name, bounds in config["historical_blocks"].items():
        start = parse_timestamp(bounds["start"])
        stop = parse_timestamp(bounds["stop"])
        block_receipt: dict[str, Any] = {}
        for side, (flank_start, flank_stop) in boundary_windows(
            start, stop, flank_hours=flank_hours
        ).items():
            overlap = interval_overlap(flank_start, flank_stop, hidden_start, hidden_stop)
            record: dict[str, Any] = {
                "start": flank_start.isoformat(),
                "stop": flank_stop.isoformat(),
                "official_hidden_overlap": overlap is not None,
            }
            if overlap is not None:
                record["overlap_start"] = overlap[0].isoformat()
                record["overlap_stop"] = overlap[1].isoformat()
                collisions.append(
                    {
                        "block": block_name,
                        "side": side,
                        "overlap_start": overlap[0].isoformat(),
                        "overlap_stop": overlap[1].isoformat(),
                    }
                )
            block_receipt[side] = record
        receipts[block_name] = block_receipt
    if not collisions:
        raise ContractError("expected fail-closed hidden-target collision disappeared")
    unevaluated_checks = {
        "pooled_delta_rmse": None,
        "2024_sep_oct_delta_rmse": None,
        "minimum_improved_blocks": None,
        "maximum_worst_block_regression": None,
        "maximum_layer_regression": None,
        "day_bootstrap_ci90_upper": None,
        "maximum_absolute_axis_cosine": None,
        "maximum_correction_p99": None,
    }
    return {
        "schema_version": "p2.boundary_residual_bridge.contract_audit.v1",
        "experiment_id": EXPERIMENT_ID,
        "decision": "NO_GO_CONTRACT_LEAKAGE",
        "family_status": "CLOSED_NO_RETRY",
        "terminal_reason": (
            "The fixed outside 72-hour flanks for two historical blocks overlap the "
            "official hidden 2025-09/10 target interval. The contract cannot be evaluated "
            "without forbidden target access."
        ),
        "baseline_commit": commit,
        "execution_count": 1,
        "bridge_fit_count": 0,
        "prediction_rows_generated": 0,
        "prediction_files_generated": 0,
        "model_files_generated": 0,
        "csv_files_generated": 0,
        "official_hidden_target_rows_read": 0,
        "official_test_index_rows_read": 0,
        "official_sample_submission_rows_read": 0,
        "official_submission_rows_read": 0,
        "source_observation_rows_read": 0,
        "data_paths_opened": [],
        "flank_contract": receipts,
        "hidden_target_collisions": collisions,
        "required_flanks": 2 * len(config["historical_blocks"]),
        "blocked_flanks": len(collisions),
        "metric_gate_evaluated": False,
        "gate_thresholds": config["gate"],
        "gate_checks": unevaluated_checks,
        "contract_reinterpreted": False,
        "internal_first_or_last_72h_used": False,
        "smoothstep_applied": False,
        "projector_applied": False,
        "submission_generated_or_uploaded": False,
        "runtime": {"python": platform.python_version()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute:
        raise SystemExit("--execute is required; the one-shot result is write-once")
    config = load_config()
    result = audit_contract(config)
    output = ROOT / config["output"]["directory"] / config["output"]["result"]
    atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
