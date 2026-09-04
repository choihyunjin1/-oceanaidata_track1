"""Sealed P3 v67 Pettitt rank-change geometry experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / "scripts"))

import run_p3_record_process_residual_cycle_20260901_v66 as v66  # noqa: E402

EXPERIMENT_ID = "p3_pettitt_rank_change_residual_cycle_20260901_v67"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS, CHANNEL_NAMES = (0, 1, 2, 5), ("hs", "tp", "hmax", "wspd")
WINDOWS = ((0, 145), (72, 145))
FEATURE_COUNT = 64
BASE = v66.BASE
SPEC_CLASS = v66.SPECS[0].__class__
SPECS = (
    SPEC_CLASS("P3_1_PETTITT64_RIDGE512_ADD10", 512.0),
    SPEC_CLASS("P3_2_PETTITT64_RIDGE2048_ADD10", 2048.0),
)
BLEND, EPSILON = 0.10, 1e-12
sha256, canonical, write_new = v66.sha256, v66.canonical, v66.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v67 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.pettitt_rank_change_residual.config.v67",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_P3_PETTITT_RANK_CHANGE_AXIS",
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "features": encoder["feature_count"] == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "official_excluded": "excluded" in config["duplication_audit"]["official_exclusion"],
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v67 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def average_ranks(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    order = np.argsort(path, kind="mergesort")
    ranks = np.empty(len(path), dtype=np.float64)
    start = 0
    while start < len(path):
        stop = start + 1
        while stop < len(path) and path[order[stop]] == path[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def pettitt_statistics(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    n = len(path)
    if n < 4 or not np.isfinite(path).all():
        raise ContractError("Pettitt path support differs")
    ranks = average_ranks(path)
    split_count = np.arange(1, n, dtype=np.float64)
    u_values = 2.0 * np.cumsum(ranks)[:-1] - split_count * (n + 1.0)
    argmax = int(np.argmax(np.abs(u_values)))
    split = argmax + 1
    q25, q75 = np.quantile(path, (0.25, 0.75))
    scale = max(float(q75 - q25), EPSILON)
    pre, post = path[:split], path[split:]
    pre_iqr = float(np.subtract(*np.quantile(pre, (0.75, 0.25))))
    post_iqr = float(np.subtract(*np.quantile(post, (0.75, 0.25))))
    result = np.asarray(
        [
            -float(u_values[argmax]) / n**1.5,
            split / n,
            (float(np.median(post)) - float(np.median(pre))) / scale,
            (post_iqr - pre_iqr) / scale,
        ],
        dtype=np.float64,
    )
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ContractError("Pettitt statistic contract differs")
    return result


def robust_level_increment_statistics(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    if len(path) < 65 or not np.isfinite(path).all():
        raise ContractError("Pettitt path below sealed support")
    median = float(np.median(path))
    q25, q75 = np.quantile(path, (0.25, 0.75))
    normalized = (path - median) / max(float(q75 - q25), EPSILON)
    return np.concatenate([pettitt_statistics(normalized), pettitt_statistics(np.diff(normalized))])


def transformed_path(sequence: np.ndarray) -> np.ndarray:
    return v66.transformed_path(sequence)


def pettitt_features(sequence: np.ndarray) -> np.ndarray:
    path = transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    output: list[float] = []
    for channel in CHANNELS:
        for start, stop in WINDOWS:
            output.extend(robust_level_increment_statistics(path[start:stop, channel]))
    features = np.asarray(output, dtype=np.float64)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("Pettitt feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    rng = np.random.default_rng(20260901)
    iid = rng.normal(size=256)
    upward = iid.copy()
    upward[128:] += 4.0
    downward = -upward
    iid_stats = pettitt_statistics(iid)
    upward_stats = pettitt_statistics(upward)
    downward_stats = pettitt_statistics(downward)
    if not 0.45 <= upward_stats[1] <= 0.55 or not upward_stats[0] > 0.0 or not upward_stats[2] > 0.8:
        raise ContractError("step-change localization/sign guard failed")
    if not abs(upward_stats[0]) > abs(iid_stats[0]) + 0.20:
        raise ContractError("iid-versus-shift Pettitt strength guard failed")
    positive = pettitt_statistics(7.0 * upward + 3.0)
    if not np.allclose(upward_stats, positive, rtol=1e-12, atol=1e-12):
        raise ContractError("positive affine invariance guard failed")
    if not np.allclose(downward_stats, [-upward_stats[0], upward_stats[1], -upward_stats[2], upward_stats[3]], rtol=1e-12, atol=1e-12):
        raise ContractError("negative affine sign-exchange guard failed")
    axis = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * axis) + 0.1 * index * axis for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    feature = pettitt_features(sequence)
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    if not np.array_equal(feature, pettitt_features(extended[:289])):
        raise ContractError("future isolation guard failed")
    return {
        "feature_count": len(feature),
        "feature_sha256": hashlib.sha256(feature.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(feature).all()),
        "iid_abs_normalized_pettitt": float(abs(iid_stats[0])),
        "step_abs_normalized_pettitt": float(abs(upward_stats[0])),
        "step_split_fraction": float(upward_stats[1]),
        "step_signed_median_shift": float(upward_stats[2]),
        "positive_affine_invariant": True,
        "negative_affine_sign_exchange": True,
        "future_isolated": True,
    }


def surface_features(cases: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    sequences = np.load(BASE.SEQUENCES, mmap_mode="r")
    station_codes = np.load(BASE.STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        if int(station_codes[anchor_id]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        features[position] = pettitt_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v66.v65.v64.v63.v62.case_surface()
    features, metadata = surface_features(cases)
    nonzero_share = float(np.mean(np.abs(features) > 1e-12))
    positive_variance = int(np.sum(np.var(features, axis=0) > 1e-12))
    gate = config["encoder"]["support_gate"]
    minimum_count = min(stop - start - 1 for start, stop in WINDOWS)
    passed = bool(
        minimum_count >= int(gate["minimum_sample_count"])
        and nonzero_share >= float(gate["minimum_nonzero_share"])
        and positive_variance >= int(gate["minimum_positive_variance_features"])
    )
    return {
        **metadata,
        "minimum_sample_count": minimum_count,
        "nonzero_share": nonzero_share,
        "positive_variance_features": positive_variance,
        "target_used": False,
        "passed": passed,
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v67 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {
        "schema_version": "p3.pettitt_rank_change_residual.preflight.v67",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXACTLY_ONCE" if support["passed"] else "STOP_SUPPORT_GATE",
        "config_sha256": sha256(CONFIG),
        "runner_sha256": sha256(Path(__file__)),
        "candidate_count": 2,
        "maximum_model_fits": 12 if support["passed"] else 0,
        "synthetic": synthetic_receipt(),
        "historical_support": support,
        "prior_outputs_used": False,
        "official_used_for_features_gates_selection": False,
        "official_access": 0,
        "csv_materializations": 0,
        "uploads": 0,
        "config_status": config["status"],
    }
    payload["receipt_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def execute(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    original_surface, original_specs = BASE.surface_features, BASE.SPECS
    BASE.surface_features, BASE.SPECS = surface_features, SPECS
    try:
        result, arrays = BASE.execute(config)
    finally:
        BASE.surface_features, BASE.SPECS = original_surface, original_specs
    result.update(
        {
            "schema_version": "p3.pettitt_rank_change_residual.result.v67",
            "experiment_id": EXPERIMENT_ID,
            "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_PETTITT_CANDIDATES",
            "duplication_audit": config["duplication_audit"],
            "primary_sources": config["primary_sources"],
        }
    )
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 Pettitt rank-change residual cycle v67",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- v67 scans all internal splits with a rank cumulative statistic; it is not BOCPD, fixed early/late OT, fixed-level crossing, or adaptive record timing.",
        "- Prior outputs and official feedback are excluded; the 182-case surface is EXPLORATORY_ONLY.",
    ]
    for item in result["candidates"]:
        metric, points = item["rmse_m"], item["expected_points"]
        lines.append(f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; delta {metric['delta_candidate_minus_uniform']:+.9f}m; nominal score {points['nominal_official_score']:.6f}; raw/planning {points['raw_gain']:+.6f} points; transport-adjusted {points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; worst block {item['worst_block_delta_m']:+.9f}m; lead {item['worst_lead_delta_m']:+.9f}m; station-lead {item['worst_station_lead_delta_m']:+.9f}m; tail {item['worst_reference_tail_block_delta_m']:+.9f}m; episode CI90 {item['episode_bootstrap']['ci90_m']}; block-station CI90 {item['block_station_bootstrap']['ci90_m']}.")
    lines.append("Official test/sample/submission/hidden access, CSV materialization, and upload were all zero.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(canonical(preflight_payload()).decode(), end="")
        return 0
    if ARTIFACT.exists() or REPORT.exists() or LOCK.exists():
        raise ContractError("v67 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    if preflight["status"] != "READY_EXACTLY_ONCE":
        raise ContractError("v67 support gate failed; zero-fit closure required")
    write_new(LOCK, canonical({"experiment_id": EXPERIMENT_ID, "status": "ATTEMPT_CONSUMED_ONE_SHOT", "runner_sha256": sha256(Path(__file__)), "config_sha256": sha256(CONFIG), "preflight_receipt_sha256": preflight["receipt_sha256"], "official_access": 0}))
    ARTIFACT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=False)
    result, arrays = execute(config)
    array_path = ARTIFACT / "evaluation-arrays.npz"
    np.savez_compressed(array_path, **arrays)
    result["provenance"] = {"runner_sha256": sha256(Path(__file__)), "config_sha256": sha256(CONFIG), "evaluation_arrays_sha256": sha256(array_path), "preflight_receipt_sha256": preflight["receipt_sha256"], "input_sha256": config["inputs"]}
    result_path = ARTIFACT / "result.json"
    write_new(result_path, canonical(result))
    write_new(REPORT / "result.json", canonical(result))
    report_path = REPORT / "report-source.md"
    write_new(report_path, render_report(result).encode())
    write_new(REPORT / "run-manifest.json", canonical({"experiment_id": EXPERIMENT_ID, "result_sha256": sha256(result_path), "arrays_sha256": sha256(array_path), "report_sha256": sha256(report_path), "fit_count": 12, "official_access": 0, "csv_materializations": 0, "uploads": 0}))
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Pettitt's statistic uses rank-based nonparametric evidence for a single change point | Pettitt 1979, DOI:10.2307/2346729 | mechanism only |\n| BOCPD, fixed-split OT, crossings and record timing have different fingerprints | repository semantic audit | novelty boundary |\n| Prior outputs and official feedback were excluded | sealed v67 contract | reuse boundary |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
