"""Sealed P3 v69 fixed SAX local-word histogram experiment."""

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

import run_p3_missingness_run_residual_cycle_20260901_v68 as v68  # noqa: E402

EXPERIMENT_ID = "p3_sax_word_histogram_residual_cycle_20260901_v69"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
CHANNELS, CHANNEL_NAMES = (0, 1, 2, 5), ("hs", "tp", "hmax", "wspd")
WINDOWS = ((0, 145), (72, 145))
BREAKPOINTS = np.asarray([-0.8416212335729143, -0.2533471031357997, 0.2533471031357997, 0.8416212335729143])
PAA_BLOCKS, ALPHABET, WORD_LENGTH, WORD_BINS = 12, 5, 3, 125
FEATURE_COUNT = 1000
BASE = v68.BASE
SPEC_CLASS = v68.SPECS[0].__class__
SPECS = (
    SPEC_CLASS("P3_1_SAXWORD1000_RIDGE512_ADD10", 512.0),
    SPEC_CLASS("P3_2_SAXWORD1000_RIDGE2048_ADD10", 2048.0),
)
BLEND, MAD_SCALE, EPSILON = 0.10, 1.4826, 1e-12
sha256, canonical, write_new = v68.sha256, v68.canonical, v68.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v69 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.sax_word_histogram_residual.config.v69",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"] == "NON_DUPLICATE_P3_FIXED_SAX_LOCAL_WORD_AXIS",
        "adjacency": "P1 v18" in config["duplication_audit"]["cross_problem_adjacency"],
        "channels": tuple(encoder["channels"]) == CHANNEL_NAMES,
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "paa": int(encoder["paa_blocks"]) == PAA_BLOCKS,
        "alphabet": int(encoder["alphabet_size"]) == ALPHABET,
        "breakpoints": np.array_equal(np.asarray(encoder["gaussian_breakpoints"]), BREAKPOINTS),
        "word": int(encoder["word_length"]) == WORD_LENGTH,
        "features": int(encoder["feature_count"]) == FEATURE_COUNT,
        "specs": tuple((item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]) == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(float(item["additive_residual_weight"]) == BLEND for item in config["model"]["candidates"]),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "official_excluded": "excluded" in config["duplication_audit"]["official_exclusion"],
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v69 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def robust_normalize(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    if len(path) < PAA_BLOCKS or not np.isfinite(path).all():
        raise ContractError("SAX path support differs")
    center = float(np.median(path))
    scale = MAD_SCALE * float(np.median(np.abs(path - center)))
    if scale <= EPSILON:
        return np.zeros_like(path)
    return (path - center) / scale


def paa_symbols(values: np.ndarray) -> np.ndarray:
    normalized = robust_normalize(values)
    blocks = np.array_split(normalized, PAA_BLOCKS)
    means = np.asarray([np.mean(block) for block in blocks], dtype=np.float64)
    symbols = np.searchsorted(BREAKPOINTS, means, side="right").astype(np.int64)
    if symbols.shape != (PAA_BLOCKS,) or np.any(symbols < 0) or np.any(symbols >= ALPHABET):
        raise ContractError("SAX symbol contract differs")
    return symbols


def word_histogram(symbols: np.ndarray) -> np.ndarray:
    sequence = np.asarray(symbols, dtype=np.int64)
    if sequence.shape != (PAA_BLOCKS,) or np.any(sequence < 0) or np.any(sequence >= ALPHABET):
        raise ContractError("SAX word support differs")
    histogram = np.zeros(WORD_BINS, dtype=np.float64)
    for start in range(PAA_BLOCKS - WORD_LENGTH + 1):
        a, b, c = sequence[start : start + WORD_LENGTH]
        histogram[int(a * ALPHABET**2 + b * ALPHABET + c)] += 1.0
    histogram /= PAA_BLOCKS - WORD_LENGTH + 1
    if not np.isfinite(histogram).all() or not np.isclose(np.sum(histogram), 1.0, atol=1e-15):
        raise ContractError("SAX histogram mass differs")
    return histogram


def transformed_path(sequence: np.ndarray) -> np.ndarray:
    return v68.v67.transformed_path(np.asarray(sequence)[:289])


def sax_features(sequence: np.ndarray) -> np.ndarray:
    path = transformed_path(sequence)[::2]
    if path.shape != (145, 12):
        raise ContractError("fixed 20-minute path differs")
    output: list[np.ndarray] = []
    for channel in CHANNELS:
        for start, stop in WINDOWS:
            output.append(word_histogram(paa_symbols(path[start:stop, channel])))
    features = np.concatenate(output)
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("SAX feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    base = np.sin(np.linspace(0.0, 8.0 * np.pi, 145)) + 0.15 * np.linspace(-1.0, 1.0, 145)
    if not np.array_equal(paa_symbols(base), paa_symbols(7.0 * base + 3.0)):
        raise ContractError("positive affine invariance guard failed")
    constant_hist = word_histogram(paa_symbols(np.full(145, 4.0)))
    middle_word = 2 * ALPHABET**2 + 2 * ALPHABET + 2
    if not constant_hist[middle_word] == 1.0 or np.count_nonzero(constant_hist) != 1:
        raise ContractError("constant path bound guard failed")
    symbols_a = np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1])
    symbols_b = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4])
    if not np.array_equal(np.bincount(symbols_a, minlength=5), np.bincount(symbols_b, minlength=5)):
        raise ContractError("equal-marginal motif guard differs")
    hist_a, hist_b = word_histogram(symbols_a), word_histogram(symbols_b)
    if np.array_equal(hist_a, hist_b):
        raise ContractError("motif-order discrimination guard failed")
    axis = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack([np.sin((index + 1) * axis) + 0.1 * index * axis for index in range(10)])
    sequence[1::7, (0, 3, 6)] = np.nan
    direct = sax_features(sequence)
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    if not np.array_equal(direct, sax_features(extended)):
        raise ContractError("future isolation guard failed")
    return {
        "feature_count": len(direct),
        "feature_sha256": hashlib.sha256(direct.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(direct).all()),
        "positive_affine_invariant": True,
        "constant_middle_word_mass": float(constant_hist[middle_word]),
        "equal_symbol_marginals": True,
        "motif_order_discriminated": True,
        "histogram_mass": float(np.sum(hist_a)),
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
        features[position] = sax_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
    }


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v68.v67.v66.v65.v64.v63.v62.case_surface()
    features, metadata = surface_features(cases)
    nonzero_share = float(np.mean(features > 0.0))
    positive_variance = int(np.sum(np.var(features, axis=0) > 1e-12))
    masses = features.reshape(len(features), len(CHANNELS) * len(WINDOWS), WORD_BINS).sum(axis=2)
    gate = config["encoder"]["support_gate"]
    passed = bool(
        len(features) >= int(gate["minimum_cases"])
        and positive_variance >= int(gate["minimum_positive_variance_features"])
        and nonzero_share >= float(gate["minimum_nonzero_share"])
        and np.allclose(masses, 1.0, rtol=0.0, atol=1e-15)
    )
    return {
        **metadata,
        "nonzero_share": nonzero_share,
        "positive_variance_features": positive_variance,
        "histogram_mass_exact": bool(np.allclose(masses, 1.0, rtol=0.0, atol=1e-15)),
        "target_used": False,
        "passed": passed,
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v69 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {
        "schema_version": "p3.sax_word_histogram_residual.preflight.v69",
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
            "schema_version": "p3.sax_word_histogram_residual.result.v69",
            "experiment_id": EXPERIMENT_ID,
            "decision": "PASS_CANDIDATE_AVAILABLE" if any(item["decision"] != "NO_GO" for item in result["candidates"]) else "NO_GO_ALL_SAX_WORD_CANDIDATES",
            "duplication_audit": config["duplication_audit"],
            "primary_sources": config["primary_sources"],
        }
    )
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 fixed SAX local-word residual cycle v69",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- v69 uses hard Gaussian SAX symbols and local length-3 word composition. It does not reuse prior P3 predictions or features.",
        "- P1-v18 soft-PAA adjacency is disclosed, while the P3 representation and action are independently sealed. The surface is EXPLORATORY_ONLY.",
    ]
    for item in result["candidates"]:
        metric, points = item["rmse_m"], item["expected_points"]
        lines.append(f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; delta {metric['delta_candidate_minus_uniform']:+.9f}m; nominal score {points['nominal_official_score']:.6f}; planning {points['raw_gain']:+.6f}; transport-adjusted {points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; worst block {item['worst_block_delta_m']:+.9f}m; lead {item['worst_lead_delta_m']:+.9f}m; station-lead {item['worst_station_lead_delta_m']:+.9f}m; tail {item['worst_reference_tail_block_delta_m']:+.9f}m; episode CI90 {item['episode_bootstrap']['ci90_m']}; block-station CI90 {item['block_station_bootstrap']['ci90_m']}.")
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
        raise ContractError("v69 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    if preflight["status"] != "READY_EXACTLY_ONCE":
        raise ContractError("v69 support gate failed; zero-fit closure required")
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
    write_new(REPORT / "claim-source-ledger.md", b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| SAX maps normalized PAA values to fixed symbols | Lin et al. 2007, DOI:10.1007/s10618-007-0064-z | representation motivation only |\n| No dedicated executed P3 SAX word-histogram residual axis exists | repository semantic audit | novelty boundary |\n| P1 v18 adjacency and prior/official exclusion | sealed v69 contract | cross-problem and reuse boundary |\n")
    print(json.dumps({"status": "COMPLETE", "decision": result["decision"], "fit_count": 12, "official_access": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
