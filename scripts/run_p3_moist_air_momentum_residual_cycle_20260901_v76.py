"""Sealed P3 v76 moist-air momentum-flux representation."""

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

import run_p3_matrix_pencil_mode_residual_cycle_20260901_v75 as v75  # noqa: E402

EXPERIMENT_ID = "p3_moist_air_momentum_residual_cycle_20260901_v76"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
WINDOWS = ((0, 145), (72, 145))
FEATURE_COUNT = 48
BASE = v75.BASE
SPEC_CLASS = v75.SPECS[0].__class__
SPECS = (
    SPEC_CLASS("P3_1_MOISTAIR48_RIDGE512_ADD10", 512.0),
    SPEC_CLASS("P3_2_MOISTAIR48_RIDGE2048_ADD10", 2048.0),
)
BLEND, R_DRY, R_VAPOR, EPSILON = 0.10, 287.05, 461.495, 1e-12
sha256, canonical, write_new = v75.sha256, v75.canonical, v75.write_new


class ContractError(RuntimeError):
    """Raised when the sealed v76 contract differs."""


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    encoder = config["encoder"]
    checks = {
        "schema": config["schema_version"] == "p3.moist_air_momentum_residual.config.v76",
        "experiment": config["experiment_id"] == EXPERIMENT_ID,
        "novel": config["duplication_audit"]["semantic_verdict"]
        == "NON_DUPLICATE_P3_MOIST_AIR_MOMENTUM_AXIS",
        "paths": tuple(encoder["derived_paths"])
        == (
            "moist_air_density",
            "rho_times_wspd_squared",
            "rho_times_gust_squared",
            "rho_times_nonnegative_gust_squared_excess",
        ),
        "windows": tuple(tuple(item) for item in encoder["windows"].values()) == WINDOWS,
        "features": int(encoder["feature_count"]) == FEATURE_COUNT,
        "specs": tuple(
            (item["name"], float(item["ridge_alpha"])) for item in config["model"]["candidates"]
        )
        == tuple((item.name, item.alpha) for item in SPECS),
        "blend": all(
            float(item["additive_residual_weight"]) == BLEND
            for item in config["model"]["candidates"]
        ),
        "fits": config["validation"]["maximum_total_fits"] == 12,
        "official_zero": all(value == 0 for value in config["official_policy"].values()),
        "official_excluded": "excluded" in config["duplication_audit"]["official_exclusion"],
        "no_posthoc": not config["duplication_audit"]["posthoc_prior_cycle_adjustment"],
    }
    if not all(checks.values()):
        raise ContractError(f"v76 config contract failed: {checks}")
    for relative, expected in config["inputs"].items():
        path = ROOT / relative
        if not path.exists() or sha256(path) != expected:
            raise ContractError(f"input pin differs: {relative}")
    return config


def fill_prefix(values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.float64)
    if raw.shape != (289, 10):
        raise ContractError("raw context shape differs")
    result = np.empty_like(raw)
    index = np.arange(len(raw), dtype=np.float64)
    for column in range(raw.shape[1]):
        finite = np.isfinite(raw[:, column])
        result[:, column] = (
            np.interp(index, index[finite], raw[finite, column]) if finite.any() else 0.0
        )
    return result


def moist_air_density(
    temperature_c: np.ndarray, relative_humidity_pct: np.ndarray, pressure_hpa: np.ndarray
) -> np.ndarray:
    temperature = np.asarray(temperature_c, dtype=np.float64)
    humidity = np.clip(np.asarray(relative_humidity_pct, dtype=np.float64), 0.0, 100.0)
    pressure = np.maximum(np.asarray(pressure_hpa, dtype=np.float64), 1.0)
    saturation_hpa = 6.112 * np.exp(17.67 * temperature / (temperature + 243.5))
    vapor_hpa = np.minimum(humidity / 100.0 * saturation_hpa, 0.99 * pressure)
    kelvin = temperature + 273.15
    density = ((pressure - vapor_hpa) * 100.0) / (R_DRY * kelvin) + (vapor_hpa * 100.0) / (
        R_VAPOR * kelvin
    )
    if not np.isfinite(density).all() or np.any(density <= 0.0):
        raise ContractError("moist-air density is invalid")
    return density


def physical_paths(sequence: np.ndarray) -> np.ndarray:
    raw = fill_prefix(np.asarray(sequence)[:289])[::2]
    wind = np.maximum(raw[:, 4], 0.0)
    gust = np.maximum(raw[:, 5], 0.0)
    density = moist_air_density(raw[:, 7], raw[:, 8], raw[:, 9])
    paths = np.column_stack(
        [
            density,
            density * wind**2,
            density * gust**2,
            density * np.maximum(gust**2 - wind**2, 0.0),
        ]
    )
    if paths.shape != (145, 4) or not np.isfinite(paths).all():
        raise ContractError("physical momentum paths differ")
    return paths


def path_statistics(values: np.ndarray) -> np.ndarray:
    path = np.asarray(values, dtype=np.float64)
    q25, q75 = np.quantile(path, [0.25, 0.75])
    time = np.arange(len(path), dtype=np.float64)
    centered_time = time - float(np.mean(time))
    slope = float(
        np.dot(centered_time, path - float(np.mean(path))) / np.dot(centered_time, centered_time)
    )
    features = np.asarray(
        [
            np.median(path),
            q75 - q25,
            np.quantile(path, 0.90),
            path[-1] - path[0],
            slope,
            np.mean(np.diff(path) > 0.0),
        ],
        dtype=np.float64,
    )
    if features.shape != (6,) or not np.isfinite(features).all():
        raise ContractError("physical path statistics differ")
    return features


def momentum_features(sequence: np.ndarray) -> np.ndarray:
    paths = physical_paths(sequence)
    features = np.concatenate(
        [
            path_statistics(paths[start:stop, column])
            for column in range(4)
            for start, stop in WINDOWS
        ]
    )
    if features.shape != (FEATURE_COUNT,) or not np.isfinite(features).all():
        raise ContractError("moist-air momentum feature contract differs")
    return features


def synthetic_receipt() -> dict[str, Any]:
    standard = float(
        moist_air_density(np.asarray([15.0]), np.asarray([50.0]), np.asarray([1013.25]))[0]
    )
    warm = float(
        moist_air_density(np.asarray([30.0]), np.asarray([50.0]), np.asarray([1013.25]))[0]
    )
    humid = float(
        moist_air_density(np.asarray([15.0]), np.asarray([90.0]), np.asarray([1013.25]))[0]
    )
    low_pressure = float(
        moist_air_density(np.asarray([15.0]), np.asarray([50.0]), np.asarray([990.0]))[0]
    )
    if (
        not 1.15 < standard < 1.30
        or not warm < standard
        or not humid < standard
        or not low_pressure < standard
    ):
        raise ContractError("moist-air thermodynamic direction guard failed")
    if not np.isclose((standard * 10.0**2) / (standard * 5.0**2), 4.0, rtol=0.0, atol=1e-12):
        raise ContractError("quadratic momentum guard failed")
    axis = np.linspace(-1.0, 1.0, 289)
    sequence = np.column_stack(
        [np.sin((column + 1) * axis) + 0.1 * column * axis for column in range(10)]
    )
    sequence[:, 4] = 8.0 + np.sin(axis)
    sequence[:, 5] = 10.0 + np.sin(axis)
    sequence[:, 7] = 15.0 + 5.0 * axis
    sequence[:, 8] = 65.0 + 10.0 * axis
    sequence[:, 9] = 1013.0 - 4.0 * axis
    sequence[1::7, (0, 3, 6)] = np.nan
    direct = momentum_features(sequence)
    extended = np.vstack([sequence, np.full((12, 10), 1e9)])
    if not np.array_equal(direct, momentum_features(extended)):
        raise ContractError("future isolation guard failed")
    return {
        "feature_count": len(direct),
        "feature_sha256": hashlib.sha256(direct.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(direct).all()),
        "standard_density_kg_m3": standard,
        "warm_density_kg_m3": warm,
        "humid_density_kg_m3": humid,
        "low_pressure_density_kg_m3": low_pressure,
        "quadratic_wind_ratio": 4.0,
        "future_isolated": True,
    }


def surface_features(cases: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    sequences = np.load(BASE.SEQUENCES, mmap_mode="r")
    station_codes = np.load(BASE.STATIONS, mmap_mode="r")
    station_map = {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}
    features = np.empty((len(cases), FEATURE_COUNT), dtype=np.float64)
    density_min, density_max = np.inf, -np.inf
    for position, row in enumerate(cases.itertuples(index=False)):
        anchor_id = int(row.anchor_id)
        if int(station_codes[anchor_id]) != station_map[str(row.station)]:
            raise ContractError("sequence station key differs")
        paths = physical_paths(sequences[anchor_id])
        density_min = min(density_min, float(np.min(paths[:, 0])))
        density_max = max(density_max, float(np.max(paths[:, 0])))
        features[position] = momentum_features(sequences[anchor_id])
    return features, {
        "rows": len(features),
        "columns": features.shape[1],
        "matrix_sha256": hashlib.sha256(features.astype("<f8").tobytes()).hexdigest(),
        "finite": bool(np.isfinite(features).all()),
        "density_min_kg_m3": density_min,
        "density_max_kg_m3": density_max,
    }


def support_receipt(config: dict[str, Any]) -> dict[str, Any]:
    cases, _, _, _ = v75.v74.v73.v72.v71.v70.v69.v68.v67.v66.v65.v64.v63.v62.case_surface()
    features, metadata = surface_features(cases)
    positive_variance = int(np.sum(np.var(features, axis=0) > 1e-12))
    gate = config["encoder"]["support_gate"]
    passed = bool(
        len(features) >= int(gate["minimum_cases"])
        and positive_variance >= int(gate["minimum_positive_variance_features"])
        and metadata["density_min_kg_m3"] >= float(gate["minimum_density_kg_m3"])
        and metadata["density_max_kg_m3"] <= float(gate["maximum_density_kg_m3"])
    )
    return {
        **metadata,
        "positive_variance_features": positive_variance,
        "target_used": False,
        "passed": passed,
    }


def preflight_payload() -> dict[str, Any]:
    config = load_config()
    if ARTIFACT.exists() or LOCK.exists():
        raise ContractError("v76 exactly-once namespace is consumed")
    support = support_receipt(config)
    payload = {
        "schema_version": "p3.moist_air_momentum_residual.preflight.v76",
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
            "schema_version": "p3.moist_air_momentum_residual.result.v76",
            "experiment_id": EXPERIMENT_ID,
            "decision": "PASS_CANDIDATE_AVAILABLE"
            if any(item["decision"] != "NO_GO" for item in result["candidates"])
            else "NO_GO_ALL_MOIST_AIR_MOMENTUM_CANDIDATES",
            "duplication_audit": config["duplication_audit"],
            "primary_sources": config["primary_sources"],
        }
    )
    return result, arrays


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# P3 moist-air momentum residual cycle v76",
        "",
        "## 결론",
        "",
        f"- overall decision: **{result['decision']}**.",
        "- v76 combines local pressure, temperature and humidity into moist-air density before forming sustained/gust momentum proxies; it reuses neither v20 predictions nor official feedback.",
        "- The repeatedly exposed 182-case surface is EXPLORATORY_ONLY.",
    ]
    for item in result["candidates"]:
        metric, points = item["rmse_m"], item["expected_points"]
        lines.append(
            f"- {item['name']}: {item['decision']}; RMSE {metric['candidate']:.9f}m; delta {metric['delta_candidate_minus_uniform']:+.9f}m; nominal score {points['nominal_official_score']:.6f}; planning {points['raw_gain']:+.6f}; transport-adjusted {points['transport_adjusted_gain']:+.6f}; blocks {item['improved_blocks']}/6; worst block {item['worst_block_delta_m']:+.9f}m; lead {item['worst_lead_delta_m']:+.9f}m; station-lead {item['worst_station_lead_delta_m']:+.9f}m; tail {item['worst_reference_tail_block_delta_m']:+.9f}m; episode CI90 {item['episode_bootstrap']['ci90_m']}; block-station CI90 {item['block_station_bootstrap']['ci90_m']}."
        )
    lines.append(
        "Official test/sample/submission/hidden access, CSV materialization, and upload were all zero."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(canonical(preflight_payload()).decode(), end="")
        return 0
    if ARTIFACT.exists() or REPORT.exists() or LOCK.exists():
        raise ContractError("v76 exactly-once namespace already exists")
    config, preflight = load_config(), preflight_payload()
    write_new(
        LOCK,
        canonical(
            {
                "experiment_id": EXPERIMENT_ID,
                "status": "ATTEMPT_CONSUMED_ONE_SHOT",
                "runner_sha256": sha256(Path(__file__)),
                "config_sha256": sha256(CONFIG),
                "preflight_receipt_sha256": preflight["receipt_sha256"],
                "official_access": 0,
            }
        ),
    )
    ARTIFACT.mkdir(parents=True, exist_ok=False)
    REPORT.mkdir(parents=True, exist_ok=False)
    if preflight["status"] == "STOP_SUPPORT_GATE":
        result = {
            "schema_version": "p3.moist_air_momentum_residual.result.v76",
            "experiment_id": EXPERIMENT_ID,
            "status": "COMPLETE",
            "decision": "STOP_SUPPORT_GATE_ZERO_FIT",
            "fit_count": 0,
            "support_receipt": preflight["historical_support"],
            "duplication_audit": config["duplication_audit"],
            "primary_sources": config["primary_sources"],
            "data_access": {
                "historical_target_rows": 0,
                "official_test_rows": 0,
                "official_sample_rows": 0,
                "official_submission_rows": 0,
                "hidden_truth_rows": 0,
                "csv_materializations": 0,
                "uploads": 0,
            },
            "provenance": {
                "runner_sha256": sha256(Path(__file__)),
                "config_sha256": sha256(CONFIG),
                "preflight_receipt_sha256": preflight["receipt_sha256"],
                "input_sha256": config["inputs"],
            },
        }
        result_path = ARTIFACT / "result.json"
        write_new(result_path, canonical(result))
        write_new(REPORT / "result.json", canonical(result))
        report_path = REPORT / "report-source.md"
        write_new(
            report_path,
            b"# P3 moist-air momentum residual cycle v76\n\n"
            b"## Conclusion\n\n"
            b"- **STOP_SUPPORT_GATE_ZERO_FIT**.\n"
            b"- The sealed physical-density support range failed because pressure is unavailable "
            b"for part of the historical surface; no target, outer score, official input, CSV, "
            b"or upload was used. The density bound and feature contract are not relaxed.\n",
        )
        write_new(
            REPORT / "run-manifest.json",
            canonical(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "result_sha256": sha256(result_path),
                    "report_sha256": sha256(report_path),
                    "fit_count": 0,
                    "official_access": 0,
                    "csv_materializations": 0,
                    "uploads": 0,
                }
            ),
        )
        write_new(
            REPORT / "claim-source-ledger.md",
            b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n"
            b"| Moist-air density depends on pressure, temperature and water-vapor content | Picard et al. 2008, DOI:10.1088/0026-1394/45/2/004 | physical motivation only |\n"
            b"| Ocean momentum flux depends on air density and squared wind speed | Large and Pond 1981, DOI:10.1175/1520-0485(1981)011<0324:OOMFMI>2.0.CO;2 | physical motivation only |\n"
            b"| Sealed target-free physical support failed before scoring | v76 preflight receipt | zero-fit decision |\n"
            b"| Prior/official outputs were excluded | sealed v76 contract | reuse boundary |\n",
        )
        print(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "decision": result["decision"],
                    "fit_count": 0,
                    "official_access": 0,
                },
                ensure_ascii=False,
            )
        )
        return 0
    result, arrays = execute(config)
    array_path = ARTIFACT / "evaluation-arrays.npz"
    np.savez_compressed(array_path, **arrays)
    result["provenance"] = {
        "runner_sha256": sha256(Path(__file__)),
        "config_sha256": sha256(CONFIG),
        "evaluation_arrays_sha256": sha256(array_path),
        "preflight_receipt_sha256": preflight["receipt_sha256"],
        "input_sha256": config["inputs"],
    }
    result_path = ARTIFACT / "result.json"
    write_new(result_path, canonical(result))
    write_new(REPORT / "result.json", canonical(result))
    report_path = REPORT / "report-source.md"
    write_new(report_path, render_report(result).encode())
    write_new(
        REPORT / "run-manifest.json",
        canonical(
            {
                "experiment_id": EXPERIMENT_ID,
                "result_sha256": sha256(result_path),
                "arrays_sha256": sha256(array_path),
                "report_sha256": sha256(report_path),
                "fit_count": 12,
                "official_access": 0,
                "csv_materializations": 0,
                "uploads": 0,
            }
        ),
    )
    write_new(
        REPORT / "claim-source-ledger.md",
        b"# Claim-source ledger\n\n| Claim | Source | Use |\n|---|---|---|\n| Moist-air density depends on pressure, temperature and water-vapor content | Picard et al. 2008, DOI:10.1088/0026-1394/45/2/004 | physical motivation only |\n| Ocean momentum flux depends on air density and squared wind speed | Large and Pond 1981, DOI:10.1175/1520-0485(1981)011<0324:OOMFMI>2.0.CO;2 | physical motivation only |\n| No executed density-weighted P3 momentum path exists | repository semantic audit | novelty boundary |\n| Prior/official outputs were excluded | sealed v76 contract | reuse boundary |\n",
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "decision": result["decision"],
                "fit_count": 12,
                "official_access": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
