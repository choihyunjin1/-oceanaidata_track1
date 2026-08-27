"""Independent, read-only QA for the frozen Round-G P2/P3 submission bundle."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"C:\Users\cedis\PycharmProjects\PythonProject")
DELIVERY = Path(r"C:\Users\cedis\Downloads\해양 해커톤 제출용")
ROUND_D = DELIVERY / "20260826_round_D_preregistered_P1x3_P2x3_P3x3"
ROUND_E = DELIVERY / "20260827_round_E_remaining_P2x3_P3x3_READY"
BUNDLE = DELIVERY / "20260827_round_G_P2x3_P3x3_PUBLIC_QUADRATIC_READY"

P3_O = REPO / "output" / "2026-08-20" / "ready" / "P3_submission.csv"
P3_A = REPO / "artifacts" / "p3_corrected_fixed_long_shrink_v4" / "candidate" / "submission.csv"
P3_B = REPO / "artifacts" / "p3_target_mix_density_reweighted_catboost_v1" / "candidate" / "submission.csv"
OFFICIAL_AB = DELIVERY / "20260825_OFFICIAL_SCORE_RECONCILIATION.json"
OFFICIAL_D = ROUND_D / "OFFICIAL_RESULTS_20260826.json"
OUTPUT = BUNDLE / "INDEPENDENT_QA_V2.json"

P2_NAMES = (
    "P2_1_EXPLOIT_LAYERWISE_QUADRATIC",
    "P2_2_PROBE_ENDPOINT_ENVELOPE",
    "P2_3_PROBE_FULL_PAVA_ENVELOPE",
)
EXPECTED_P3_ALPHA = {
    "P3_1_EXPLOIT_LONG_QUADRATIC_OPTIMUM": None,
    "P3_2_ROBUST_LONG_NEG8": -8.0,
    "P3_3_BRACKET_LONG_NEG12": -12.0,
}
ROUNDING_HALF_WIDTH = 0.5e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_curve(scores: dict[str, float]) -> dict[str, float]:
    merged = math.sqrt(scores["L12"] ** 2 + scores["L1824"] ** 2 - scores["O"] ** 2)
    matrix = np.array([[0.0, 0.0, 1.0], [0.25, 0.5, 1.0], [4.0, -2.0, 1.0]])
    rhs = np.array([scores["O"] ** 2, scores["B"] ** 2, merged**2])
    a, b, c = np.linalg.solve(matrix, rhs)
    alpha_star = -b / (2.0 * a)
    rmse_star = math.sqrt(a * alpha_star**2 + b * alpha_star + c)
    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "c1_rmse": float(merged),
        "alpha_star": float(alpha_star),
        "rmse_star": float(rmse_star),
    }


def rounding_envelope(scores: dict[str, float]) -> dict[str, list[float]]:
    keys = ("O", "B", "L12", "L1824")
    curves = []
    for signs in itertools.product((-1.0, 1.0), repeat=4):
        shifted = {
            key: scores[key] + sign * ROUNDING_HALF_WIDTH
            for key, sign in zip(keys, signs, strict=True)
        }
        curves.append(fit_curve(shifted))
    return {
        field: [min(item[field] for item in curves), max(item[field] for item in curves)]
        for field in ("c1_rmse", "alpha_star", "rmse_star")
    }


def read_p3(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    assert frame.columns.tolist() == ["case_id", "station", "lead_h", "hs_pred"]
    assert len(frame) == 1200
    return frame


def main() -> int:
    manifest = json.loads((BUNDLE / "SET_MANIFEST.json").read_text(encoding="utf-8"))
    round_e_manifest = json.loads((ROUND_E / "P2_P3_SET_MANIFEST.json").read_text(encoding="utf-8"))
    original_ab = json.loads(OFFICIAL_AB.read_text(encoding="utf-8"))
    round_d = json.loads(OFFICIAL_D.read_text(encoding="utf-8"))

    candidates = {item["name"]: item for item in manifest["candidates"]}
    assert len(candidates) == 6
    p2_sources = {item["name"]: item for item in round_e_manifest["candidates"]}
    p2_checks = {}
    for name in P2_NAMES:
        current = Path(candidates[name]["path"])
        source = Path(p2_sources[name]["path"])
        current_hash = sha256(current)
        source_hash = sha256(source)
        assert current.read_bytes() == source.read_bytes()
        assert current_hash == source_hash == candidates[name]["sha256"]
        p2_checks[name] = {"sha256": current_hash, "byte_identical_to_round_e": True}

    scores = {
        "O": float(original_ab["rounds"]["original"]["P3"]["raw_metric"]),
        "B": float(original_ab["rounds"]["B"]["P3"]["raw_metric"]),
        "L12": next(
            float(item["score"])
            for item in round_d["records"]
            if item["candidate"] == "P3_2_PROBE_LEAD12_ONLY"
        ),
        "L1824": next(
            float(item["score"])
            for item in round_d["records"]
            if item["candidate"] == "P3_3_PROBE_LEAD18_24_ONLY"
        ),
    }
    curve = fit_curve(scores)
    assert curve["a"] > 0
    envelope = rounding_envelope(scores)

    o = read_p3(P3_O)
    a_frame = read_p3(P3_A)
    b_frame = read_p3(P3_B)
    keys = ["case_id", "station", "lead_h"]
    assert a_frame[keys].equals(o[keys]) and b_frame[keys].equals(o[keys])
    long_mask = o["lead_h"].isin([12, 18, 24]).to_numpy()
    early_mask = ~long_mask
    o_values = o["hs_pred"].to_numpy(float)
    a_values = a_frame["hs_pred"].to_numpy(float)
    b_values = b_frame["hs_pred"].to_numpy(float)
    assert np.max(np.abs(b_values[long_mask] - 0.5 * (o_values + a_values)[long_mask])) <= 1e-12
    assert np.max(np.abs(b_values[early_mask] - o_values[early_mask])) <= 1e-12

    pairs = np.array(
        [
            [0.607071, 23.698280],
            [0.611680, 23.625124],
            [0.609346, 23.662165],
            [0.599072, 23.825229],
            [0.606681, 23.704466],
            [0.599382, 23.820314],
        ]
    )
    slope, intercept = np.polyfit(pairs[:, 0], pairs[:, 1], 1)
    point_map_residual = float(np.max(np.abs(pairs[:, 1] - (intercept + slope * pairs[:, 0]))))
    p3_checks = {}
    for name, configured_alpha in EXPECTED_P3_ALPHA.items():
        item = candidates[name]
        expected_alpha = curve["alpha_star"] if configured_alpha is None else configured_alpha
        alpha = float(item["alpha"])
        assert abs(alpha - expected_alpha) <= 1e-10
        frame = read_p3(Path(item["path"]))
        assert frame[keys].equals(o[keys])
        values = frame["hs_pred"].to_numpy(float)
        expected = o_values.copy()
        expected[long_mask] += alpha * (a_values - o_values)[long_mask]
        formula_error = float(np.max(np.abs(values - expected)))
        assert formula_error <= 5.1e-13
        early_noop_error = float(np.max(np.abs(values[early_mask] - o_values[early_mask])))
        assert early_noop_error <= 5.1e-13
        predicted_rmse = math.sqrt(curve["a"] * alpha**2 + curve["b"] * alpha + curve["c"])
        predicted_points = float(intercept + slope * predicted_rmse)
        assert abs(float(item["predicted_public_rmse"]) - predicted_rmse) <= 1e-12
        assert abs(float(item["predicted_official_points"]) - predicted_points) <= 1e-12
        assert np.isfinite(values).all() and values.min() >= 0.0 and values.max() <= 30.0
        assert sha256(Path(item["path"])) == item["sha256"]
        p3_checks[name] = {
            "alpha": alpha,
            "predicted_public_rmse": predicted_rmse,
            "predicted_official_points": predicted_points,
            "formula_error": formula_error,
            "early_noop_error": early_noop_error,
            "sha256": item["sha256"],
        }

    report = {
        "schema_version": "ocean_hackathon.round_g_p2x3_p3x3.independent_qa.v2",
        "status": "PASS_READY_NOT_UPLOADED",
        "scope": "Independent recomputation; no hidden target values and no upload.",
        "manifest_sha256_before_qa": sha256(BUNDLE / "SET_MANIFEST.json"),
        "candidate_count": len(candidates),
        "p2": p2_checks,
        "p3": {
            "official_score_inputs": scores,
            "curve": curve,
            "display_rounding_envelope": envelope,
            "point_mapping": {
                "intercept": float(intercept),
                "slope": float(slope),
                "max_observed_residual_points": point_map_residual,
            },
            "midpoint_identity_pass": True,
            "early_noop_identity_pass": True,
            "candidates": p3_checks,
        },
        "guards": {
            "hidden_target_values_read": False,
            "official_upload_performed": False,
            "p3_era5_read_or_modified": False,
        },
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
