from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"C:\Users\cedis\PycharmProjects\PythonProject")
REPORT_DIR = ROOT / "reports" / "next_day_breakthrough_deep_research_20260827_v1"
OUTPUT = REPORT_DIR / "local_official_calibration.json"
ROUND_D = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260826_round_D_preregistered_P1x3_P2x3_P3x3"
)
OFFICIAL = ROUND_D / "OFFICIAL_RESULTS_20260826.json"
P1_PRED = ROOT / "artifacts" / "p1_matched_budget_local_compare_20260825_v1" / "predictions.parquet"
P1_TRUTH = ROOT / "artifacts" / "runs" / "20260813T153038+0900_cv_378a4e89" / "oof.parquet"
P2_OOF = ROOT / "artifacts" / "p2_authoritative_nested_surrogate_actual_20260825_v5" / "evaluated_oof_100.parquet"
P2_OBSERVATIONS = Path(
    r"C:\Users\cedis\Downloads\p2\데이터셋_P2"
    r"\P2_profile_restore\observations.csv"
)
P3_OOF = ROOT / "artifacts" / "p3_corrected_fixed_long_shrink_v4" / "oof.parquet"

P2_ALPHA = {2: 0.14425740489974548, 3: -0.09317805131171075, 4: -0.207647470805942}
P2_T = 0.15897699928943132


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def f1(y: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(y, dtype=np.int8)
    pred = np.asarray(prediction, dtype=np.int8)
    tp = int(((truth == 1) & (pred == 1)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    return 2.0 * tp / (2.0 * tp + fp + fn)


def rmse(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(y) - np.asarray(prediction)))))


def calibration_row(
    *,
    problem: str,
    family: str,
    contrast: str,
    local_delta: float,
    official_delta: float,
    higher_is_better: bool,
    comparability: str,
    caveat: str,
) -> dict[str, object]:
    local_gain = local_delta if higher_is_better else -local_delta
    official_gain = official_delta if higher_is_better else -official_delta
    sign_agreement = bool(
        (local_gain == 0 and official_gain == 0) or (local_gain * official_gain > 0)
    )
    magnitude_ratio = (
        abs(official_gain / local_gain) if abs(local_gain) > 1e-15 else None
    )
    return {
        "problem": problem,
        "family": family,
        "contrast": contrast,
        "local_delta_candidate_minus_anchor": local_delta,
        "official_delta_candidate_minus_anchor": official_delta,
        "local_gain_positive_is_better": local_gain,
        "official_gain_positive_is_better": official_gain,
        "sign_agreement": sign_agreement,
        "magnitude_ratio_abs_official_over_local": magnitude_ratio,
        "comparability_grade": comparability,
        "caveat": caveat,
    }


def p1_analysis() -> tuple[dict[str, object], list[dict[str, object]]]:
    predictions = pd.read_parquet(P1_PRED)
    truth = pd.read_parquet(P1_TRUTH)[["station", "year", "layer", "time", "label"]]
    keys = ["station", "year", "layer", "time"]
    merged = predictions.merge(truth, on=keys, how="left", validate="one_to_one")
    if merged["label"].isna().any():
        raise AssertionError("P1 truth merge failed")
    y = merged["label"].to_numpy(dtype=np.int8)
    o = merged["incumbent_offline_xgboost__default"].to_numpy(dtype=np.int8)
    a = merged["causal_event_rescue_ensemble__default"].to_numpy(dtype=np.int8)
    b = merged["event_day_balanced_lightgbm__default"].to_numpy(dtype=np.int8)
    station = merged["station"].astype(str).to_numpy()
    layer = merged["layer"].to_numpy(dtype=int)
    o_only = (o == 1) & (b == 0)
    b_only = (o == 0) & (b == 1)
    g_add = o_only & (station == "G-ORS") & (layer == 1)
    i_add = o_only & (station == "I-ORS") & (layer == 2)
    remove = b_only & (
        ((station == "S-ORS") & np.isin(layer, [1, 5, 6]))
        | ((station == "I-ORS") & (layer == 4))
    )
    candidates = {
        "O": o,
        "A": a,
        "B": b,
        "G": np.where(g_add, 1, b),
        "I": np.where(i_add, 1, b),
        "GI": np.where(g_add | i_add, 1, b),
        "Router": np.where(remove, 0, np.where(g_add | i_add, 1, b)),
        "Intersection": o & b,
        "Union": o | b,
    }
    scores = {name: f1(y, values) for name, values in candidates.items()}
    rows = [
        calibration_row(
            problem="P1",
            family="backbone",
            contrast="A vs O",
            local_delta=scores["A"] - scores["O"],
            official_delta=0.786145 - 0.790709,
            higher_is_better=True,
            comparability="B",
            caveat="Matched local family, but deployment surface and hidden official population differ.",
        ),
        calibration_row(
            problem="P1",
            family="backbone",
            contrast="B vs O",
            local_delta=scores["B"] - scores["O"],
            official_delta=0.793710 - 0.790709,
            higher_is_better=True,
            comparability="B",
            caveat="Direction transported once; uncertainty and station/fold heterogeneity remain.",
        ),
        calibration_row(
            problem="P1",
            family="disagreement_router",
            contrast="Router vs B",
            local_delta=scores["Router"] - scores["B"],
            official_delta=0.817873 - 0.793710,
            higher_is_better=True,
            comparability="A-",
            caveat="Same frozen cell rules; support frequency moved 9-16x for additions.",
        ),
        calibration_row(
            problem="P1",
            family="set_operation",
            contrast="Intersection vs B",
            local_delta=scores["Intersection"] - scores["B"],
            official_delta=0.802928 - 0.793710,
            higher_is_better=True,
            comparability="A-",
            caveat="Same file operation, but F1 is nonlinear and official evaluation may use a hidden subset.",
        ),
        calibration_row(
            problem="P1",
            family="set_operation",
            contrast="Union vs B",
            local_delta=scores["Union"] - scores["B"],
            official_delta=0.782306 - 0.793710,
            higher_is_better=True,
            comparability="A-",
            caveat="Same file operation, but F1 is nonlinear and official evaluation may use a hidden subset.",
        ),
    ]
    analysis = {
        "rows": int(len(merged)),
        "scores": scores,
        "local_cell_counts": {
            "G_add": int(g_add.sum()),
            "I_add": int(i_add.sum()),
            "removal": int(remove.sum()),
        },
        "local_cell_truth_utility": {
            "G_add_beneficial": int(y[g_add].sum()),
            "G_add_harmful": int(g_add.sum() - y[g_add].sum()),
            "I_add_beneficial": int(y[i_add].sum()),
            "I_add_harmful": int(i_add.sum() - y[i_add].sum()),
            "removal_beneficial": int((y[remove] == 0).sum()),
            "removal_harmful": int((y[remove] == 1).sum()),
        },
        "test_cell_counts": {"G_add": 81, "I_add": 136, "removal": 12},
        "support_rate_ratio_test_over_local": {
            "G_add": (81 / 169_011) / (int(g_add.sum()) / len(merged)),
            "I_add": (136 / 169_011) / (int(i_add.sum()) / len(merged)),
            "removal": (12 / 169_011) / (int(remove.sum()) / len(merged)),
        },
        "decision": "Use family/cell-specific diagnostics; do not fit a global local-to-official scalar.",
    }
    return analysis, rows


def rowwise_envelope(
    frame: pd.DataFrame, values: np.ndarray, endpoints: pd.DataFrame
) -> tuple[np.ndarray, int, int]:
    keyed = frame[["time"]].copy()
    keyed["time"] = pd.to_datetime(keyed["time"], utc=True)
    keyed["_row"] = np.arange(len(keyed))
    public = endpoints.copy()
    public["time"] = pd.to_datetime(public["time"], utc=True)
    merged = keyed.merge(public, on="time", how="left", validate="many_to_one")
    available = np.isfinite(merged["temp_1"]) & np.isfinite(merged["temp_5"])
    selected = merged.loc[available]
    rows = selected["_row"].to_numpy(dtype=int)
    output = values.copy()
    output[rows] = np.clip(
        values[rows],
        np.minimum(selected["temp_1"], selected["temp_5"]),
        np.maximum(selected["temp_1"], selected["temp_5"]),
    )
    active = int((np.abs(output - values) > 1e-12).sum())
    return output, int(available.sum()), active


def p2_analysis() -> tuple[dict[str, object], list[dict[str, object]]]:
    sys.path.insert(0, str(ROOT / "src"))
    from p2_restore.profile_projection import (  # noqa: PLC0415
        project_profiles_vectorized,
        public_endpoint_frame,
    )

    data = pd.read_parquet(P2_OOF)
    y = data["truth"].to_numpy(float)
    o = data["INCUMBENT_NOOP"].to_numpy(float)
    a = data["STACK_W0625"].to_numpy(float)
    axis = a - o
    layer = data["layer"].to_numpy(dtype=int)
    candidates: dict[str, np.ndarray] = {"O": o}
    for name, selected_layers in {
        "global_neg_t": {2, 3, 4},
        "L2_neg_t": {2},
        "L3_neg_t": {3},
        "L4_neg_t": {4},
    }.items():
        values = o.copy()
        mask = np.isin(layer, sorted(selected_layers))
        values[mask] -= P2_T * axis[mask]
        candidates[name] = values
    u = o.copy()
    for current_layer, alpha in P2_ALPHA.items():
        mask = layer == current_layer
        u[mask] += alpha * axis[mask]
    observations = pd.read_csv(P2_OBSERVATIONS, usecols=["time", "layer", "temp"])
    endpoints = public_endpoint_frame(observations)
    e, envelope_eligible, envelope_active = rowwise_envelope(data, u, endpoints)
    f_result = project_profiles_vectorized(data[["station", "time", "layer"]], u, endpoints)
    candidates.update({"U": u, "E": e, "F": f_result.prediction})
    scores = {name: rmse(y, values) for name, values in candidates.items()}
    official_l3 = math.sqrt(0.541085**2 + (-0.000147268316))
    rows = [
        calibration_row(
            problem="P2",
            family="A-O axis",
            contrast="global -t vs O",
            local_delta=scores["global_neg_t"] - scores["O"],
            official_delta=0.537238 - 0.541085,
            higher_is_better=False,
            comparability="C",
            caveat="Same algebra but local surrogate and official A-O directions are not exact-lineage matches.",
        ),
        calibration_row(
            problem="P2",
            family="A-O axis",
            contrast="L2 -t vs O",
            local_delta=scores["L2_neg_t"] - scores["O"],
            official_delta=0.541917 - 0.541085,
            higher_is_better=False,
            comparability="C",
            caveat="Layer-2 sign reverses; local axis is unsuitable for layer selection.",
        ),
        calibration_row(
            problem="P2",
            family="A-O axis",
            contrast="L3 -t vs O",
            local_delta=scores["L3_neg_t"] - scores["O"],
            official_delta=official_l3 - 0.541085,
            higher_is_better=False,
            comparability="C",
            caveat="Official L3 score is inferred by additive MSE decomposition.",
        ),
        calibration_row(
            problem="P2",
            family="A-O axis",
            contrast="L4 -t vs O",
            local_delta=scores["L4_neg_t"] - scores["O"],
            official_delta=0.536536 - 0.541085,
            higher_is_better=False,
            comparability="C",
            caveat="Direction agrees but magnitude differs by more than an order of magnitude.",
        ),
    ]
    analysis = {
        "rows": int(len(data)),
        "scores": scores,
        "upcoming_local_deltas_vs_O": {
            name: scores[name] - scores["O"] for name in ("U", "E", "F")
        },
        "postprocess": {
            "envelope_eligible_rows": envelope_eligible,
            "envelope_active_rows": envelope_active,
            "full_eligible_rows": int(f_result.eligible_mask.sum()),
            "full_active_rows": int(f_result.active_mask.sum()),
        },
        "decision": (
            "The official all-row quadratic is algebraically reliable; local surrogate scores are only weak "
            "postprocess ordering evidence and must not replace it."
        ),
    }
    return analysis, rows


def p3_analysis() -> tuple[dict[str, object], list[dict[str, object]]]:
    data = pd.read_parquet(P3_OOF)
    y = data["target_hs"].to_numpy(float)
    o = data["final_prediction"].to_numpy(float)
    a = data["candidate_prediction"].to_numpy(float)
    axis = a - o
    lead = data["lead_h"].to_numpy(dtype=int)
    candidates = {"O": o, "A": a}
    # Local B is the 22.5% shrink midpoint between the 20% and 25% recipes.
    candidates["B"] = o + 0.5 * axis
    for name, active_leads, alpha in (
        ("C1", {12, 18, 24}, -2.0),
        ("C2", {12, 18, 24}, -4.0),
        ("C3", {18, 24}, -4.0),
        ("L12_neg2", {12}, -2.0),
        ("L1824_neg2", {18, 24}, -2.0),
    ):
        values = o.copy()
        mask = np.isin(lead, sorted(active_leads))
        values[mask] += alpha * axis[mask]
        candidates[name] = values
    scores = {name: rmse(y, values) for name, values in candidates.items()}
    rows = [
        calibration_row(
            problem="P3",
            family="shrink coefficient",
            contrast="A vs O",
            local_delta=scores["A"] - scores["O"],
            official_delta=0.611680 - 0.607071,
            higher_is_better=False,
            comparability="D",
            caveat="Official A includes base-lineage drift and early-lead changes absent from local A.",
        ),
        calibration_row(
            problem="P3",
            family="shrink coefficient",
            contrast="B vs O",
            local_delta=scores["B"] - scores["O"],
            official_delta=0.609346 - 0.607071,
            higher_is_better=False,
            comparability="C-",
            caveat="B is long-lead midpoint officially, but score population is a hidden 66-case subset.",
        ),
        calibration_row(
            problem="P3",
            family="reverse long axis",
            contrast="C1/long -2 vs O",
            local_delta=scores["C1"] - scores["O"],
            official_delta=0.5989869937185615 - 0.607071,
            higher_is_better=False,
            comparability="C",
            caveat="Official value is preregistered from exact disjoint-support additivity, not yet observed.",
        ),
        calibration_row(
            problem="P3",
            family="reverse long axis",
            contrast="L12 -2 vs O",
            local_delta=scores["L12_neg2"] - scores["O"],
            official_delta=0.606681 - 0.607071,
            higher_is_better=False,
            comparability="C",
            caveat="Same lead support but local and official delta lineages are not identical.",
        ),
        calibration_row(
            problem="P3",
            family="reverse long axis",
            contrast="L18/24 -2 vs O",
            local_delta=scores["L1824_neg2"] - scores["O"],
            official_delta=0.599382 - 0.607071,
            higher_is_better=False,
            comparability="C",
            caveat="Same lead support but local and official delta lineages are not identical.",
        ),
    ]
    analysis = {
        "rows": int(len(data)),
        "scores": scores,
        "upcoming_local_deltas_vs_O": {
            name: scores[name] - scores["O"] for name in ("C1", "C2", "C3")
        },
        "decision": (
            "Reject the current local analogue as a scalar selection metric for this official correction family. "
            "Use C1 as the scorer/lineage guard and C2/C3 to identify Public curvature."
        ),
    }
    return analysis, rows


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    REPORT_DIR.mkdir(parents=True, exist_ok=False)
    p1, rows1 = p1_analysis()
    p2, rows2 = p2_analysis()
    p3, rows3 = p3_analysis()
    rows = [*rows1, *rows2, *rows3]
    comparable = [row for row in rows if row["comparability_grade"] in {"A", "A-", "B"}]
    payload = {
        "schema_version": "ocean_hackathon.local_official_calibration_20260827.v1",
        "status": "PASS",
        "global_scalar_fit_performed": False,
        "reason_no_global_fit": (
            "n is small, observations are family-correlated, treatment lineages differ, and sign reversals occur."
        ),
        "problem_analysis": {"P1": p1, "P2": p2, "P3": p3},
        "calibration_rows": rows,
        "summary": {
            "rows": len(rows),
            "sign_agreement_all": int(sum(bool(row["sign_agreement"]) for row in rows)),
            "sign_agreement_high_comparability": int(
                sum(bool(row["sign_agreement"]) for row in comparable)
            ),
            "high_comparability_rows": len(comparable),
            "policy": {
                "P1": "Use cell-specific support transport and official factorial contrasts.",
                "P2": "Trust official all-row quadratic algebra; use local only for postprocess ordering.",
                "P3": "Do not use current local analogue for selection; identify hidden-Public curves with official probes.",
            },
        },
        "source_hashes": {
            str(path): sha256(path)
            for path in (OFFICIAL, P1_PRED, P1_TRUTH, P2_OOF, P2_OBSERVATIONS, P3_OOF)
        },
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
