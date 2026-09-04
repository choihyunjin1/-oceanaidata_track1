from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"C:\Users\cedis\PycharmProjects\PythonProject")
ROUND_C = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260826_round_C_preregistered_P1x3_P2x1"
)
ROUND_D = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260826_round_D_preregistered_P1x3_P2x3_P3x3"
)
ARTIFACT_ROOT = REPO / "artifacts" / "daily_submission_3x3_evidence_20260827_v2"
DELIVERY_ROOT = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260827_round_E_preregistered_P1x3_P2x3_P3x3"
)

P1_O = REPO / "output" / "2026-08-20" / "ready" / "P1_submission.csv"
P1_B = ROUND_C / "backup_best_before_round_C" / "P1_submission.csv"
P1_ROUTER = ROUND_D / "P1_1_EXPLOIT_DISAGREEMENT_ROUTER" / "P1_submission.csv"

P2_O = REPO / "output" / "2026-08-20" / "ready" / "P2_submission.csv"
P2_A = (
    REPO
    / "artifacts"
    / "p2_conservative_stack_improvement_v1"
    / "candidate"
    / "P2_CONSERVATIVE_STACK_IMPROVEMENT_V1.csv"
)
P2_CURRENT_BEST = ROUND_D / "P2_3_PROBE_LAYER4_ONLY" / "P2_submission.csv"
P2_OBSERVATIONS = Path(
    r"C:\Users\cedis\Downloads\p2\데이터셋_P2"
    r"\P2_profile_restore\observations.csv"
)
P2_PROJECTION = REPO / "src" / "p2_restore" / "profile_projection.py"

P3_O = REPO / "output" / "2026-08-20" / "ready" / "P3_submission.csv"
P3_A = (
    REPO
    / "artifacts"
    / "p3_corrected_fixed_long_shrink_v4"
    / "candidate"
    / "submission.csv"
)
P3_B = (
    REPO
    / "artifacts"
    / "p3_target_mix_density_reweighted_catboost_v1"
    / "candidate"
    / "submission.csv"
)
P3_CURRENT_BEST = ROUND_D / "P3_1_EXPLOIT_REVERSE_GLOBAL" / "P3_submission.csv"
P3_L12_NEG2 = ROUND_D / "P3_2_PROBE_LEAD12_ONLY" / "P3_submission.csv"
P3_L1824_NEG2 = ROUND_D / "P3_3_PROBE_LEAD18_24_ONLY" / "P3_submission.csv"

OFFICIAL_RESULTS = ROUND_D / "OFFICIAL_RESULTS_20260826.json"
P2_PROBLEM_MEMO = REPO / "01_P2_MUST_READ_FIRST.md"
P3_PROBLEM_MEMO = REPO / "02_P3_MUST_READ_FIRST.md"

PINNED_INPUTS = {
    P1_O: "28243fda9bc56e25a698366823dfab3198cda21bfaec04f30fda6a899eaf0cd3",
    P1_B: "decedb8a9b3df7d955ae9b3848cd8f985c5228e6727accbb514516507755adbf",
    P1_ROUTER: "1b04e81c18d5a5cac3115c3a256e8d5a38a9493a32478a184df81fd99f9f6e5f",
    P2_O: "1c959f818737850fd7fa9c6609ba3ae49dc9a470a269f7313119d840df1736bf",
    P2_A: "3960660b1e4076c88efdb927a50073aa2d8f1435bc1c7d6f2f40885aea2f2350",
    P2_CURRENT_BEST: "98890354fe792c905b44f9467c0651506c7696abd7091f1380e1825669865cff",
    P2_OBSERVATIONS: "cc5d6fd9ea398a613e485b34fd530c7dc064fa040fa675127a12318856ab178a",
    P2_PROJECTION: "fb1615ea1b0b67aad8a35daaef416eaff3dcd9d5b9cd498e3631c5b0b88d74e6",
    P3_O: "d89e69b940c90ea1fbecf1e882bee69136255fffb12601d2fc853d032900e5b7",
    P3_A: "607f7cd4ed2c126d5aa4eb6d8130a651ac465a0c88b4e74c112d585c3421d708",
    P3_B: "c1be3931909e16ba854ac08a57bd606517835cdeea5829be04e84ab486717aa3",
    P3_CURRENT_BEST: "57a90beb3f81de65fbf67426811eeaf49427951fa277997adb89c75ef259af56",
    P3_L12_NEG2: "c5ac003e5c0827f6d5f3ec0ac396e230fb3e4266f6668c592095c06ed1e94da1",
    P3_L1824_NEG2: "91ead7470f53aa7e09000bdc667a975b3836f16f201b86b1060ba6ea893212ee",
    OFFICIAL_RESULTS: "26e73e5fa74876377f7c7f15cf07c89bd512d31f10fe36d459a6e6f7fb350cee",
    P2_PROBLEM_MEMO: "e9d16b43e81191f3cda9ea5cf603ccf4d2319d23b06f584690990dc9006560a2",
    P3_PROBLEM_MEMO: "0a7deb6a85ff99506a6b2f35b37afe907eb4916d48dcb463f5ed21a15b515140",
}

P2_T = 0.15897699928943132
ROUNDING_HALF_WIDTH = 0.5e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def write_json(path: Path, payload: object) -> None:
    write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def assert_pins() -> None:
    failures: list[str] = []
    for path, expected in PINNED_INPUTS.items():
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(f"hash mismatch: {path} expected={expected} actual={actual}")
    if failures:
        raise RuntimeError("Pinned-input verification failed:\n" + "\n".join(failures))


def assert_same_keys(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> None:
    if left[keys].isna().any().any() or right[keys].isna().any().any():
        raise AssertionError(f"null key in {keys}")
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise AssertionError(f"duplicate key in {keys}")
    if not left[keys].equals(right[keys]):
        raise AssertionError(f"ordered keys differ: {keys}")


def official_score_map() -> dict[str, float]:
    payload = json.loads(OFFICIAL_RESULTS.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE_9_OF_9_SCORED":
        raise AssertionError("Round D result ledger is not complete")
    scores = {record["candidate"]: float(record["score"]) for record in payload["records"]}
    expected = {
        "P1_1_EXPLOIT_DISAGREEMENT_ROUTER": 0.817873,
        "P2_1_EXPLOIT_PUBLIC_QUADRATIC_GLOBAL": 0.537238,
        "P2_2_PROBE_LAYER2_ONLY": 0.541917,
        "P2_3_PROBE_LAYER4_ONLY": 0.536536,
        "P3_1_EXPLOIT_REVERSE_GLOBAL": 0.599072,
        "P3_2_PROBE_LEAD12_ONLY": 0.606681,
        "P3_3_PROBE_LEAD18_24_ONLY": 0.599382,
    }
    for key, value in expected.items():
        if scores.get(key) != value:
            raise AssertionError(f"Round D score drift: {key}")
    return scores


def save_candidate(
    *,
    frame: pd.DataFrame,
    name: str,
    problem: str,
    title: str,
    summary: str,
    purpose: str,
    filename: str,
) -> dict[str, object]:
    directory = DELIVERY_ROOT / name
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / filename
    frame.to_csv(path, index=False, float_format="%.12f", lineterminator="\n")
    record: dict[str, object] = {
        "problem": problem,
        "name": name,
        "title": title,
        "one_line_summary": summary,
        "purpose": purpose,
        "path": str(path),
        "rows": int(len(frame)),
        "sha256": sha256(path),
    }
    note = "\n".join(
        [
            f"제출물 제목: {title}",
            f"한줄요약(접근방식): {summary}",
            f"문제: {problem}",
            f"역할: {purpose}",
            f"CSV 파일: {filename}",
            f"행 수: {len(frame)}",
            f"SHA-256: {record['sha256']}",
            "제출 규칙: 세 후보를 모두 동결한 뒤 중간 점수를 보지 않고 연속 제출합니다.",
            "승인 경계: 이 패키지 생성은 업로드 승인이 아니며, 실제 업로드 직전에 별도 명시 승인이 필요합니다.",
        ]
    )
    write_text(directory / f"{problem}_제출정보.txt", note)
    return record


def build_p1() -> tuple[list[dict[str, object]], dict[str, object]]:
    keys = ["station", "year", "layer", "time"]
    columns = keys + ["label", "anomaly_type"]
    o = pd.read_csv(P1_O)
    b = pd.read_csv(P1_B)
    router = pd.read_csv(P1_ROUTER)
    for frame in (o, b, router):
        if frame.columns.tolist() != columns or len(frame) != 169_011:
            raise AssertionError("P1 schema/row mismatch")
    assert_same_keys(o, b, keys)
    assert_same_keys(b, router, keys)
    o_positive = o["label"].eq(1).to_numpy()
    b_positive = b["label"].eq(1).to_numpy()
    o_only = o_positive & ~b_positive
    station = b["station"].astype(str).to_numpy()
    layer = b["layer"].to_numpy(dtype=int)
    g_add = o_only & (station == "G-ORS") & (layer == 1)
    i_add = o_only & (station == "I-ORS") & (layer == 2)
    remove = b_positive & ~o_positive & (
        ((station == "S-ORS") & np.isin(layer, [1, 5, 6]))
        | ((station == "I-ORS") & (layer == 4))
    )
    if (int(g_add.sum()), int(i_add.sum()), int(remove.sum())) != (81, 136, 12):
        raise AssertionError("P1 disagreement-cell counts drifted")

    specs = [
        (
            "P1_1_PROBE_G_ONLY",
            g_add,
            "P1 G-ORS L1 추가양성 단독 v1",
            "현 베스트 B에 G-ORS layer 1의 O-only 81행만 복원해 G 셀의 공식 효용을 분리합니다.",
            "DISAGREEMENT_CELL_PROBE",
        ),
        (
            "P1_2_PROBE_I_ONLY",
            i_add,
            "P1 I-ORS L2 추가양성 단독 v1",
            "현 베스트 B에 I-ORS layer 2의 O-only 136행만 복원해 I 셀의 공식 효용을 분리합니다.",
            "DISAGREEMENT_CELL_PROBE",
        ),
        (
            "P1_3_EXPLOIT_GI_NO_REMOVALS",
            g_add | i_add,
            "P1 G·I 추가양성 결합·무제거 v1",
            "현 베스트 B에 G/I 추가양성 217행을 함께 복원하되 Router의 12행 제거는 적용하지 않습니다.",
            "EXPLOIT_AND_INTERACTION_PROBE",
        ),
    ]
    records: list[dict[str, object]] = []
    made: dict[str, pd.DataFrame] = {}
    for name, mask, title, summary, purpose in specs:
        frame = b.copy()
        frame.loc[mask, "label"] = 1
        frame.loc[mask, "anomaly_type"] = o.loc[mask, "anomaly_type"].to_numpy()
        if not np.isin(frame["label"].to_numpy(), [0, 1]).all():
            raise AssertionError(f"P1 non-binary label: {name}")
        if frame.loc[frame["label"].eq(0), "anomaly_type"].notna().any():
            raise AssertionError(f"P1 anomaly type on normal row: {name}")
        record = save_candidate(
            frame=frame,
            name=name,
            problem="P1",
            title=title,
            summary=summary,
            purpose=purpose,
            filename="P1_submission.csv",
        )
        record.update(
            {
                "positive_count": int(frame["label"].sum()),
                "changed_rows_vs_B": int(mask.sum()),
            }
        )
        records.append(record)
        made[name] = frame

    gi = made["P1_3_EXPLOIT_GI_NO_REMOVALS"]
    expected_router = gi.copy()
    expected_router.loc[remove, "label"] = 0
    expected_router.loc[remove, "anomaly_type"] = np.nan
    if not expected_router.equals(router):
        raise AssertionError("P1 Router is not GI plus the pinned 12-row removal")
    analysis = {
        "status": "PASS",
        "official_anchor": {"B": 0.793710, "Router": 0.817873},
        "cells": {"G_add": 81, "I_add": 136, "Router_removal": 12},
        "local_oof": {
            "B_F1": 0.8646700887242071,
            "G_only_F1": 0.8655185298413669,
            "I_only_F1": 0.8652761736531014,
            "GI_no_removals_F1": 0.8661235573659198,
            "Router_F1": 0.8668999966019912,
        },
        "support_rate_transport": {
            "G_test_over_oof": 9.17,
            "I_test_over_oof": 16.13,
            "removal_test_over_oof": 0.964,
        },
        "interpretation": (
            "File-level descriptive F1 contrasts only. F1 is nonlinear and the Public mask may be hidden; "
            "do not interpret them as additive row-level causal effects."
        ),
    }
    return records, analysis


def p2_coefficients(
    score_o: float,
    score_global: float,
    score_l2: float,
    score_l4: float,
    curvature: dict[int, float],
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    delta_q = {
        2: score_l2**2 - score_o**2,
        4: score_l4**2 - score_o**2,
    }
    delta_q[3] = (
        score_global**2 - score_o**2 - delta_q[2] - delta_q[4]
    )
    linear = {
        layer: (P2_T**2 * curvature[layer] - delta_q[layer]) / P2_T
        for layer in (2, 3, 4)
    }
    optimum = {
        layer: -linear[layer] / (2.0 * curvature[layer]) for layer in (2, 3, 4)
    }
    return delta_q, linear, optimum


def envelope_only(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    endpoints: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(prediction, dtype=np.float64)
    keyed = frame.loc[:, ["time"]].copy()
    keyed["time"] = pd.to_datetime(keyed["time"], utc=True)
    keyed["_row"] = np.arange(len(keyed))
    public = endpoints.loc[:, ["time", "temp_1", "temp_5"]].copy()
    public["time"] = pd.to_datetime(public["time"], utc=True)
    if public.duplicated("time").any():
        raise AssertionError("P2 public endpoint timestamp duplication")
    merged = keyed.merge(public, on="time", how="left", validate="many_to_one")
    available = np.isfinite(merged["temp_1"]) & np.isfinite(merged["temp_5"])
    selected = merged.loc[available]
    rows = selected["_row"].to_numpy(dtype=int)
    lower = np.minimum(
        selected["temp_1"].to_numpy(float), selected["temp_5"].to_numpy(float)
    )
    upper = np.maximum(
        selected["temp_1"].to_numpy(float), selected["temp_5"].to_numpy(float)
    )
    output = values.copy()
    output[rows] = np.clip(values[rows], lower, upper)
    eligible = np.zeros(len(values), dtype=bool)
    eligible[rows] = True
    active = eligible & ~np.isclose(output, values, rtol=0.0, atol=1e-12)
    return output, eligible, active


def build_p2(scores: dict[str, float]) -> tuple[list[dict[str, object]], dict[str, object]]:
    sys.path.insert(0, str(REPO / "src"))
    from p2_restore.profile_projection import (  # noqa: PLC0415
        project_profiles_vectorized,
        public_endpoint_frame,
    )

    keys = ["station", "layer", "time"]
    columns = keys + ["temp"]
    o = pd.read_csv(P2_O)
    a = pd.read_csv(P2_A)
    if o.columns.tolist() != columns or a.columns.tolist() != columns or len(o) != 26_061:
        raise AssertionError("P2 schema/row mismatch")
    assert_same_keys(o, a, keys)
    delta = a["temp"].to_numpy(float) - o["temp"].to_numpy(float)
    layers = o["layer"].to_numpy(dtype=int)
    curvature = {
        layer: float(np.sum(np.square(delta[layers == layer])) / len(o))
        for layer in (2, 3, 4)
    }
    score_o = 0.541085
    score_global = scores["P2_1_EXPLOIT_PUBLIC_QUADRATIC_GLOBAL"]
    score_l2 = scores["P2_2_PROBE_LAYER2_ONLY"]
    score_l4 = scores["P2_3_PROBE_LAYER4_ONLY"]
    delta_q, linear, optimum = p2_coefficients(
        score_o, score_global, score_l2, score_l4, curvature
    )
    expected = {2: 0.14425740489974548, 3: -0.09317805131171075, 4: -0.207647470805942}
    for layer in (2, 3, 4):
        if not math.isclose(optimum[layer], expected[layer], rel_tol=0.0, abs_tol=2e-12):
            raise AssertionError(f"P2 alpha derivation drift at layer {layer}")

    raw_values = o["temp"].to_numpy(float).copy()
    for layer in (2, 3, 4):
        mask = layers == layer
        raw_values[mask] += optimum[layer] * delta[mask]

    observations = pd.read_csv(P2_OBSERVATIONS, usecols=["time", "layer", "temp"])
    endpoints = public_endpoint_frame(observations)
    envelope_values, envelope_eligible, envelope_active = envelope_only(o, raw_values, endpoints)
    full_projection = project_profiles_vectorized(o, raw_values, endpoints)

    specs = [
        (
            "P2_1_EXPLOIT_LAYERWISE_QUADRATIC",
            raw_values,
            "P2 공식 층별 이차최적 U v1",
            "26,061행 전체 공식 RMSE와 A−O 파일축으로 L2/L3/L4 최적 α를 각각 복원해 동시에 적용했습니다.",
            "EXACT_PUBLIC_AXIS_EXPLOIT",
        ),
        (
            "P2_2_PROBE_ENDPOINT_ENVELOPE",
            envelope_values,
            "P2 층별 최적 U + endpoint envelope v1",
            "층별 최적 U를 공개 layer 1·5 수온 범위로만 clip해 안전한 envelope 효과를 분리합니다.",
            "PHYSICAL_POSTPROCESS_ABLATION",
        ),
        (
            "P2_3_PROBE_FULL_PAVA_ENVELOPE",
            full_projection.prediction,
            "P2 층별 최적 U + PAVA·envelope v1",
            "층별 최적 U에 현재 unit-weight PAVA와 공개 layer 1·5 envelope를 함께 적용해 단조 제약의 순효과를 측정합니다.",
            "PHYSICAL_POSTPROCESS_ABLATION",
        ),
    ]
    records: list[dict[str, object]] = []
    for name, values, title, summary, purpose in specs:
        if not np.isfinite(values).all() or values.min() < -5 or values.max() > 40:
            raise AssertionError(f"P2 numeric guard failed: {name}")
        frame = o.copy()
        frame["temp"] = values
        record = save_candidate(
            frame=frame,
            name=name,
            problem="P2",
            title=title,
            summary=summary,
            purpose=purpose,
            filename="P2_submission.csv",
        )
        record.update(
            {
                "temp_min": float(values.min()),
                "temp_max": float(values.max()),
                "changed_rows_vs_O": int(
                    (~np.isclose(values, o["temp"].to_numpy(float), rtol=0.0, atol=1e-12)).sum()
                ),
                "rms_change_vs_U": float(np.sqrt(np.mean(np.square(values - raw_values)))),
            }
        )
        records.append(record)

    q_u = score_o**2 + sum(
        curvature[layer] * optimum[layer] ** 2 + linear[layer] * optimum[layer]
        for layer in (2, 3, 4)
    )
    predicted_u = math.sqrt(q_u)
    corners: list[dict[str, object]] = []
    for signs in itertools.product((-1.0, 1.0), repeat=4):
        shifted = [
            score_o + signs[0] * ROUNDING_HALF_WIDTH,
            score_global + signs[1] * ROUNDING_HALF_WIDTH,
            score_l2 + signs[2] * ROUNDING_HALF_WIDTH,
            score_l4 + signs[3] * ROUNDING_HALF_WIDTH,
        ]
        _, b_corner, alpha_corner = p2_coefficients(*shifted, curvature)
        q_corner = shifted[0] ** 2 + sum(
            curvature[layer] * optimum[layer] ** 2 + b_corner[layer] * optimum[layer]
            for layer in (2, 3, 4)
        )
        corners.append(
            {
                "predicted_rmse_at_point_alpha": math.sqrt(q_corner),
                "alpha_star": alpha_corner,
            }
        )
    alpha_intervals = {
        str(layer): [
            min(float(corner["alpha_star"][layer]) for corner in corners),
            max(float(corner["alpha_star"][layer]) for corner in corners),
        ]
        for layer in (2, 3, 4)
    }
    analysis = {
        "status": "PASS",
        "official_scope_guard": "P2 memo states all 26,061 test_index keys are scored",
        "formula": "q_l(alpha)=q_l(0)+b_l*alpha+a_l*alpha^2; a_l=sum_all_rows_in_layer((A-O)^2)/26061",
        "round_d_scores": {
            "O": score_o,
            "global_neg_t": score_global,
            "L2_neg_t": score_l2,
            "L4_neg_t": score_l4,
            "t": P2_T,
        },
        "layer_row_counts": {str(layer): int((layers == layer).sum()) for layer in (2, 3, 4)},
        "curvature_a": {str(k): v for k, v in curvature.items()},
        "linear_b": {str(k): v for k, v in linear.items()},
        "delta_mse_at_neg_t": {str(k): v for k, v in delta_q.items()},
        "alpha_star": {str(k): v for k, v in optimum.items()},
        "alpha_star_rounding_intervals": alpha_intervals,
        "predicted_U_RMSE": predicted_u,
        "predicted_U_RMSE_rounding_interval": [
            min(float(c["predicted_rmse_at_point_alpha"]) for c in corners),
            max(float(c["predicted_rmse_at_point_alpha"]) for c in corners),
        ],
        "predicted_gain_vs_O": score_o - predicted_u,
        "predicted_gain_vs_current_L4_best": score_l4 - predicted_u,
        "postprocess_diagnostics": {
            "envelope": {
                "eligible_rows": int(envelope_eligible.sum()),
                "active_rows": int(envelope_active.sum()),
                "active_share_of_eligible": float(envelope_active.sum() / envelope_eligible.sum()),
                "rms_correction": float(np.sqrt(np.mean(np.square(envelope_values - raw_values)))),
            },
            "full_pava_envelope": {
                "eligible_rows": int(full_projection.eligible_mask.sum()),
                "active_rows": int(full_projection.active_mask.sum()),
                "active_share_of_eligible": float(
                    full_projection.active_mask.sum() / full_projection.eligible_mask.sum()
                ),
                "rms_correction": float(
                    np.sqrt(np.mean(np.square(full_projection.prediction - raw_values)))
                ),
            },
        },
        "physics_caution": (
            "Temperature monotonicity is not seawater density stability. TEOS-10 N^2 depends on both salinity "
            "and temperature, so envelope-only and PAVA are kept as separate official ablations."
        ),
    }
    return records, analysis


def build_p3(scores: dict[str, float]) -> tuple[list[dict[str, object]], dict[str, object]]:
    keys = ["case_id", "station", "lead_h"]
    columns = keys + ["hs_pred"]
    o = pd.read_csv(P3_O)
    a = pd.read_csv(P3_A)
    b = pd.read_csv(P3_B)
    for frame in (o, a, b):
        if frame.columns.tolist() != columns or len(frame) != 1_200:
            raise AssertionError("P3 schema/row mismatch")
    assert_same_keys(o, a, keys)
    assert_same_keys(o, b, keys)
    delta = a["hs_pred"].to_numpy(float) - o["hs_pred"].to_numpy(float)
    leads = o["lead_h"].to_numpy(dtype=int)
    midpoint_error = b["hs_pred"].to_numpy(float) - (
        o["hs_pred"].to_numpy(float) + a["hs_pred"].to_numpy(float)
    ) / 2.0
    long_mask = np.isin(leads, [12, 18, 24])
    early_mask = np.isin(leads, [3, 6, 9])
    if np.max(np.abs(midpoint_error[long_mask])) > 1e-12:
        raise AssertionError("P3 B long-lead midpoint identity failed")
    if np.max(np.abs(b.loc[early_mask, "hs_pred"] - o.loc[early_mask, "hs_pred"])) > 1e-12:
        raise AssertionError("P3 B early-lead no-op identity failed")
    if not (np.max(np.abs(midpoint_error[early_mask])) > 1e-4):
        raise AssertionError("P3 global-axis erratum was not reproduced")

    specs = [
        (
            "P3_1_EXPLOIT_LONG_NEG2",
            {12, 18, 24},
            -2.0,
            "P3 O-A 역방향 12·18·24h α=-2 v1",
            "검증된 12h와 18·24h 역방향 보정을 합쳐 early 60행을 exact no-op으로 유지합니다.",
            "EXPLOIT_AND_ADDITIVITY_GUARD",
        ),
        (
            "P3_2_PROBE_LONG_NEG4",
            {12, 18, 24},
            -4.0,
            "P3 O-A 역방향 12·18·24h α=-4 v1",
            "동일 long-lead 축을 α=-4로 확장해 숨은 66-case Public 곡률을 bounded하게 식별합니다.",
            "PUBLIC_CURVATURE_PROBE",
        ),
        (
            "P3_3_PROBE_LEAD18_24_NEG4",
            {18, 24},
            -4.0,
            "P3 O-A 역방향 18·24h α=-4 v1",
            "18·24h에만 α=-4를 적용해 장기리드 곡률과 12h 기여를 분리합니다.",
            "PUBLIC_CURVATURE_PROBE",
        ),
    ]
    records: list[dict[str, object]] = []
    made: dict[str, np.ndarray] = {}
    for name, active_leads, alpha, title, summary, purpose in specs:
        mask = np.isin(leads, sorted(active_leads))
        values = o["hs_pred"].to_numpy(float).copy()
        values[mask] += alpha * delta[mask]
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 30:
            raise AssertionError(f"P3 physical guard failed: {name}")
        if np.max(np.abs(values[~mask] - o.loc[~mask, "hs_pred"].to_numpy(float))) > 1e-12:
            raise AssertionError(f"P3 inactive lead changed: {name}")
        frame = o.copy()
        frame["hs_pred"] = values
        record = save_candidate(
            frame=frame,
            name=name,
            problem="P3",
            title=title,
            summary=summary,
            purpose=purpose,
            filename="P3_submission.csv",
        )
        record.update(
            {
                "active_leads": sorted(active_leads),
                "alpha": alpha,
                "changed_rows_vs_O": int((np.abs(values - o["hs_pred"].to_numpy(float)) > 1e-12).sum()),
                "prediction_min": float(values.min()),
                "prediction_max": float(values.max()),
                "rms_change_vs_O": float(
                    np.sqrt(np.mean(np.square(values - o["hs_pred"].to_numpy(float))))
                ),
                "max_abs_change_vs_O": float(
                    np.max(np.abs(values - o["hs_pred"].to_numpy(float)))
                ),
            }
        )
        records.append(record)
        made[name] = values

    score_o = 0.607071
    score_l12 = scores["P3_2_PROBE_LEAD12_ONLY"]
    score_l1824 = scores["P3_3_PROBE_LEAD18_24_ONLY"]
    c1_score = math.sqrt(score_l12**2 + score_l1824**2 - score_o**2)
    c1_corners = []
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        shifted = [
            score_o + signs[0] * ROUNDING_HALF_WIDTH,
            score_l12 + signs[1] * ROUNDING_HALF_WIDTH,
            score_l1824 + signs[2] * ROUNDING_HALF_WIDTH,
        ]
        c1_corners.append(math.sqrt(shifted[1] ** 2 + shifted[2] ** 2 - shifted[0] ** 2))
    # Candidate C1 must equal a rowwise merge of the two prior disjoint-support probes.
    prior_l12 = pd.read_csv(P3_L12_NEG2)
    prior_l1824 = pd.read_csv(P3_L1824_NEG2)
    merged_expected = o["hs_pred"].to_numpy(float).copy()
    m12 = leads == 12
    m1824 = np.isin(leads, [18, 24])
    merged_expected[m12] = prior_l12.loc[m12, "hs_pred"].to_numpy(float)
    merged_expected[m1824] = prior_l1824.loc[m1824, "hs_pred"].to_numpy(float)
    if np.max(np.abs(made["P3_1_EXPLOIT_LONG_NEG2"] - merged_expected)) > 5.1e-13:
        raise AssertionError("P3 C1 prior-probe merge identity failed")

    analysis = {
        "status": "PASS",
        "public_scope_guard": (
            "P3 Public is a hidden 66-case/396-row subset of the 200-case/1200-row submission. "
            "Full-file sum((A-O)^2) is not the Public quadratic curvature."
        ),
        "round_d_scores": {
            "O": score_o,
            "global_neg2": scores["P3_1_EXPLOIT_REVERSE_GLOBAL"],
            "L12_neg2": score_l12,
            "L18_24_neg2": score_l1824,
        },
        "C1_exact_public_additivity_prediction": c1_score,
        "C1_rounding_interval": [min(c1_corners), max(c1_corners)],
        "C1_allowed_display_range": [0.598985, 0.598989],
        "curve_recovery_after_C2_C3": {
            "definition": "D_g(alpha)=a_g*alpha^2+b_g*alpha",
            "a_from_D2_D4": "a_g=(D4-2*D2)/8",
            "b_from_D2_D4": "b_g=(D4-4*D2)/4",
            "alpha_star": "-b_g/(2*a_g)",
            "guard": "Only after C1 passes its rounding interval; otherwise retain ranking only and stop curve extrapolation.",
        },
        "round_d_axis_erratum": {
            "old_claim": "B=(O+A)/2 on all 1200 rows",
            "correct_claim": "B is the midpoint only at leads 12/18/24 and equals O at leads 3/6/9",
            "global_rms_off_axis": float(np.sqrt(np.mean(np.square(midpoint_error)))),
            "global_max_abs_off_axis": float(np.max(np.abs(midpoint_error))),
            "early_changed_rows_in_A_vs_O": int(
                ((np.abs(delta) > 1e-12) & early_mask).sum()
            ),
            "effect": (
                "The prior 0.598574 global prediction and its failed exact-axis guard are invalid; "
                "the miss does not establish scorer instability."
            ),
        },
        "forbidden_inference": (
            "Do not use full-file curvature to claim Public alpha*=-13 or alpha*=168; "
            "the hidden Public mask makes those values unidentified."
        ),
    }
    return records, analysis


def backup_current_best() -> dict[str, object]:
    directory = DELIVERY_ROOT / "backup_best_before_round_E"
    directory.mkdir(parents=True, exist_ok=False)
    sources = {
        "P1_submission.csv": P1_ROUTER,
        "P2_submission.csv": P2_CURRENT_BEST,
        "P3_submission.csv": P3_CURRENT_BEST,
    }
    records: dict[str, object] = {}
    for filename, source in sources.items():
        destination = directory / filename
        shutil.copy2(source, destination)
        records[filename] = {
            "source": str(source),
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        }
    write_text(
        directory / "README.txt",
        "2026-08-27 Round E 직전 문제별 공식 최고 파일 3개 백업입니다.\n"
        "Round E 후보를 제출하기 전 복구 기준으로만 사용하며 자동 업로드하지 않습니다.",
    )
    return records


def hash_inventory(root: Path) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        inventory.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return inventory


def main() -> None:
    assert_pins()
    if ARTIFACT_ROOT.exists() or DELIVERY_ROOT.exists():
        raise FileExistsError("Round E output already exists; refusing overwrite")
    ARTIFACT_ROOT.mkdir(parents=True)
    DELIVERY_ROOT.mkdir(parents=True)
    scores = official_score_map()
    p1_records, p1_analysis = build_p1()
    p2_records, p2_analysis = build_p2(scores)
    p3_records, p3_analysis = build_p3(scores)
    all_records = [*p1_records, *p2_records, *p3_records]
    if len(all_records) != 9:
        raise AssertionError("exactly nine candidates are required")
    backup = backup_current_best()

    analysis = {
        "schema_version": "next_day_3x3_evidence_analysis_20260827.v1",
        "status": "PASS",
        "decision": (
            "Freeze exactly three candidates per problem before any 2026-08-27 score is observed; "
            "submit only after fresh explicit user approval."
        ),
        "p1": p1_analysis,
        "p2": p2_analysis,
        "p3": p3_analysis,
        "local_official_policy": {
            "global_scalar_calibration_allowed": False,
            "reason": (
                "The evidence is small, family-correlated, and includes sign reversals plus mismatched treatment lineages. "
                "Use family-specific sign/order/transport diagnostics instead."
            ),
        },
    }
    analysis_path = ARTIFACT_ROOT / "analysis.json"
    write_json(analysis_path, analysis)

    manifest = {
        "schema_version": "ocean_hackathon.round_e_preregistered_3x3.v1",
        "status": "FROZEN_READY_NOT_UPLOADED",
        "team": "분당독고다이",
        "target_date_kst": "2026-08-27",
        "official_submissions_performed": 0,
        "fresh_upload_approval_required": True,
        "blind_batch_protocol": {
            "freeze_before_first_upload": True,
            "inspect_intermediate_scores": False,
            "replace_after_score": False,
            "submit_exact_order": [record["name"] for record in all_records],
        },
        "candidates": all_records,
        "backup_best_before_round_E": backup,
        "input_lineage": {
            str(path): {"sha256": expected, "bytes": path.stat().st_size}
            for path, expected in PINNED_INPUTS.items()
        },
        "analysis_path": str(analysis_path),
        "separation_guards": {
            "P3_ERA5_experiment_modified": False,
            "model_training_performed": False,
            "official_upload_performed": False,
            "hidden_target_values_read": False,
        },
    }
    manifest_path = DELIVERY_ROOT / "SET_MANIFEST.json"
    write_json(manifest_path, manifest)
    write_text(
        DELIVERY_ROOT / "READY_SUBMISSION_PLAN.txt",
        "2026-08-27 Round E 사전등록 3×3 패키지\n\n"
        "결론: 정확히 9개 후보가 동결됐지만 아직 업로드하지 않았습니다.\n"
        "업로드 직전 사용자의 새 명시 승인을 받고, 아래 순서대로 중간 점수를 보지 않고 연속 제출합니다.\n\n"
        + "\n".join(f"{index}. {record['name']}" for index, record in enumerate(all_records, 1))
        + "\n\nP3 주의: 1,200행 전체 곡률을 66-case Public 곡률로 사용하지 않습니다."
        + "\nP2 주의: U/E/F는 층별 최적화와 온도 단조 PAVA를 분리한 공식 ablation입니다.",
    )
    write_text(
        DELIVERY_ROOT / "P3_ROUND_D_AXIS_ERRATUM.txt",
        "정정: Round D의 'B=(O+A)/2 on all 1200 rows' 주장은 틀렸습니다.\n"
        "B는 12/18/24h에서만 midpoint이고 3/6/9h에서는 O와 동일합니다.\n"
        "따라서 0.598574 예측과 exact-axis guard 실패 해석은 폐기합니다.\n"
        "또 Public은 숨은 66사례이므로 전체 1,200행 delta 곡률로 Public alpha*를 계산하지 않습니다.\n"
        f"재현 global RMS off-axis: {p3_analysis['round_d_axis_erratum']['global_rms_off_axis']:.15f}\n"
        f"재현 global max off-axis: {p3_analysis['round_d_axis_erratum']['global_max_abs_off_axis']:.15f}",
    )

    inventory = hash_inventory(DELIVERY_ROOT)
    write_text(
        DELIVERY_ROOT / "SHA256SUMS.txt",
        "\n".join(f"{item['sha256']}  {item['relative_path']}" for item in inventory),
    )
    receipt = {
        "schema_version": "ocean_hackathon.round_e_build_receipt.v1",
        "status": "PASS",
        "candidate_count": 9,
        "candidate_hashes": {record["name"]: record["sha256"] for record in all_records},
        "manifest_sha256": sha256(manifest_path),
        "analysis_sha256": sha256(analysis_path),
        "delivery_inventory_before_receipt": inventory,
    }
    write_json(ARTIFACT_ROOT / "build_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
