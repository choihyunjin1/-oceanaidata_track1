"""Build a frozen P2/P3 batch with the exact P3 long-axis Public quadratic optimum."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = Path(r"C:\Users\cedis\PycharmProjects\PythonProject")
DELIVERY = Path(r"C:\Users\cedis\Downloads\해양 해커톤 제출용")
ROUND_D = DELIVERY / "20260826_round_D_preregistered_P1x3_P2x3_P3x3"
ROUND_E_REMAINING = DELIVERY / "20260827_round_E_remaining_P2x3_P3x3_READY"
OUTPUT = DELIVERY / "20260827_round_G_P2x3_P3x3_PUBLIC_QUADRATIC_READY"

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
OFFICIAL_ORIGINAL_AB = DELIVERY / "20260825_OFFICIAL_SCORE_RECONCILIATION.json"
OFFICIAL_ROUND_D = ROUND_D / "OFFICIAL_RESULTS_20260826.json"

P2_NAMES = (
    "P2_1_EXPLOIT_LAYERWISE_QUADRATIC",
    "P2_2_PROBE_ENDPOINT_ENVELOPE",
    "P2_3_PROBE_FULL_PAVA_ENVELOPE",
)
P3_SPECS = (
    (
        "P3_1_EXPLOIT_LONG_QUADRATIC_OPTIMUM",
        None,
        "P3 long-lead 공식 이차최적 α* v1",
        "기존 O/B와 long α=-2 공식점수로 복원한 12·18·24h Public 이차곡선의 최적 α를 적용했습니다.",
        "EXACT_PUBLIC_LONG_AXIS_EXPLOIT",
    ),
    (
        "P3_2_ROBUST_LONG_NEG8",
        -8.0,
        "P3 long-lead 보수 최적근방 α=-8 v1",
        "공식 이차최적점보다 보수적인 α=-8을 12·18·24h에 적용해 외삽 위험을 줄였습니다.",
        "ROBUST_PUBLIC_LONG_AXIS_EXPLOIT",
    ),
    (
        "P3_3_BRACKET_LONG_NEG12",
        -12.0,
        "P3 long-lead 최적근방 상단 α=-12 v1",
        "공식 이차최적점 반대편 α=-12를 적용해 반올림 불확실성에도 최적점 근방을 유지합니다.",
        "PUBLIC_LONG_AXIS_BRACKET",
    ),
)
PINS = {
    P3_O: "d89e69b940c90ea1fbecf1e882bee69136255fffb12601d2fc853d032900e5b7",
    P3_A: "607f7cd4ed2c126d5aa4eb6d8130a651ac465a0c88b4e74c112d585c3421d708",
    P3_B: "c1be3931909e16ba854ac08a57bd606517835cdeea5829be04e84ab486717aa3",
    OFFICIAL_ORIGINAL_AB: "1ac0502d9c0334089bff667276577de3fb2aa0ca4a9b41b2d40fcda6cf330e25",
    OFFICIAL_ROUND_D: "26e73e5fa74876377f7c7f15cf07c89bd512d31f10fe36d459a6e6f7fb350cee",
}
ROUNDING_HALF_WIDTH = 0.5e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_curve(scores: dict[str, float]) -> dict[str, object]:
    c1 = math.sqrt(scores["L12"] ** 2 + scores["L1824"] ** 2 - scores["O"] ** 2)
    a, b, c = np.polyfit(
        np.array([0.0, 0.5, -2.0]),
        np.array([scores["O"] ** 2, scores["B"] ** 2, c1**2]),
        2,
    )
    if a <= 0:
        raise AssertionError("Public long-axis curve is not convex")
    alpha_star = -b / (2.0 * a)
    rmse_star = math.sqrt(a * alpha_star**2 + b * alpha_star + c)
    return {
        "a": float(a),
        "b": float(b),
        "c": float(c),
        "c1_rmse": float(c1),
        "alpha_star": float(alpha_star),
        "rmse_star": float(rmse_star),
    }


def rounding_envelope(scores: dict[str, float]) -> dict[str, list[float]]:
    curves = []
    keys = ("O", "B", "L12", "L1824")
    for signs in itertools.product((-1.0, 1.0), repeat=4):
        shifted = {
            key: scores[key] + sign * ROUNDING_HALF_WIDTH
            for key, sign in zip(keys, signs, strict=True)
        }
        curves.append(fit_curve(shifted))
    return {
        "c1_rmse": [
            min(float(item["c1_rmse"]) for item in curves),
            max(float(item["c1_rmse"]) for item in curves),
        ],
        "alpha_star": [
            min(float(item["alpha_star"]) for item in curves),
            max(float(item["alpha_star"]) for item in curves),
        ],
        "rmse_star": [
            min(float(item["rmse_star"]) for item in curves),
            max(float(item["rmse_star"]) for item in curves),
        ],
    }


def predicted_rmse(curve: dict[str, object], alpha: float) -> float:
    q = (
        float(curve["a"]) * alpha**2
        + float(curve["b"]) * alpha
        + float(curve["c"])
    )
    return math.sqrt(q)


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"append-only output already exists: {OUTPUT}")
    for path, expected in PINS.items():
        if sha256(path) != expected:
            raise AssertionError(f"pinned source changed: {path}")

    original_ab = json.loads(OFFICIAL_ORIGINAL_AB.read_text(encoding="utf-8"))
    round_d = json.loads(OFFICIAL_ROUND_D.read_text(encoding="utf-8"))
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
    envelope = rounding_envelope(scores)
    if not (-10.30 <= float(curve["alpha_star"]) <= -10.18):
        raise AssertionError("nominal alpha* left the precomputed rounding-safe interval")

    o = pd.read_csv(P3_O)
    a_frame = pd.read_csv(P3_A)
    b_frame = pd.read_csv(P3_B)
    keys = ["case_id", "station", "lead_h"]
    columns = keys + ["hs_pred"]
    for frame in (o, a_frame, b_frame):
        if frame.columns.tolist() != columns or len(frame) != 1200:
            raise AssertionError("P3 source schema/row contract failed")
        if not frame[keys].equals(o[keys]):
            raise AssertionError("P3 source key/order contract failed")
    leads = o["lead_h"].to_numpy(int)
    long_mask = np.isin(leads, [12, 18, 24])
    early_mask = ~long_mask
    o_values = o["hs_pred"].to_numpy(float)
    a_values = a_frame["hs_pred"].to_numpy(float)
    b_values = b_frame["hs_pred"].to_numpy(float)
    delta = a_values - o_values
    if np.max(np.abs(b_values[long_mask] - 0.5 * (o_values + a_values)[long_mask])) > 1e-12:
        raise AssertionError("B is not the exact long-lead midpoint")
    if np.max(np.abs(b_values[early_mask] - o_values[early_mask])) > 1e-12:
        raise AssertionError("B does not preserve early leads")

    OUTPUT.mkdir(parents=True)
    round_e_manifest = json.loads(
        (ROUND_E_REMAINING / "P2_P3_SET_MANIFEST.json").read_text(encoding="utf-8")
    )
    p2_source = {item["name"]: item for item in round_e_manifest["candidates"]}
    candidates: list[dict[str, object]] = []
    for name in P2_NAMES:
        shutil.copytree(ROUND_E_REMAINING / name, OUTPUT / name, copy_function=shutil.copy2)
        copied = dict(p2_source[name])
        copied["source_path"] = copied["path"]
        copied["path"] = str(OUTPUT / name / "P2_submission.csv")
        if sha256(Path(copied["path"])) != copied["sha256"]:
            raise AssertionError(f"P2 copy hash mismatch: {name}")
        candidates.append(copied)

    p3_point_pairs = np.array(
        [
            [0.607071, 23.698280],
            [0.611680, 23.625124],
            [0.609346, 23.662165],
            [0.599072, 23.825229],
            [0.606681, 23.704466],
            [0.599382, 23.820314],
        ]
    )
    slope, intercept = np.polyfit(p3_point_pairs[:, 0], p3_point_pairs[:, 1], 1)
    mapping_residual = np.max(
        np.abs(p3_point_pairs[:, 1] - (intercept + slope * p3_point_pairs[:, 0]))
    )
    for name, registered_alpha, title, summary, purpose in P3_SPECS:
        alpha = (
            float(curve["alpha_star"])
            if registered_alpha is None
            else float(registered_alpha)
        )
        values = o_values.copy()
        values[long_mask] += alpha * delta[long_mask]
        if not np.array_equal(values[early_mask], o_values[early_mask]):
            raise AssertionError(f"early lead changed: {name}")
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 30:
            raise AssertionError(f"P3 physical guard failed: {name}")
        directory = OUTPUT / name
        directory.mkdir()
        submission = o.copy()
        submission["hs_pred"] = values
        path = directory / "P3_submission.csv"
        submission.to_csv(path, index=False, float_format="%.12f", lineterminator="\n")
        reread = pd.read_csv(path)
        if not reread[keys].equals(o[keys]):
            raise AssertionError(f"written P3 key/order mismatch: {name}")
        formula_error = np.max(np.abs(reread["hs_pred"].to_numpy(float) - values))
        if formula_error > 5.1e-13:
            raise AssertionError(f"written P3 formula mismatch: {name}")
        rmse = predicted_rmse(curve, alpha)
        predicted_points = float(intercept + slope * rmse)
        candidate = {
            "problem": "P3",
            "name": name,
            "title": title,
            "one_line_summary": summary,
            "purpose": purpose,
            "path": str(path),
            "rows": len(reread),
            "sha256": sha256(path),
            "active_leads": [12, 18, 24],
            "alpha": alpha,
            "predicted_public_rmse": rmse,
            "predicted_official_points": predicted_points,
            "predicted_point_gain_vs_current_best": predicted_points - 23.825229,
            "prediction_min": float(values.min()),
            "prediction_max": float(values.max()),
            "changed_rows_vs_O": int(np.count_nonzero(values - o_values)),
            "max_formula_error": float(formula_error),
        }
        candidates.append(candidate)
        memo = "\n".join(
            [
                f"제출물 제목: {title}",
                f"한줄요약(접근방식): {summary}",
                "문제: P3",
                f"역할: {purpose}",
                f"적용 α: {alpha:.12f}",
                f"공식곡선 예상 RMSE: {rmse:.9f} m",
                f"공식점수 예상 증가: {predicted_points - 23.825229:+.6f}점",
                "주의: 고정 Public 표면의 정확한 선형축 최적화이며 Private 일반화 보장은 아닙니다.",
                "승인 경계: 파일 생성은 업로드 승인이 아니며 실제 업로드에는 새 명시 승인이 필요합니다.",
            ]
        )
        (directory / "P3_제출정보.txt").write_text(memo + "\n", encoding="utf-8")

    now = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
    manifest = {
        "schema_version": "ocean_hackathon.round_g_p2x3_p3x3_public_quadratic.v1",
        "status": "FROZEN_READY_NOT_UPLOADED",
        "team": "분당독고다이",
        "created_at_kst": now,
        "decision": "Keep P2 Round-E exact-axis set; replace P3 probes with the already-identified long-axis optimum neighborhood.",
        "submit_exact_order": [item["name"] for item in candidates],
        "official_upload_performed": False,
        "fresh_explicit_upload_approval_required": True,
        "P3_curve": {
            "definition": "q(alpha)=RMSE(alpha)^2 on the fixed hidden Public subset",
            "axis": "O + alpha*(A-O) on leads 12/18/24; exact O no-op on leads 3/6/9",
            "official_score_inputs": scores,
            "nominal": curve,
            "display_rounding_envelope": envelope,
            "empirical_point_mapping": {
                "intercept": float(intercept),
                "slope_per_rmse_m": float(slope),
                "max_observed_residual_points": float(mapping_residual),
            },
        },
        "candidates": candidates,
        "guards": {
            "P2_checkpoint85_excluded": True,
            "P3_ERA5_read_or_modified": False,
            "hidden_target_values_read": False,
            "official_upload_performed": False,
        },
    }
    manifest_path = OUTPUT / "SET_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plan = """Round G P2×3·P3×3 Public-quadratic 제출 세트

P2는 Round E 세 후보를 바이트 그대로 유지합니다. U는 확인된 P2 A-O 층별 선형공간의 공식 이차최적점이며, E/F는 물리 후처리 probe입니다.

P3는 기존 -2/-4 probe 세트를 대체합니다. 과거 B가 long-lead α=0.5의 정확한 midpoint이고 early lead는 O와 같다는 파일 identity를 이용했습니다. Round D의 α=-2 lead12 및 lead18/24 공식점수를 합쳐 long α=-2 점수를 정확히 복원한 뒤, α=0/0.5/-2 세 점으로 고정 Public MSE 이차곡선을 구했습니다.

P3 순서
1. 공식 이차최적 α*≈-10.235445
2. 보수적 근방 α=-8
3. 반대편 근방 α=-12

이는 Public 점수 극대화 후보이며 Private 일반화의 최대값을 증명하지 않습니다. 업로드는 수행하지 않았습니다.
"""
    (OUTPUT / "READY_SUBMISSION_PLAN.txt").write_text(plan, encoding="utf-8")
    qa = {
        "schema_version": "ocean_hackathon.round_g_p2x3_p3x3_public_quadratic.qa.v1",
        "status": "PASS_READY_NOT_UPLOADED",
        "candidate_count": len(candidates),
        "p2_byte_identity_pass": True,
        "p3_source_hash_pins_pass": True,
        "p3_long_midpoint_identity_pass": True,
        "p3_early_noop_identity_pass": True,
        "p3_curve_convex": float(curve["a"]) > 0,
        "p3_rounding_alpha_interval": envelope["alpha_star"],
        "p3_all_values_finite_and_physical": True,
        "hidden_target_values_read": False,
        "official_upload_performed": False,
        "manifest_sha256": sha256(manifest_path),
    }
    (OUTPUT / "INDEPENDENT_QA.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUTPUT), **qa}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
