from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"C:\Users\cedis\PycharmProjects\PythonProject")
ROUND_C = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260826_round_C_preregistered_P1x3_P2x1"
)
ARTIFACT_ROOT = REPO / "artifacts" / "daily_submission_3x3_evidence_20260826_v1"
DELIVERY_ROOT = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260826_round_D_preregistered_P1x3_P2x3_P3x3"
)

P1_ORIGINAL = REPO / "output" / "2026-08-20" / "ready" / "P1_submission.csv"
P1_BEST = ROUND_C / "backup_best_before_round_C" / "P1_submission.csv"
P2_O = REPO / "output" / "2026-08-20" / "ready" / "P2_submission.csv"
P2_A = (
    REPO
    / "artifacts"
    / "p2_conservative_stack_improvement_v1"
    / "candidate"
    / "P2_CONSERVATIVE_STACK_IMPROVEMENT_V1.csv"
)
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

PINNED_INPUTS = {
    P1_ORIGINAL: "28243fda9bc56e25a698366823dfab3198cda21bfaec04f30fda6a899eaf0cd3",
    P1_BEST: "decedb8a9b3df7d955ae9b3848cd8f985c5228e6727accbb514516507755adbf",
    P2_O: "1c959f818737850fd7fa9c6609ba3ae49dc9a470a269f7313119d840df1736bf",
    P2_A: "3960660b1e4076c88efdb927a50073aa2d8f1435bc1c7d6f2f40885aea2f2350",
    P3_O: "d89e69b940c90ea1fbecf1e882bee69136255fffb12601d2fc853d032900e5b7",
    P3_A: "607f7cd4ed2c126d5aa4eb6d8130a651ac465a0c88b4e74c112d585c3421d708",
    P3_B: "c1be3931909e16ba854ac08a57bd606517835cdeea5829be04e84ab486717aa3",
}

P1_SOURCES = [
    (
        "P1_1_EXPLOIT_DISAGREEMENT_ROUTER",
        ROUND_C / "P1_1_EXPLOIT_DISAGREEMENT_ROUTER" / "P1_submission.csv",
        "1b04e81c18d5a5cac3115c3a256e8d5a38a9493a32478a184df81fd99f9f6e5f",
        "P1 disagreement router 확인 v1",
        "현 베스트 B와 구 모델의 불일치 중 로컬 시간검증에서 재현된 정점·층만 선택적으로 복원·제거했습니다.",
        "EXPLOIT_CONFIRMATION",
    ),
    (
        "P1_2_PROBE_INTERSECTION",
        ROUND_C / "P1_2_PROBE_INTERSECTION" / "P1_submission.csv",
        "0ac5a6abe623e59236f56164004d654fe7c4e448d9e64f47149cfd16d9d84be3",
        "P1 O∩B 추가양성 절제 실험 v1",
        "구 모델 O와 현 베스트 B의 양성 교집합만 남겨 B가 추가한 176개 양성의 공식 효용을 분리합니다.",
        "MECHANISM_PROBE",
    ),
    (
        "P1_3_PROBE_UNION",
        ROUND_C / "P1_3_PROBE_UNION" / "P1_submission.csv",
        "c8b72922f42dc0ea0ed2487c541850984b21f491c22a682db2355b4aac533ed6",
        "P1 O∪B 제거양성 복원 실험 v1",
        "구 모델 O와 현 베스트 B의 양성 합집합으로 B가 제거한 824개 양성의 공식 효용을 분리합니다.",
        "MECHANISM_PROBE",
    ),
]

P2_ALPHA = -0.15897699928943132
P3_ALPHA = -2.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.rstrip() + "\n")


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
    for _, path, expected, *_ in P1_SOURCES:
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


def validate_p1() -> dict[str, object]:
    keys = ["station", "year", "layer", "time"]
    columns = keys + ["label", "anomaly_type"]
    o = pd.read_csv(P1_ORIGINAL)
    b = pd.read_csv(P1_BEST)
    if o.columns.tolist() != columns or b.columns.tolist() != columns:
        raise AssertionError("P1 schema mismatch")
    if len(o) != 169_011 or len(b) != 169_011:
        raise AssertionError("P1 row count mismatch")
    assert_same_keys(o, b, keys)
    frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, object] = {}
    for name, path, expected_hash, *_ in P1_SOURCES:
        frame = pd.read_csv(path)
        if frame.columns.tolist() != columns or len(frame) != 169_011:
            raise AssertionError(f"P1 candidate shape/schema mismatch: {name}")
        assert_same_keys(b, frame, keys)
        labels = frame["label"].to_numpy()
        if not np.isin(labels, [0, 1]).all():
            raise AssertionError(f"P1 invalid labels: {name}")
        if frame.loc[frame["label"] == 0, "anomaly_type"].notna().any():
            raise AssertionError(f"P1 anomaly_type populated for label=0: {name}")
        if sha256(path) != expected_hash:
            raise AssertionError(f"P1 candidate hash mismatch: {name}")
        frames[name] = frame
        summaries[name] = {
            "rows": int(len(frame)),
            "positive_count": int(frame["label"].sum()),
            "differences_vs_current_best": int((frame["label"] != b["label"]).sum()),
            "sha256": expected_hash,
        }
    intersection = frames["P1_2_PROBE_INTERSECTION"]["label"].to_numpy()
    union = frames["P1_3_PROBE_UNION"]["label"].to_numpy()
    o_label = o["label"].to_numpy()
    b_label = b["label"].to_numpy()
    if not np.array_equal(intersection, o_label & b_label):
        raise AssertionError("P1 intersection set identity failed")
    if not np.array_equal(union, o_label | b_label):
        raise AssertionError("P1 union set identity failed")
    return {
        "status": "PASS",
        "keys": keys,
        "candidates": summaries,
        "factorial": {
            "o_positive": int(o_label.sum()),
            "b_positive": int(b_label.sum()),
            "o_only": int(((o_label == 1) & (b_label == 0)).sum()),
            "b_only": int(((o_label == 0) & (b_label == 1)).sum()),
            "identity": "I=O∩B; U=O∪B",
        },
    }


def make_p2_candidates(candidate_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    keys = ["station", "layer", "time"]
    columns = keys + ["temp"]
    o = pd.read_csv(P2_O)
    a = pd.read_csv(P2_A)
    if o.columns.tolist() != columns or a.columns.tolist() != columns:
        raise AssertionError("P2 schema mismatch")
    if len(o) != 26_061 or len(a) != 26_061:
        raise AssertionError("P2 row count mismatch")
    assert_same_keys(o, a, keys)
    if not np.isfinite(o["temp"]).all() or not np.isfinite(a["temp"]).all():
        raise AssertionError("P2 non-finite source")

    specs = [
        (
            "P2_1_EXPLOIT_PUBLIC_QUADRATIC_GLOBAL",
            None,
            "P2 공개 이차최적 전체층 v1",
            "기존 공식 α=0·0.5·1 RMSE로 복원한 MSE 이차곡선의 최적 α=-0.158977을 모든 층에 적용했습니다.",
            "EXPLOIT_AND_SCORING_CHECK",
        ),
        (
            "P2_2_PROBE_LAYER2_ONLY",
            {2},
            "P2 공개 이차최적 2층 단독 v1",
            "동일한 역방향 보정 α=-0.158977을 layer 2에만 적용해 공개 MSE의 2층 기여를 직접 측정합니다.",
            "LAYER_MECHANISM_PROBE",
        ),
        (
            "P2_3_PROBE_LAYER4_ONLY",
            {4},
            "P2 공개 이차최적 4층 단독 v1",
            "동일한 역방향 보정 α=-0.158977을 layer 4에만 적용해 가장 불확실한 층 기여를 직접 측정합니다.",
            "LAYER_MECHANISM_PROBE",
        ),
    ]
    expected_hashes = {
        "P2_1_EXPLOIT_PUBLIC_QUADRATIC_GLOBAL": "9cc951801cf6b6cdacc2c826126d9c2f72ef34fc67e46c6a21261c7a1ba845ff",
        "P2_2_PROBE_LAYER2_ONLY": "5507317f45bf06969d7da6c2ebd750bc5805564d1e3955920eab036724fc1ccc",
        "P2_3_PROBE_LAYER4_ONLY": "98890354fe792c905b44f9467c0651506c7696abd7091f1380e1825669865cff",
    }
    records: list[dict[str, object]] = []
    generated: dict[str, pd.DataFrame] = {}
    delta = a["temp"].to_numpy() - o["temp"].to_numpy()
    for name, layers, title, summary, purpose in specs:
        frame = o.copy()
        mask = np.ones(len(frame), dtype=bool) if layers is None else frame["layer"].isin(layers).to_numpy()
        expected_values = o["temp"].to_numpy().copy()
        expected_values[mask] += P2_ALPHA * delta[mask]
        frame["temp"] = expected_values
        out = candidate_root / name / "P2_submission.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False, float_format="%.12f", lineterminator="\n")
        reread = pd.read_csv(out)
        assert_same_keys(o, reread, keys)
        if not np.isfinite(reread["temp"]).all():
            raise AssertionError(f"P2 non-finite output: {name}")
        max_formula_error = float(np.max(np.abs(reread["temp"].to_numpy() - expected_values)))
        if max_formula_error > 5.1e-13:
            raise AssertionError(f"P2 formula error: {name} {max_formula_error}")
        actual_hash = sha256(out)
        if actual_hash != expected_hashes[name]:
            raise AssertionError(
                f"P2 serialization hash mismatch: {name} expected={expected_hashes[name]} actual={actual_hash}"
            )
        generated[name] = reread
        records.append(
            {
                "problem": "P2",
                "name": name,
                "title": title,
                "one_line_summary": summary,
                "purpose": purpose,
                "path": str(out),
                "rows": int(len(reread)),
                "active_layers": "all" if layers is None else sorted(layers),
                "alpha": P2_ALPHA,
                "changed_rows_vs_O": int((np.abs(reread["temp"].to_numpy() - o["temp"].to_numpy()) > 5.1e-13).sum()),
                "temp_min": float(reread["temp"].min()),
                "temp_max": float(reread["temp"].max()),
                "max_formula_error": max_formula_error,
                "sha256": actual_hash,
            }
        )
    curve_a = 0.16414671028600014
    curve_b = 0.05219110288899986
    curve_c = 0.29277297722500006
    predicted_rmse = math.sqrt(curve_a * P2_ALPHA**2 + curve_b * P2_ALPHA + curve_c)
    analysis = {
        "status": "PASS",
        "formula": "P(alpha)=O+alpha*(A-O)",
        "alpha_star": P2_ALPHA,
        "official_rmse_points": {"alpha_0_O": 0.541085, "alpha_0.5_B": 0.599921, "alpha_1_A": 0.713520},
        "mse_quadratic": {"a": curve_a, "b": curve_b, "c": curve_c},
        "predicted_global_rmse": predicted_rmse,
        "predicted_improvement_vs_O": 0.541085 - predicted_rmse,
        "rounding_robust_predicted_rmse_interval": [0.537236416, 0.537239056],
        "official_interpretation": {
            "q_definition": "q = displayed_RMSE^2",
            "layer2_delta_mse": "q_L2-q_O",
            "layer4_delta_mse": "q_L4-q_O",
            "layer3_delta_mse": "q_GLOBAL-q_L2-q_L4+q_O",
            "guard": "If GLOBAL is outside the preregistered interval, stop layer interpretation and audit scoring/lineage.",
        },
        "local_p100_delta_rmse": {
            "GLOBAL": -0.004883,
            "L2_ONLY": -0.001121,
            "L4_ONLY": -0.000357,
            "INFERRED_L3": -0.003400,
        },
        "local_p100_day_bootstrap_ci90": {
            "GLOBAL": [-0.007395, -0.002311],
            "L2_ONLY": [-0.001459, -0.000768],
            "L4_ONLY": [-0.001795, 0.001121],
            "INFERRED_L3": [-0.004331, -0.002453],
        },
    }
    return records, analysis


def make_p3_candidates(candidate_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    keys = ["case_id", "station", "lead_h"]
    columns = keys + ["hs_pred"]
    o = pd.read_csv(P3_O)
    a = pd.read_csv(P3_A)
    b = pd.read_csv(P3_B)
    for frame in (o, a, b):
        if frame.columns.tolist() != columns or len(frame) != 1_200:
            raise AssertionError("P3 shape/schema mismatch")
        if not np.isfinite(frame["hs_pred"]).all():
            raise AssertionError("P3 non-finite source")
    assert_same_keys(o, a, keys)
    assert_same_keys(o, b, keys)

    specs = [
        (
            "P3_1_EXPLOIT_REVERSE_GLOBAL",
            None,
            "P3 O-A 역방향 전체리드 v1",
            "현 베스트 O에서 열화 모델 A의 방향을 α=-2로 반전해 모든 예측 리드에 적용했습니다.",
            "EXPLOIT_AND_MECHANISM_CHECK",
        ),
        (
            "P3_2_PROBE_LEAD12_ONLY",
            {12},
            "P3 O-A 역방향 12시간 단독 v1",
            "동일한 α=-2 역방향 보정을 12시간 리드에만 적용해 공개 MSE의 12시간 기여를 측정합니다.",
            "LEAD_MECHANISM_PROBE",
        ),
        (
            "P3_3_PROBE_LEAD18_24_ONLY",
            {18, 24},
            "P3 O-A 역방향 18·24시간 단독 v1",
            "동일한 α=-2 역방향 보정을 18·24시간 리드에만 적용해 장기리드 기여를 측정합니다.",
            "LEAD_MECHANISM_PROBE",
        ),
    ]
    records: list[dict[str, object]] = []
    delta = a["hs_pred"].to_numpy() - o["hs_pred"].to_numpy()
    for name, leads, title, summary, purpose in specs:
        frame = o.copy()
        mask = np.ones(len(frame), dtype=bool) if leads is None else frame["lead_h"].isin(leads).to_numpy()
        expected_values = o["hs_pred"].to_numpy().copy()
        expected_values[mask] += P3_ALPHA * delta[mask]
        frame["hs_pred"] = expected_values
        out = candidate_root / name / "P3_submission.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False, float_format="%.12f", lineterminator="\n")
        reread = pd.read_csv(out)
        assert_same_keys(o, reread, keys)
        values = reread["hs_pred"].to_numpy()
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 30:
            raise AssertionError(f"P3 physical/range guard failed: {name}")
        max_formula_error = float(np.max(np.abs(values - expected_values)))
        if max_formula_error > 5.1e-13:
            raise AssertionError(f"P3 formula error: {name} {max_formula_error}")
        records.append(
            {
                "problem": "P3",
                "name": name,
                "title": title,
                "one_line_summary": summary,
                "purpose": purpose,
                "path": str(out),
                "rows": int(len(reread)),
                "active_leads": "all" if leads is None else sorted(leads),
                "alpha": P3_ALPHA,
                "changed_rows_vs_O": int((np.abs(values - o["hs_pred"].to_numpy()) > 5.1e-13).sum()),
                "prediction_min": float(values.min()),
                "prediction_max": float(values.max()),
                "rms_change_vs_O": float(np.sqrt(np.mean((values - o["hs_pred"].to_numpy()) ** 2))),
                "max_abs_change_vs_O": float(np.max(np.abs(values - o["hs_pred"].to_numpy()))),
                "max_formula_error": max_formula_error,
                "sha256": sha256(out),
            }
        )
    long_mask = o["lead_h"].isin([12, 18, 24]).to_numpy()
    midpoint_error = b["hs_pred"].to_numpy() - (
        o["hs_pred"].to_numpy() + a["hs_pred"].to_numpy()
    ) / 2
    if float(np.max(np.abs(midpoint_error))) > 1e-12:
        raise AssertionError("P3 B is not the pinned exact midpoint of O/A")
    official_scores = np.array([0.607071, 0.609346, 0.611680], dtype=float)
    design = np.array([[0.0, 0.0, 1.0], [0.25, 0.5, 1.0], [1.0, 1.0, 1.0]])
    curve_a, curve_b, curve_c = np.linalg.solve(design, official_scores**2)
    predicted_global_rmse = math.sqrt(curve_a * P3_ALPHA**2 + curve_b * P3_ALPHA + curve_c)
    rounding_predictions = []
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        rounded = official_scores + np.asarray(signs) * 0.5e-6
        qa, qb, qc = np.linalg.solve(design, rounded**2)
        rounding_predictions.append(math.sqrt(qa * P3_ALPHA**2 + qb * P3_ALPHA + qc))
    analysis = {
        "status": "PASS",
        "formula": "P(alpha)=O+alpha*(A-O); alpha=-2",
        "official_rmse_history": {"O": 0.607071, "A": 0.611680, "B_long_midpoint": 0.609346},
        "official_interpretation": {
            "q_definition": "q = displayed_RMSE^2",
            "lead12_delta_mse": "q_L12-q_O",
            "lead18_24_delta_mse": "q_L18_24-q_O",
            "early_lead_delta_mse": "q_GLOBAL-q_L12-q_L18_24+q_O",
            "rounding_rule": "Propagate ±0.5e-6 RMSE display intervals before assigning a sign.",
        },
        "exact_axis_guard": "B=(O+A)/2 to <=1e-12 on every row",
        "mse_quadratic": {"a": float(curve_a), "b": float(curve_b), "c": float(curve_c)},
        "unconstrained_alpha_star": float(-curve_b / (2 * curve_a)),
        "chosen_alpha_reason": "Conservative bounded extrapolation; alpha=-2 is far inside the unconstrained public optimum and keeps predictions physical.",
        "predicted_global_rmse": predicted_global_rmse,
        "predicted_improvement_vs_O": float(official_scores[0] - predicted_global_rmse),
        "rounding_robust_predicted_rmse_interval": [min(rounding_predictions), max(rounding_predictions)],
        "prediction_confidence": "HIGH for the same fixed public scorer, not evidence of private-set generalization",
        "local_analogue_delta_rmse": {
            "GLOBAL": 0.00216615,
            "L12_ONLY": 0.00012925,
            "L18_24_ONLY": 0.00203724,
        },
        "local_official_sign_reversal_test": "Local analogue worsens; official improvement would be a public/local sign reversal, not private-generalization proof.",
        "b_midpoint_diagnostic": {
            "long_lead_max_abs_error": float(np.max(np.abs(midpoint_error[long_mask]))),
            "global_rms_off_axis": float(np.sqrt(np.mean(midpoint_error**2))),
            "global_max_abs_off_axis": float(np.max(np.abs(midpoint_error))),
        },
        "separation_guard": "Historical O/A/B submissions only; no ERA5 file, process, model, feature, gate, or active experiment was read or changed.",
    }
    return records, analysis


def candidate_note(record: dict[str, object], interpretation: str) -> str:
    return "\n".join(
        [
            f"제출물 제목: {record['title']}",
            f"한줄요약(접근방식): {record['one_line_summary']}",
            f"문제: {record['problem']}",
            f"역할: {record['purpose']}",
            f"사전등록 해석: {interpretation}",
            f"행 수: {record['rows']}",
            f"SHA-256: {record['sha256']}",
            "주의: 9개 파일을 모두 업로드하기 전에는 중간 점수를 열람하거나 설계를 변경하지 않습니다.",
        ]
    )


def main() -> None:
    if ARTIFACT_ROOT.exists() or DELIVERY_ROOT.exists():
        raise FileExistsError(
            f"Refusing to overwrite frozen output: artifact={ARTIFACT_ROOT.exists()} delivery={DELIVERY_ROOT.exists()}"
        )
    assert_pins()
    p1_analysis = validate_p1()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=False)
    DELIVERY_ROOT.mkdir(parents=True, exist_ok=False)
    candidate_root = ARTIFACT_ROOT / "candidates"

    p1_records: list[dict[str, object]] = []
    for name, source, _, title, summary, purpose in P1_SOURCES:
        out = candidate_root / name / "P1_submission.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, out)
        details = p1_analysis["candidates"][name]
        p1_records.append(
            {
                "problem": "P1",
                "name": name,
                "title": title,
                "one_line_summary": summary,
                "purpose": purpose,
                "path": str(out),
                **details,
            }
        )

    p2_records, p2_analysis = make_p2_candidates(candidate_root)
    p3_records, p3_analysis = make_p3_candidates(candidate_root)
    records = p1_records + p2_records + p3_records
    if len(records) != 9 or {p: sum(r["problem"] == p for r in records) for p in ("P1", "P2", "P3")} != {
        "P1": 3,
        "P2": 3,
        "P3": 3,
    }:
        raise AssertionError("candidate count is not exactly 3/3/3")

    interpretations = {
        "P1": "R은 개선 전이를 확인하고, I/U는 기존 O/B와 함께 2×2 양성집합 factorial을 완성합니다.",
        "P2": "표시 RMSE를 제곱해 q를 만든 뒤 L2·L4를 직접, L3를 qG-q2-q4+qO로 추론합니다.",
        "P3": "표시 RMSE를 제곱해 q를 만든 뒤 12h·18/24h를 직접, 단기리드를 qG-q12-q18/24+qO로 추론합니다.",
    }

    delivery_records: list[dict[str, object]] = []
    for record in records:
        name = str(record["name"])
        problem = str(record["problem"])
        source = Path(str(record["path"]))
        target_dir = DELIVERY_ROOT / name
        target_dir.mkdir(parents=True, exist_ok=False)
        target = target_dir / f"{problem}_submission.csv"
        shutil.copyfile(source, target)
        if sha256(target) != record["sha256"]:
            raise AssertionError(f"delivery copy hash mismatch: {name}")
        note_record = dict(record)
        note_record["path"] = str(target)
        write_text(target_dir / f"{problem}_제출정보.txt", candidate_note(note_record, interpretations[problem]))
        delivery_records.append({**note_record, "path": str(target)})

    backup_dir = DELIVERY_ROOT / "backup_best_before_round_D"
    backup_dir.mkdir(parents=True, exist_ok=False)
    backups = []
    for problem, source in (("P1", P1_BEST), ("P2", P2_O), ("P3", P3_O)):
        target = backup_dir / f"{problem}_submission.csv"
        shutil.copyfile(source, target)
        backups.append(
            {
                "problem": problem,
                "source": str(source),
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )
    write_text(
        backup_dir / "README.txt",
        "Round D 제출 직전의 문제별 공식 최선 파일 백업입니다. 새 결과가 확정되기 전에는 덮어쓰지 않습니다.",
    )

    analysis = {
        "schema_version": "preregistered_3x3_evidence_analysis_20260826.v1",
        "research_question": "How should nine expiring daily submissions maximize score potential and durable causal evidence?",
        "decision": "Use a frozen blind 3x3 batch: one exploit/confirmation plus orthogonal or factorial mechanism probes per problem.",
        "p1": p1_analysis,
        "p2": p2_analysis,
        "p3": p3_analysis,
        "anti_adaptation_protocol": {
            "freeze_before_first_upload": True,
            "inspect_intermediate_scores": False,
            "same_day_result_driven_replacement": False,
            "official_submissions_performed_by_builder": 0,
        },
    }
    write_json(ARTIFACT_ROOT / "analysis.json", analysis)

    manifest = {
        "schema_version": "preregistered_submission_bundle_20260826.v2",
        "status": "FILES_FROZEN_AWAITING_INDEPENDENT_QA_AND_USER_CONFIRMATION",
        "created_kst": "2026-08-26",
        "team": "분당독고다이",
        "official_submissions_performed": 0,
        "daily_limits_confirmed_before_freeze": {"P1_remaining": 3, "P2_remaining": 3, "P3_remaining": 3},
        "submission_plan": {"P1": 3, "P2": 3, "P3": 3},
        "blind_batch_protocol": "Upload all nine frozen hashes without viewing intermediate scores; then record all results.",
        "candidates": delivery_records,
        "backup_best_before_round_D": backups,
        "input_lineage": {str(path): digest for path, digest in PINNED_INPUTS.items()},
        "analysis_path": str(ARTIFACT_ROOT / "analysis.json"),
        "era5_separation": "This bundle uses historical P3 submissions only and does not modify or replace the active fixed ERA5 experiment.",
    }
    write_json(DELIVERY_ROOT / "SET_MANIFEST.json", manifest)

    order_lines = [
        "2026-08-26 Round D 사전등록 3×3 제출 계획",
        "",
        "결론: P1/P2/P3 각각 정확히 3개, 총 9개가 동결되었습니다.",
        "공식 확인 당시 잔여 횟수: P1 3/3, P2 3/3, P3 3/3.",
        "실제 업로드: 0회. 사용자 action-time 승인 전에는 업로드하지 않습니다.",
        "",
        "강제 순서:",
    ]
    for index, record in enumerate(delivery_records, start=1):
        order_lines.append(f"{index}. {record['name']} | {record['title']} | {record['sha256']}")
    order_lines.extend(
        [
            "",
            "블라인드 규칙:",
            "- 첫 업로드 전에 9개 파일·해시·해석식을 모두 동결합니다.",
            "- 9개 업로드가 끝날 때까지 중간 점수를 열람하지 않습니다.",
            "- 오늘 결과를 보고 같은 날 후보를 교체·튜닝하지 않습니다.",
            "- 업로드 후 submission ID, 시각, 표시 점수, 파일 SHA-256을 기록합니다.",
            "",
            "해석 핵심:",
            "- P1: 기존 O/B와 I/U로 양성집합 2×2 factorial, Router는 별도 전이 확인.",
            "- P2: RMSE²에서 layer 2·4 직접 측정, layer 3 차감 추론.",
            "- P3: RMSE²에서 lead 12·18/24 직접 측정, 단기 lead 차감 추론.",
        ]
    )
    write_text(DELIVERY_ROOT / "READY_SUBMISSION_PLAN.txt", "\n".join(order_lines))

    hash_lines = []
    for record in delivery_records:
        path = Path(str(record["path"]))
        hash_lines.append(f"{sha256(path)}  {path.relative_to(DELIVERY_ROOT).as_posix()}")
    for backup in backups:
        path = Path(str(backup["path"]))
        hash_lines.append(f"{sha256(path)}  {path.relative_to(DELIVERY_ROOT).as_posix()}")
    write_text(DELIVERY_ROOT / "SHA256SUMS.txt", "\n".join(hash_lines))

    build_receipt = {
        "status": "PASS",
        "artifact_root": str(ARTIFACT_ROOT),
        "delivery_root": str(DELIVERY_ROOT),
        "candidate_counts": {p: sum(r["problem"] == p for r in records) for p in ("P1", "P2", "P3")},
        "candidate_hashes": {r["name"]: r["sha256"] for r in delivery_records},
        "no_upload_performed": True,
    }
    write_json(ARTIFACT_ROOT / "build_receipt.json", build_receipt)
    print(json.dumps(build_receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
