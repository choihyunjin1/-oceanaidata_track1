"""Build a compact, information-rich final daily submission probe set.

This does not upload anything.  It reuses frozen official submissions and
sealed local models, writes new CSVs, and records exact hashes/differences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p2_restore.data import KEYS, load_p2_data
from p2_restore.p2_alpha40_quasiperiodic_gp_residual_20260828_v1 import bounded_profile_correction
from p2_restore.profile_projection import project_profiles_vectorized, public_endpoint_frame
from p2_restore.submission import build_submission as build_p2_submission
from p2_restore.submission import validate_submission as validate_p2_submission
from p2_restore.supervised_rank1_functional_residual import (
    TARGET_LAYERS,
    SupervisedRank1Residual,
    build_public_functional_features,
)

@dataclass(frozen=True)
class ProbePaths:
    output_dir: Path
    p1_set: Path
    p2_data: Path
    p2_base: Path
    p2_anchor: Path
    p3_current: Path
    p3_old: Path
    p3_kma: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission-archive",
        type=Path,
        required=True,
        help="Directory containing the frozen prior submission packages.",
    )
    parser.add_argument(
        "--p2-data-dir",
        type=Path,
        required=True,
        help="Directory containing the official P2 input files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New package directory; defaults below --submission-archive.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> ProbePaths:
    archive = args.submission_archive.expanduser().resolve()
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else archive / "20260828_DEADLINE_INFORMATION_PROBES_READY"
    )
    return ProbePaths(
        output_dir=output,
        p1_set=archive / "20260827_round_F_mstcn_e150_P1x3",
        p2_data=args.p2_data_dir.expanduser().resolve(),
        p2_base=archive
        / "20260828_P2_SEASONAL_OAS_TS50_PROJECTED_READY"
        / "P2_submission.csv",
        p2_anchor=ROOT / "artifacts" / "p2_state_conditional_lean_v1" / "oof.parquet",
        p3_current=archive
        / "20260827_P3_REFINED_PUBLIC_OPTIMUM_READY"
        / "P3_submission.csv",
        p3_old=ROOT / "submissions" / "p3_long_persistence_shrink" / "submission.csv",
        p3_kma=ROOT
        / "submissions"
        / "p3_kma_calibrated_longlead_secondary_v1"
        / "submission.csv",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, directory: Path, filename: str, title: str, summary: str) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / filename
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    note = directory / "제출정보.txt"
    note.write_text(
        f"제출물 제목: {title}\n한줄요약(접근방식): {summary}\n파일 SHA-256: {sha256(path)}\n",
        encoding="utf-8-sig",
    )
    return {"path": str(path), "rows": len(frame), "sha256": sha256(path), "title": title, "summary": summary}


def build_p1(paths: ProbePaths) -> list[dict]:
    anchor_path = paths.p1_set / "P1_1_MSTCN_E150_ROUTER_UNION_ALL" / "P1_submission.csv"
    gi_path = paths.p1_set / "P1_3_EXPLOIT_GI_NO_REMOVALS" / "P1_submission.csv"
    anchor = pd.read_csv(anchor_path, dtype={"station": "string", "time": "string"})
    gi = pd.read_csv(gi_path, dtype={"station": "string", "time": "string"})
    keys = ["station", "year", "layer", "time"]
    if not anchor[keys].equals(gi[keys]) or list(anchor.columns) != keys + ["label"]:
        raise RuntimeError("P1 frozen key/schema mismatch")
    novel = anchor["label"].eq(0) & gi["label"].eq(1)
    if int(novel.sum()) != 6:
        raise RuntimeError("P1 novel GI row count drifted")
    spike = novel & gi["anomaly_type"].eq("spike")
    s_only = novel & gi["station"].eq("S-ORS")
    specs = [
        ("P1_1_E150_PLUS_GI_SPIKE2", spike, "P1 e150 + GI 고신뢰 spike 2행", "공식 최고 e150을 유지하고 GI가 spike 유형까지 명시한 미검출 2행만 추가합니다."),
        ("P1_2_E150_PLUS_GI_S5", s_only, "P1 e150 + S-ORS GI 5행", "공식 최고 e150에 GI가 추가 검출한 S-ORS 5행만 더해 station 선택 효과를 측정합니다."),
        ("P1_3_E150_PLUS_GI_ALL6", novel, "P1 e150 + GI 신규 6행", "공식 최고 e150과 GI 후보의 합집합으로 기존 최고가 놓친 6행의 순효과를 측정합니다."),
    ]
    output = []
    for name, mask, title, summary in specs:
        candidate = anchor.copy()
        candidate.loc[mask, "label"] = 1
        record = write_csv(candidate, paths.output_dir / name, "P1_submission.csv", title, summary)
        record.update(problem="P1", changed_rows_vs_best=int((candidate.label != anchor.label).sum()))
        output.append(record)
    return output


def build_p3(paths: ProbePaths) -> list[dict]:
    current = pd.read_csv(paths.p3_current)
    old = pd.read_csv(paths.p3_old)
    kma = pd.read_csv(paths.p3_kma)
    key_columns = [column for column in current.columns if column not in {"significant_wave_height", "hs", "prediction"}]
    value_column = [column for column in current.columns if column not in key_columns]
    if len(value_column) != 1:
        value_column = [current.columns[-1]]
        key_columns = list(current.columns[:-1])
    value_column = value_column[0]
    if list(current.columns) != list(old.columns) or list(current.columns) != list(kma.columns):
        raise RuntimeError("P3 schema mismatch")
    if not current[key_columns].equals(old[key_columns]) or not current[key_columns].equals(kma[key_columns]):
        raise RuntimeError("P3 key/order mismatch")
    delta = kma[value_column].to_numpy(float) - old[value_column].to_numpy(float)
    output = []
    for fraction, suffix in ((0.5, "KMA_ALPHA20"), (1.0, "KMA_ALPHA40")):
        candidate = current.copy()
        candidate[value_column] = np.clip(current[value_column].to_numpy(float) + fraction * delta, 0.0, 30.0)
        title = f"P3 공개최고 + KMA 18/24h 보정 {int(40*fraction)}%"
        summary = f"공식 최고 long-axis 예측을 유지하면서 독립 KMA 장기파고 보정축을 {int(40*fraction)}%만 18·24h에 결합합니다."
        record = write_csv(candidate, paths.output_dir / f"P3_{1 if fraction == 0.5 else 2}_{suffix}", "P3_submission.csv", title, summary)
        change = candidate[value_column].to_numpy(float) - current[value_column].to_numpy(float)
        record.update(problem="P3", changed_rows_vs_best=int(np.sum(np.abs(change) > 1e-12)), rms_change_vs_best=float(np.sqrt(np.mean(change**2))), min=float(candidate[value_column].min()), max=float(candidate[value_column].max()))
        output.append(record)
    return output


def build_p2(paths: ProbePaths) -> list[dict]:
    data = load_p2_data(paths.p2_data)
    observations = data.observations.copy()
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    features = build_public_functional_features(observations, ridge=0.001, change_hours=(6, 24, 72))
    anchor = pd.read_parquet(paths.p2_anchor, columns=["time", "layer", "truth", "current_blend50"])
    anchor["time"] = pd.to_datetime(anchor["time"], utc=True)
    anchor["residual"] = anchor["truth"].to_numpy(float) - anchor["current_blend50"].to_numpy(float)
    pivot = anchor.pivot(index="time", columns="layer", values="residual").reindex(columns=TARGET_LAYERS)
    pivot = pivot.dropna()
    train_features = features.loc[pivot.index].reset_index(drop=True)
    valid = train_features["public_profile_valid"].to_numpy(bool)
    model = SupervisedRank1Residual.fit(train_features.loc[valid].reset_index(drop=True), pivot.to_numpy(float)[valid], pd.DatetimeIndex(pivot.index[valid]))

    test = data.test_index.copy()
    test_times = pd.to_datetime(test["time"], utc=True)
    query_times = pd.DatetimeIndex(test_times.drop_duplicates().sort_values())
    query_features = features.loc[query_times].reset_index(drop=True)
    profile_prediction, profile_enabled, _ = model.predict(query_features, query_times)
    local = query_times.tz_convert("Asia/Seoul")
    season_bins = ((local.dayofyear.to_numpy() - 1) // 14).astype(int)
    profile_enabled &= np.isin(season_bins, [17, 18])
    time_to_row = {time: row for row, time in enumerate(query_times)}
    layer_to_column = {layer: column for column, layer in enumerate(TARGET_LAYERS)}
    raw = np.zeros(len(test), dtype=float)
    enabled = np.zeros(len(test), dtype=bool)
    for row, (time, layer) in enumerate(zip(test_times, test["layer"].astype(int), strict=True)):
        profile_row = time_to_row[pd.Timestamp(time)]
        raw[row] = profile_prediction[profile_row, layer_to_column[layer]]
        enabled[row] = profile_enabled[profile_row]
    correction, cap = bounded_profile_correction(raw, enabled, rms_cap=0.05, p99_cap=0.20)
    base = pd.read_csv(paths.p2_base, dtype={"station": "string", "time": "string"})
    if list(base.columns) != KEYS + ["temp"] or not base[KEYS].equals(test[KEYS]):
        raise RuntimeError("P2 alpha50 base key/schema mismatch")
    endpoints = public_endpoint_frame(observations)
    final = project_profiles_vectorized(test, base["temp"].to_numpy(float) + correction, endpoints).prediction
    candidate = build_p2_submission(test, final)
    title = "P2 alpha50 + 교차검증 계절 veto rank-1"
    summary = "공식 최고 alpha50에 과거 3창에서 일관 개선된 supervised rank-1 보정만 계절 bin 17·18에 제한 적용합니다."
    record = write_csv(candidate, paths.output_dir / "P2_1_ALPHA50_CROSSFIT_VETO_RANK1", "P2_submission.csv", title, summary)
    validation = validate_p2_submission(Path(record["path"]), test)
    change = final - base["temp"].to_numpy(float)
    record.update(problem="P2", changed_rows_vs_best=int(np.sum(np.abs(change) > 1e-12)), rms_change_vs_best=float(np.sqrt(np.mean(change**2))), p99_abs_change=float(np.quantile(np.abs(change), 0.99)), cap=cap, validation=validation, enabled_profile_share=float(profile_enabled.mean()))
    return [record]


def main() -> None:
    paths = resolve_paths(parse_args())
    if paths.output_dir.exists():
        raise FileExistsError(paths.output_dir)
    paths.output_dir.mkdir(parents=True)
    records = build_p1(paths) + build_p2(paths) + build_p3(paths)
    hashes = [record["sha256"] for record in records]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("candidate hash collision")
    manifest = {"status": "READY_NOT_UPLOADED", "created_at_kst": pd.Timestamp.now(tz="Asia/Seoul").isoformat(), "records": records}
    (paths.output_dir / "SET_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
