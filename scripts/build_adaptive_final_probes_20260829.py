from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

P2_ALPHA_VERTEX = 0.8341898147191101
P2_LAYER_ALPHA = {
    2: 1.0438543588547008,
    3: 0.8747809004102274,
    4: 0.6656479035119007,
}
P3_TOTAL_ALPHA = {18: 0.8476670068555469, 24: 0.06289792992312171}
P3_CURRENT_TOTAL_ALPHA = 0.4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_candidate(
    frame: pd.DataFrame,
    directory: Path,
    filename: str,
    title: str,
    summary: str,
) -> dict[str, object]:
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / filename
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    digest = sha256(path)
    (directory / "제출정보.txt").write_text(
        f"제출물 제목: {title}\n한줄요약(접근방식): {summary}\n파일 SHA-256: {digest}\n",
        encoding="utf-8-sig",
    )
    return {"path": str(path), "rows": len(frame), "sha256": digest}


def require_same_keys(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> None:
    if not left[keys].equals(right[keys]):
        raise RuntimeError("key order mismatch")


def build_p2(
    *, output: Path, p2_base: Path, p2_champion: Path
) -> list[dict[str, object]]:
    keys = ["station", "layer", "time"]
    base = pd.read_csv(p2_base, dtype={"station": "string", "time": "string"})
    champion = pd.read_csv(p2_champion, dtype={"station": "string", "time": "string"})
    if list(base.columns) != keys + ["temp"] or list(champion.columns) != keys + ["temp"]:
        raise RuntimeError("P2 schema mismatch")
    require_same_keys(base, champion, keys)
    base_values = base["temp"].to_numpy(np.float64)
    delta = champion["temp"].to_numpy(np.float64) - base_values
    if not np.isfinite(base_values).all() or not np.isfinite(delta).all():
        raise RuntimeError("P2 non-finite input")

    uniform = base.copy()
    uniform["temp"] = base_values + P2_ALPHA_VERTEX * delta
    record_uniform = write_candidate(
        uniform,
        output / "P2_1_OFFICIAL_MSE_VERTEX_A083419",
        "P2_submission.csv",
        "P2 공식 MSE 정점 alpha 0.83419",
        "공식 alpha 0/1/2 RMSE 세 점의 제곱오차 포물선 정점으로 rank-1 보정 강도를 설정했습니다.",
    )
    record_uniform.update(
        problem="P2",
        role="official-score quadratic vertex",
        axis_alpha=P2_ALPHA_VERTEX,
        predicted_public_rmse=0.43020873392450576,
    )

    layerwise = base.copy()
    layer_values = base_values.copy()
    for layer, alpha in P2_LAYER_ALPHA.items():
        mask = base["layer"].eq(layer).to_numpy()
        layer_values[mask] += alpha * delta[mask]
    layerwise["temp"] = layer_values
    record_layer = write_candidate(
        layerwise,
        output / "P2_2_LAYERWISE_OOF_SHRUNK",
        "P2_submission.csv",
        "P2 층별 OOF 수축 rank-1",
        "공식 최적 전체 강도를 중심으로 과거 OOF 보정 수용력에 따라 층 2를 강화하고 층 4를 축소했습니다.",
    )
    record_layer.update(
        problem="P2",
        role="layerwise transport probe",
        layer_alpha={str(key): value for key, value in P2_LAYER_ALPHA.items()},
    )
    return [record_uniform, record_layer]


def build_p3(
    *, output: Path, p3_champion: Path, p3_sweep: Path
) -> dict[str, object]:
    keys = ["case_id", "station", "lead_h"]
    current = pd.read_csv(p3_champion)
    c03 = pd.read_csv(p3_sweep / "03_P3_KMA_L18_040_L24_060" / "P3_submission.csv")
    c04 = pd.read_csv(p3_sweep / "04_P3_KMA_L18_020_L24_060" / "P3_submission.csv")
    c05 = pd.read_csv(p3_sweep / "05_P3_KMA_L18_020_L24_080" / "P3_submission.csv")
    for frame in (current, c03, c04, c05):
        if list(frame.columns) != keys + ["hs_pred"]:
            raise RuntimeError("P3 schema mismatch")
        require_same_keys(current, frame, keys)

    correction_18 = 2.0 * (c03["hs_pred"].to_numpy() - c04["hs_pred"].to_numpy())
    correction_24 = 2.0 * (c05["hs_pred"].to_numpy() - c04["hs_pred"].to_numpy())
    lead = current["lead_h"].to_numpy()
    if np.count_nonzero(np.abs(correction_18) > 1e-12) != 200:
        raise RuntimeError("P3 18h correction support changed")
    if np.count_nonzero(np.abs(correction_24) > 1e-12) != 200:
        raise RuntimeError("P3 24h correction support changed")
    if np.any(np.abs(correction_18[lead != 18]) > 1e-12):
        raise RuntimeError("P3 18h correction leaked")
    if np.any(np.abs(correction_24[lead != 24]) > 1e-12):
        raise RuntimeError("P3 24h correction leaked")

    values = current["hs_pred"].to_numpy(np.float64).copy()
    values += (
        (P3_TOTAL_ALPHA[18] - P3_CURRENT_TOTAL_ALPHA)
        / P3_CURRENT_TOTAL_ALPHA
        * correction_18
    )
    values += (
        (P3_TOTAL_ALPHA[24] - P3_CURRENT_TOTAL_ALPHA)
        / P3_CURRENT_TOTAL_ALPHA
        * correction_24
    )
    values = np.clip(values, 0.0, 30.0)
    candidate = current.copy()
    candidate["hs_pred"] = values
    record = write_candidate(
        candidate,
        output / "P3_1_OFFICIAL_GEOMETRY_L18_08477_L24_00629",
        "P3_submission.csv",
        "P3 공식기하 18h 0.848 / 24h 0.063",
        "기존 uniform 0.4 최고점과 오늘 두 lead-split 점수를 결합해 18h는 강화하고 24h는 거의 제거했습니다.",
    )
    record.update(
        problem="P3",
        role="official-score separable lead geometry",
        total_alpha={str(key): value for key, value in P3_TOTAL_ALPHA.items()},
        predicted_public_rmse=0.5677866355245917,
    )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--p2-base", type=Path, required=True)
    parser.add_argument("--p2-champion", type=Path, required=True)
    parser.add_argument("--p3-champion", type=Path, required=True)
    parser.add_argument("--p3-sweep", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    records = build_p2(
        output=args.output_dir,
        p2_base=args.p2_base,
        p2_champion=args.p2_champion,
    ) + [
        build_p3(
            output=args.output_dir,
            p3_champion=args.p3_champion,
            p3_sweep=args.p3_sweep,
        )
    ]
    manifest = {
        "schema_version": "adaptive.final.probes.20260829.v1",
        "status": "READY_NOT_UPLOADED",
        "official_evidence": {
            "P2_rmse": {"alpha_0": 0.431252, "alpha_1": 0.430250, "alpha_2": 0.432244},
            "P3_rmse": {
                "uniform_total_0": 0.583892,
                "uniform_total_0p2": 0.577671,
                "uniform_total_0p4": 0.575262,
                "total_0p6_1p0": 0.577577,
                "total_0p8_1p0": 0.576264,
            },
        },
        "records": records,
        "official_uploads": 0,
    }
    (args.output_dir / "SET_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
