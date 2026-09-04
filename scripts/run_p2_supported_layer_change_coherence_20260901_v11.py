"""Exactly-once P2 supported-public-layer change-coherence sensitivity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_public_sensor_influence_shrink_20260901_v10 as engine  # noqa: E402

EXPERIMENT_ID = "p2_supported_layer_change_coherence_20260901_v11"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RUNNER = Path(__file__)

engine.EXPERIMENT_ID = EXPERIMENT_ID
engine.CONFIG = CONFIG
engine.ARTIFACT = ARTIFACT
engine.REPORT = REPORT
engine.RUNNER = RUNNER
engine.SCHEMA_VERSION = "p2.supported_layer_change_coherence.result.20260901.v11"


def build_supported_layer_coherence(
    observations: pd.DataFrame, blind: pd.DataFrame, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute a label-free isolated-layer change-coherence score."""

    contract = config["training_only_influence"]
    layers = tuple(int(value) for value in contract["public_layers"])
    if layers != (1, 5, 6, 7):
        raise engine.ContractError("supported public-layer set drift")
    public = observations.loc[
        observations["layer"].isin(layers), ["station", "time", "layer", "temp"]
    ].copy()
    public = public.sort_values(["station", "layer", "time"], kind="stable")
    grouped = public.groupby(["station", "layer"], sort=False, observed=True)
    previous_time = grouped["time"].shift(1)
    elapsed_seconds = (public["time"] - previous_time).dt.total_seconds()
    public["temp_diff"] = grouped["temp"].diff().where(elapsed_seconds.eq(600.0))
    train_mask = engine._training_window_mask(
        public["time"], contract["registered_windows_kst"]
    )

    receipts: dict[str, Any] = {}
    for layer in layers:
        values = public.loc[
            train_mask & public["layer"].eq(layer), "temp_diff"
        ].dropna()
        expected = int(contract["support_receipt_exact10min_counts"][str(layer)])
        if len(values) != expected:
            raise engine.ContractError(
                f"supported layer {layer} count drift: {len(values)} != {expected}"
            )
        median = float(values.median())
        mad = float((values - median).abs().median())
        scale = max(1.4826 * mad, 0.01)
        mask = public["layer"].eq(layer) & public["temp_diff"].notna()
        public.loc[mask, "standardized_signed_change"] = (
            public.loc[mask, "temp_diff"] - median
        ) / scale
        receipts[str(layer)] = {
            "training_exact10min_differences": int(len(values)),
            "median_signed_difference_C": median,
            "mad_C": mad,
            "robust_scale_C": scale,
        }

    wide = public.pivot(
        index=["station", "time"], columns="layer", values="standardized_signed_change"
    ).reindex(columns=list(layers))
    key = pd.MultiIndex.from_frame(blind[["station", "time"]])
    aligned = wide.reindex(key)
    available = aligned.notna().sum(axis=1).to_numpy(int)
    cross_layer_median = aligned.median(axis=1, skipna=True)
    deviation = aligned.sub(cross_layer_median, axis=0).abs()
    score = deviation.max(axis=1, skipna=True).to_numpy(float)
    score[available < int(contract["minimum_available_public_layers"])] = np.nan
    cutoff = float(contract["huber_cutoff_coherence_score"])
    floor = float(contract["minimum_influence_weight"])
    weight = np.ones(len(blind), dtype=float)
    active = np.isfinite(score) & (score > cutoff)
    weight[active] = np.maximum(floor, cutoff / score[active])
    if not (np.isfinite(weight).all() and np.all((weight >= floor) & (weight <= 1.0))):
        raise engine.ContractError("coherence influence weights violate bounds")
    return score, weight, {
        "signal": contract["signal"],
        "public_layers": list(layers),
        "minimum_available_public_layers": int(contract["minimum_available_public_layers"]),
        "per_layer_training_stats": receipts,
        "query_rows": int(len(blind)),
        "query_rows_minimum_layers_available": int(np.count_nonzero(np.isfinite(score))),
        "active_rows": int(active.sum()),
        "active_share": float(active.mean()),
        "weight_min": float(weight.min()),
        "weight_mean": float(weight.mean()),
        "rows_deleted": 0,
        "target_truth_used": False,
    }


def preflight() -> dict[str, Any]:
    config = engine.load_config()
    audit = engine.semantic_audit(config)
    if config["candidate"]["name"] != "P2_V11_HUBER6_SUPPORTED_LAYER_CHANGE_COHERENCE":
        raise engine.ContractError("candidate drift")
    contract = config["training_only_influence"]
    if contract["public_layers"] != [1, 5, 6, 7]:
        raise engine.ContractError("supported layer drift")
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "ZERO_OPERATION_PREFLIGHT_PASS",
        "candidate": config["candidate"]["name"],
        "semantic_fingerprint": config["semantic_fingerprint"],
        "semantic_audit_sha256": engine.sha256_json(audit),
        "supported_public_layers": contract["public_layers"],
        "minimum_available_public_layers": contract["minimum_available_public_layers"],
        "config_sha256": engine.sha256_file(CONFIG),
        "runner_sha256": engine.sha256_file(RUNNER),
        "data_rows_read": 0,
        "model_fits": 0,
        "artifacts_written": 0,
        "official_rows_read": 0,
        "hidden_rows_read": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }
    payload["preflight_sha256"] = engine.sha256_json(payload)
    return payload


def write_report(result: dict[str, Any]) -> None:
    item = result["candidate"]
    REPORT.mkdir(parents=True, exist_ok=True)
    fold_text = ", ".join(
        f"{key} {value['delta_rmse']:+.9f}" for key, value in item["by_fold"].items()
    )
    layer_text = ", ".join(
        f"L{key} {value['delta_rmse']:+.9f}" for key, value in item["by_layer"].items()
    )
    month_text = ", ".join(
        f"{key} {value['delta_rmse']:+.9f}" for key, value in item["by_month"].items()
    )
    lines = [
        "# P2 supported-layer change-coherence exploratory cycle 20260901 v11",
        "",
        "## 결론",
        "",
        f"상태: `{result['status']}`. pooled ΔRMSE {item['delta_rmse']:+.9f}°C, 명목 {item['nominal_expected_points_delta']:+.6f}점, transport-adjusted {item['transport_adjusted_points_delta']:+.6f}점이다.",
        f"fold: {fold_text}. layer: {layer_text}.",
        f"month: {month_text}.",
        f"active {item['active_rows']:,}/{result['rows']:,}행, active-slice ΔRMSE {item['worst_and_identity_slices']['active_influence']['delta_rmse']}.",
        "세 historical fold는 모두 exposed exploratory surface이며 fresh confirmation이 아니다.",
        "",
        "## 구조와 중복 배제",
        "",
        "훈련 support가 확인된 L1/L5/L6/L7만 사전 고정했다. 각 층의 exact-10-minute signed change를 training-only median/MAD로 표준화하고, 동시간 cross-layer median에서 한 층만 고립 이탈한 최대값을 coherence score로 썼다.",
        "최소 3개 층이 없거나 score<=6이면 bit-exact champion이다. active 행도 endpoint baseline 대비 champion correction을 최소 50% 보존한다. 행 삭제, fit, learned gate, threshold search, v10 metric tuning은 모두 0이다.",
        "",
        "## v10·v7 상태",
        "",
        "v10은 L8 training support 0건으로 0-fit technical INVALID이며 같은 ID를 재실행하지 않았다.",
        f"v7 ready pack은 미접촉·미업로드 상태를 유지했다: `{result['v7_readiness']['status']}`, report-calibrated +{result['v7_readiness']['transport_calibrated_expected_points_delta']:.6f}점. 챔피언 보존이 기본이다.",
        "",
        "## 접근 경계",
        "",
        "official/test/sample/baseline/score/query support/hidden/submission CSV/upload 접근은 모두 0이다.",
    ]
    (REPORT / "report-source.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


engine.build_public_influence = build_supported_layer_coherence
engine.write_report = write_report


def run() -> dict[str, Any]:
    return engine.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.preflight == args.execute:
        raise SystemExit("Choose exactly one of --preflight or --execute")
    value = preflight() if args.preflight else run()
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
