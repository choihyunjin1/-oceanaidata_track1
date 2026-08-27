"""Build the preregistered P2 seasonal OAS 10% profile submission.

This deployment runner is deterministic. It reads the official P2 observations,
test index, sample schema, and the SHA-pinned current public-best P2 U file. It
never reads an answer file. Hidden-layer labels inside the official gap are
explicitly excluded from every fit.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import OAS

from p2_restore.data import KEYS, load_p2_data
from p2_restore.profile_projection import (
    project_profiles_vectorized,
    public_endpoint_frame,
)
from p2_restore.submission import build_submission, validate_submission


REPO = Path(r"C:\Users\cedis\PycharmProjects\PythonProject")
DATA = Path(r"C:\Users\cedis\Downloads\p2\데이터셋_P2\P2_profile_restore")
BASE = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260827_round_G_P2x3_P3x3_PUBLIC_QUADRATIC_READY"
    r"\P2_1_EXPLOIT_LAYERWISE_QUADRATIC\P2_submission.csv"
)
BASE_SHA256 = "13181dff0e749a1ea6dac7327b4ea34b8a7efd57a2f57170ba0d206f919cf592"
DEPLOY_TAG = os.environ.get("P2_OAS_DEPLOY_TAG", "p2_seasonal_oas_submission_20260827_v1")
OUT_DIR = REPO / "artifacts" / DEPLOY_TAG
OUTPUT = OUT_DIR / "P2_submission.csv"
RECEIPT = OUT_DIR / "receipt.json"
READY_DIR = Path(
    os.environ.get(
        "P2_OAS_READY_DIR",
        r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
        r"\20260827_P2_SEASONAL_OAS_TS10_PROJECTED_READY",
    )
)
READY_OUTPUT = READY_DIR / "P2_submission.csv"
READY_NOTE = READY_DIR / "제출정보.txt"

PUBLIC = (1, 5, 6, 7)
TARGET = (2, 3, 4)
ALPHA = float(os.environ.get("P2_OAS_ALPHA", "0.10"))
SEASON_BIN_DAYS = 14
SEASON_WINDOW_DAYS = 60.0
GAP_START = pd.Timestamp("2025-09-01", tz="Asia/Seoul").tz_convert("UTC")
GAP_STOP = pd.Timestamp("2025-11-01", tz="Asia/Seoul").tz_convert("UTC")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_panel(observations: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    obs = observations.copy()
    obs["time"] = pd.to_datetime(obs["time"], utc=True)
    temp = obs.pivot(index="time", columns="layer", values="temp").sort_index()
    psal = obs.pivot(index="time", columns="layer", values="psal").sort_index()
    panel = pd.DataFrame(index=temp.index)
    x_columns: list[str] = []
    y_columns: list[str] = []
    for layer in PUBLIC:
        for name, values in (("temp", temp), ("psal", psal)):
            column = f"{name}_{layer}"
            panel[column] = values[layer]
            x_columns.append(column)
    local = panel.index.tz_convert("Asia/Seoul")
    minute = local.hour.to_numpy() * 60 + local.minute.to_numpy()
    doy = local.dayofyear.to_numpy() + minute / 1440.0
    for harmonic in (1, 2, 3, 4):
        for kind, fn in (("sin", np.sin), ("cos", np.cos)):
            column = f"doy_{kind}_{harmonic}"
            panel[column] = fn(2 * np.pi * harmonic * doy / 365.2425)
            x_columns.append(column)
    for layer in TARGET:
        for name, values in (("temp", temp), ("psal", psal)):
            column = f"{name}_{layer}"
            panel[column] = values[layer]
            y_columns.append(column)
    return panel, x_columns, y_columns


def conditional_predict(
    panel: pd.DataFrame,
    query_index: pd.DatetimeIndex,
    x_columns: list[str],
    y_columns: list[str],
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    evaluate = panel.loc[query_index, x_columns].copy()
    nx = len(x_columns)
    x = evaluate.to_numpy(float)
    patterns = np.isfinite(x)
    yhat_z = np.full((len(evaluate), len(y_columns)), np.nan, dtype=float)
    yhat = np.full_like(yhat_z, np.nan)
    query_local = evaluate.index.tz_convert("Asia/Seoul")
    season_bins = ((query_local.dayofyear.to_numpy() - 1) // SEASON_BIN_DAYS).astype(int)
    outside_gap = (panel.index < GAP_START) | (panel.index >= GAP_STOP)
    train_index = panel.index[outside_gap]
    train_doy = train_index.tz_convert("Asia/Seoul").dayofyear.to_numpy(float)
    receipts: list[dict[str, float | int]] = []

    for season_bin in np.unique(season_bins):
        center = float(season_bin * SEASON_BIN_DAYS + 7.5)
        distance = np.abs(train_doy - center)
        distance = np.minimum(distance, 365.2425 - distance)
        train = panel.loc[
            train_index[distance <= SEASON_WINDOW_DAYS], x_columns + y_columns
        ].dropna()
        values = train.to_numpy(float)
        if len(values) < 100:
            raise RuntimeError(f"insufficient complete seasonal rows for bin {season_bin}")
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale == 0] = 1.0
        estimator = OAS(store_precision=False, assume_centered=False).fit(
            (values - mean) / scale
        )
        covariance = estimator.covariance_
        sigma_xx = covariance[:nx, :nx]
        sigma_yx = covariance[nx:, :nx]
        bin_rows = np.flatnonzero(season_bins == season_bin)
        for pattern in np.unique(patterns[bin_rows], axis=0):
            row_ids = bin_rows[np.all(patterns[bin_rows] == pattern, axis=1)]
            observed = np.flatnonzero(pattern)
            if len(observed) == 0:
                yhat_z[row_ids] = 0.0
            else:
                conditional = sigma_yx[:, observed] @ np.linalg.pinv(
                    sigma_xx[np.ix_(observed, observed)], rcond=1e-10
                )
                xz = (x[np.ix_(row_ids, observed)] - mean[observed]) / scale[observed]
                yhat_z[row_ids] = xz @ conditional.T
            yhat[row_ids] = mean[nx:] + yhat_z[row_ids] * scale[nx:]
        receipts.append(
            {
                "season_bin": int(season_bin),
                "center_doy": center,
                "train_timestamps": int(len(train)),
                "oas_shrinkage": float(estimator.shrinkage_),
            }
        )
    if not np.isfinite(yhat).all():
        raise RuntimeError("OAS produced non-finite deployment predictions")
    return yhat, receipts


def main() -> None:
    if sha256(BASE) != BASE_SHA256:
        raise RuntimeError("current P2 U base hash changed")
    data = load_p2_data(DATA)
    observations = data.observations.copy()
    observation_times = pd.to_datetime(observations["time"], utc=True)
    hidden = (
        observation_times.ge(GAP_START)
        & observation_times.lt(GAP_STOP)
        & observations["layer"].isin(TARGET)
    )
    if not observations.loc[hidden, ["temp", "psal"]].isna().all().all():
        raise RuntimeError("official hidden labels are unexpectedly populated")

    panel, x_columns, y_columns = build_panel(observations)
    test = data.test_index.copy()
    test_times = pd.to_datetime(test["time"], utc=True)
    query_times = pd.DatetimeIndex(test_times.drop_duplicates().sort_values())
    if not query_times.isin(panel.index).all():
        raise RuntimeError("test timestamp is absent from observations panel")
    yhat, fit_receipts = conditional_predict(panel, query_times, x_columns, y_columns)

    candidate_parts = []
    for position, layer in enumerate(TARGET):
        temp_position = y_columns.index(f"temp_{layer}")
        candidate_parts.append(
            pd.DataFrame(
                {
                    "time_utc": query_times,
                    "layer": layer,
                    "oas_temp": yhat[:, temp_position],
                }
            )
        )
    candidate = pd.concat(candidate_parts, ignore_index=True)
    keyed_test = test.assign(time_utc=test_times)
    keyed_test["_row"] = np.arange(len(keyed_test))
    keyed_test = keyed_test.merge(
        candidate, on=["time_utc", "layer"], how="left", validate="many_to_one"
    ).sort_values("_row")
    if keyed_test["oas_temp"].isna().any():
        raise RuntimeError("missing OAS prediction for official test row")

    base = pd.read_csv(BASE, dtype={"station": "string", "time": "string"})
    if list(base.columns) != KEYS + ["temp"] or not base[KEYS].equals(test[KEYS]):
        raise RuntimeError("P2 U base keys/order differ from test index")
    base_values = base["temp"].to_numpy(float)
    oas_values = keyed_test["oas_temp"].to_numpy(float)
    blended = (1.0 - ALPHA) * base_values + ALPHA * oas_values
    endpoints = public_endpoint_frame(observations)
    projection = project_profiles_vectorized(test, blended, endpoints)
    final_values = projection.prediction
    submission = build_submission(test, final_values)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    READY_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT, index=False, encoding="utf-8", lineterminator="\n")
    submission.to_csv(READY_OUTPUT, index=False, encoding="utf-8", lineterminator="\n")
    validation = validate_submission(OUTPUT, test)
    if OUTPUT.read_bytes() != READY_OUTPUT.read_bytes():
        raise RuntimeError("ready copy differs from canonical output")

    difference = final_values - base_values
    unprojected_difference = blended - base_values
    receipt = {
        "schema_version": f"p2.seasonal_oas_projected.submission.20260827.{DEPLOY_TAG}",
        "status": "READY_FOR_EXPLICIT_UPLOAD",
        "candidate": f"P2-SEASONAL-OAS-TS-{int(round(ALPHA * 100)):02d}-PROJECTED",
        "title": f"P2 계절 국소 T/S 조건부 프로파일 OAS {int(round(ALPHA * 100))}% v1",
        "one_line_summary": (
            f"기존 U를 {int(round((1.0 - ALPHA) * 100))}% 유지하고 공개층 T/S의 계절 국소 OAS 조건부 프로파일을 "
            f"{int(round(ALPHA * 100))}% 결합한 뒤 endpoint/PAVA 물리 투영을 적용했습니다."
        ),
        "alpha": ALPHA,
        "season_bin_days": SEASON_BIN_DAYS,
        "season_window_days": SEASON_WINDOW_DAYS,
        "feature_contract": {"x": x_columns, "y": y_columns},
        "fit_receipts": fit_receipts,
        "validation": validation,
        "projection": projection.diagnostics(),
        "difference_vs_current_u": {
            "rms": float(np.sqrt(np.mean(difference**2))),
            "unprojected_rms": float(np.sqrt(np.mean(unprojected_difference**2))),
            "changed_rows": int((np.abs(difference) > 1e-12).sum()),
            "maximum_absolute": float(np.max(np.abs(difference))),
        },
        "inputs": {
            "observations": {"path": str(DATA / "observations.csv"), "sha256": sha256(DATA / "observations.csv")},
            "test_index": {"path": str(DATA / "test_index.csv"), "sha256": sha256(DATA / "test_index.csv")},
            "sample_submission": {"path": str(DATA / "sample_submission.csv"), "sha256": sha256(DATA / "sample_submission.csv")},
            "base_u": {"path": str(BASE), "sha256": sha256(BASE)},
        },
        "outputs": {
            "canonical": {"path": str(OUTPUT), "sha256": sha256(OUTPUT)},
            "ready": {"path": str(READY_OUTPUT), "sha256": sha256(READY_OUTPUT)},
        },
        "leakage_contract": {
            "answer_file_read": False,
            "official_gap_hidden_label_reads": 0,
            "official_gap_hidden_labels_confirmed_nan": True,
            "official_upload_performed": False,
        },
    }
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    READY_NOTE.write_text(
        "제출물 제목: " + receipt["title"] + "\n"
        "한줄요약(접근방식): " + receipt["one_line_summary"] + "\n"
        "파일 SHA-256: " + receipt["outputs"]["ready"]["sha256"] + "\n"
        "상태: 사용자 명시 승인 후 P2 한 장 제출\n",
        encoding="utf-8-sig",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
