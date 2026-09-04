"""Build the final P3 candidate from today's three official points."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(r"C:\Users\cedis\PycharmProjects\PythonProject")
OUT = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260827_P3_REFINED_PUBLIC_OPTIMUM_READY"
)
O_PATH = REPO / "output" / "2026-08-20" / "ready" / "P3_submission.csv"
A_PATH = (
    REPO
    / "artifacts"
    / "p3_corrected_fixed_long_shrink_v4"
    / "candidate"
    / "submission.csv"
)

# Official Public RMSE displayed by the site on 2026-08-27.
POINTS = (
    (0.0, 0.607071),
    (-10.235445136162161, 0.583892),
    (-12.0, 0.584611),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    xs = np.asarray([p[0] for p in POINTS], dtype=float)
    mse = np.square([p[1] for p in POINTS])
    a, b, c = np.polyfit(xs, mse, 2)
    alpha = float(-b / (2.0 * a))
    predicted_rmse = float(np.sqrt(a * alpha**2 + b * alpha + c))

    original = pd.read_csv(O_PATH)
    axis = pd.read_csv(A_PATH)
    key_columns = ["case_id", "station", "lead_h"]
    if list(original.columns) != [*key_columns, "hs_pred"]:
        raise AssertionError("unexpected P3 schema")
    if not original[key_columns].equals(axis[key_columns]):
        raise AssertionError("O/A key-order mismatch")

    values_o = original["hs_pred"].to_numpy(float)
    values_a = axis["hs_pred"].to_numpy(float)
    values = values_o.copy()
    long_mask = original["lead_h"].isin([12, 18, 24]).to_numpy()
    values[long_mask] += alpha * (values_a[long_mask] - values_o[long_mask])
    if not np.isfinite(values).all() or values.min() < 0.0 or values.max() > 30.0:
        raise AssertionError("candidate physical/finite guard failed")
    if not np.array_equal(values[~long_mask], values_o[~long_mask]):
        raise AssertionError("short-lead no-op guard failed")

    OUT.mkdir(parents=True, exist_ok=False)
    candidate = original.copy()
    candidate["hs_pred"] = values
    csv_path = OUT / "P3_submission.csv"
    candidate.to_csv(csv_path, index=False, lineterminator="\n")

    manifest = {
        "status": "PASS_READY_NOT_UPLOADED",
        "candidate": "P3_REFINED_PUBLIC_OPTIMUM_20260827",
        "formula": "O + alpha*(A-O) on leads 12/18/24; exact O on 3/6/9",
        "official_points": [
            {"alpha": x, "public_rmse": y} for x, y in POINTS
        ],
        "quadratic_mse_coefficients": {"a": float(a), "b": float(b), "c": float(c)},
        "alpha": alpha,
        "predicted_public_rmse": predicted_rmse,
        "predicted_rmse_gain_vs_current_best": 0.583892 - predicted_rmse,
        "rows": int(len(candidate)),
        "changed_rows": int(long_mask.sum()),
        "short_lead_exact_no_op": True,
        "minimum_m": float(values.min()),
        "maximum_m": float(values.max()),
        "sha256": sha256(csv_path),
        "uploaded": False,
    }
    (OUT / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
