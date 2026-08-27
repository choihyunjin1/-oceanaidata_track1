from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\cedis\PycharmProjects\PythonProject")
SOURCE = ROOT / "artifacts" / "p1_matched_budget_local_compare_20260825_v1" / "predictions.parquet"
OUTPUT_DIR = ROOT / "artifacts" / "p1_current_router_oof_anchor_v1"
OUTPUT = OUTPUT_DIR / "anchor.parquet"
MANIFEST = OUTPUT_DIR / "manifest.json"

SOURCE_SHA256 = "23f9b59cc54a7502c87786280ef76319e64288a95a44f8c9ec37188a761033c5"
EXPECTED_ROWS = 421_032
KEYS = ["station", "year", "layer", "time"]
O_COLUMN = "incumbent_offline_xgboost__default"
B_COLUMN = "event_day_balanced_lightgbm__default"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("pinned P1 matched-budget prediction surface changed")
    if OUTPUT.exists() or MANIFEST.exists():
        raise FileExistsError("current-router anchor artifact already exists")

    columns = [*KEYS, "fold", O_COLUMN, B_COLUMN]
    source = pd.read_parquet(SOURCE, columns=columns)
    if len(source) != EXPECTED_ROWS or source.duplicated(KEYS).any():
        raise RuntimeError("P1 anchor source row/key contract changed")
    o = source[O_COLUMN].to_numpy(dtype=np.int8)
    b = source[B_COLUMN].to_numpy(dtype=np.int8)
    if not np.isin(o, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise RuntimeError("P1 anchor source prediction is not binary")

    station = source["station"].astype(str).to_numpy()
    layer = source["layer"].to_numpy(dtype=int)
    o_only = (o == 1) & (b == 0)
    b_only = (o == 0) & (b == 1)
    additions = o_only & (
        ((station == "G-ORS") & (layer == 1))
        | ((station == "I-ORS") & (layer == 2))
    )
    removals = b_only & (
        ((station == "S-ORS") & np.isin(layer, [1, 5, 6]))
        | ((station == "I-ORS") & (layer == 4))
    )
    router = np.where(removals, 0, np.where(additions, 1, b)).astype(np.int8)

    artifact = source.loc[:, [*KEYS, "fold"]].copy()
    artifact["current_router_prediction"] = router
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    temporary = OUTPUT.with_suffix(".parquet.partial")
    artifact.to_parquet(temporary, index=False)
    os.replace(temporary, OUTPUT)

    fold_counts = {
        str(fold): {
            "rows": int(len(frame)),
            "positive_rows": int(frame["current_router_prediction"].sum()),
        }
        for fold, frame in artifact.groupby("fold", sort=False)
    }
    manifest = {
        "schema_version": "p1.current_router_oof_anchor.v1",
        "status": "PASS_LABEL_FREE_RECONSTRUCTION",
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": SOURCE_SHA256,
        "source_columns": columns,
        "truth_columns_read": 0,
        "rules": {
            "base": B_COLUMN,
            "add_only_when_o1_b0": ["G-ORS/layer1", "I-ORS/layer2"],
            "remove_only_when_o0_b1": [
                "S-ORS/layer1",
                "S-ORS/layer5",
                "S-ORS/layer6",
                "I-ORS/layer4",
            ],
        },
        "rows": int(len(artifact)),
        "ordered_keys_unique": True,
        "additions_vs_b": int(additions.sum()),
        "removals_vs_b": int(removals.sum()),
        "positive_rows": int(router.sum()),
        "folds": fold_counts,
        "output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "output_bytes": OUTPUT.stat().st_size,
        "output_sha256": sha256(OUTPUT),
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
