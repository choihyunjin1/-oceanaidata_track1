"""Key-safe union of independently generated P1 model outputs; no I/O or fitting."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

KEYS = ("station", "year", "layer", "time")


def canonical_keys(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(KEYS).difference(frame.columns)
    if missing:
        raise ValueError(f"missing prediction keys: {sorted(missing)}")
    keys = frame.loc[:, KEYS].copy().reset_index(drop=True)
    if keys.isna().any().any() or len(keys) == 0:
        raise ValueError("prediction keys must be nonempty and nonmissing")
    keys["station"] = keys.station.astype(str)
    for name in ("year", "layer"):
        numeric = pd.to_numeric(keys[name], errors="raise")
        if not np.isfinite(numeric).all() or not numeric.eq(np.floor(numeric)).all():
            raise ValueError(f"noninteger key: {name}")
        keys[name] = numeric.astype(np.int64)
    # Distributed naive timestamps are KST; offset-bearing timestamps stay absolute.
    parsed = pd.to_datetime(keys.time, errors="raise", format="mixed")
    if not hasattr(parsed.dtype, "tz"):
        if not pd.api.types.is_datetime64_dtype(parsed):
            raise ValueError("mixed naive/aware prediction timestamps are unsupported")
        parsed = parsed.dt.tz_localize("Asia/Seoul")
    keys["time"] = parsed.dt.tz_convert("UTC")
    if keys.time.isna().any() or keys.duplicated().any():
        raise ValueError("invalid or duplicate prediction keys")
    return keys


def ordered_key_sha256(frame: pd.DataFrame) -> str:
    keys = canonical_keys(frame)
    keys["time"] = keys.time.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    payload = keys.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def binary_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    values = pd.to_numeric(frame[name], errors="raise").to_numpy()
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"nonbinary prediction: {name}")
    return values.astype(np.int8)


def combine(
    tree: pd.DataFrame,
    mstcn: pd.DataFrame,
    *,
    tree_column: str = "tree",
    proposal_column: str = "proposal",
) -> tuple[pd.DataFrame, dict]:
    """Require identical key sets; preserve tree order and OR every station.

    No old-answer path, station exclusion, manual row override, threshold search,
    ground-truth access, or leaderboard-derived weighting is implemented here.
    Historical subsets must be explicitly resolved and reported by the caller.
    """
    left, right = canonical_keys(tree), canonical_keys(mstcn)
    tree_bits = binary_column(tree, tree_column)
    proposal = binary_column(mstcn, proposal_column)
    left_index = pd.MultiIndex.from_frame(left)
    right_index = pd.MultiIndex.from_frame(right)
    if len(left) != len(right) or not left_index.isin(right_index).all():
        raise ValueError("model prediction key sets differ; implicit intersection forbidden")
    permutation = right_index.get_indexer(left_index)
    proposal = proposal[permutation]
    candidate = np.maximum(tree_bits, proposal)
    result = tree.loc[:, KEYS].copy().reset_index(drop=True)
    result["label"] = candidate
    receipt = {
        "rows": len(result),
        "ordered_keys_sha256": ordered_key_sha256(tree),
        "tree_positive": int(tree_bits.sum()),
        "proposal_positive": int(proposal.sum()),
        "union_positive": int(candidate.sum()),
        "added_rows": int(((tree_bits == 0) & (candidate == 1)).sum()),
        "tree_positive_removed_rows": int(((tree_bits == 1) & (candidate == 0)).sum()),
        "proposal_reordered": not np.array_equal(permutation, np.arange(len(tree))),
        "manual_overrides": 0,
        "new_fit_count": 0,
    }
    return result, receipt
