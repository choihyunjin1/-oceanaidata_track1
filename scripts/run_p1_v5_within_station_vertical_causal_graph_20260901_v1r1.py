"""Time-unit-only repair launcher for the frozen P1 v5 graph science."""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "scripts/run_p1_v5_within_station_vertical_causal_graph_20260901_v1.py"
ORIGINAL_SHA256 = "b2cccef50328378a4f1802015139119502542a49115b62dd6ded8725a9568a71"
OLD_ID = "p1_v5_within_station_vertical_causal_graph_20260901_v1"
NEW_ID = "p1_v5_within_station_vertical_causal_graph_20260901_v1r1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _patched_source() -> str:
    if _sha(ORIGINAL) != ORIGINAL_SHA256:
        raise RuntimeError("frozen v5 runner hash drifted")
    source = ORIGINAL.read_text(encoding="utf-8")
    if source.count(OLD_ID) != 1:
        raise RuntimeError("unexpected predecessor namespace occurrences")
    source = source.replace(OLD_ID, NEW_ID)
    marker = "TRANSPORT_FACTOR = 0.30\n"
    helper = r'''

def _time_ns(values: pd.Series) -> np.ndarray:
    index = pd.DatetimeIndex(values)
    if index.tz is None or index.hasnans:
        raise RuntimeError("time contract requires complete timezone-aware timestamps")
    result = index.as_unit("ns").asi8
    lower = pd.Timestamp("2020-01-01T00:00:00Z").value
    upper = pd.Timestamp("2030-01-01T00:00:00Z").value
    if len(result) and (int(result.min()) < lower or int(result.max()) >= upper):
        raise RuntimeError("time integer is not epoch nanoseconds in the authorized range")
    return result


def _boundary_contract(train_csv: Path, parts: dict[str, Any], fraction: float) -> dict[str, Any]:
    parsed = pd.to_datetime(
        pd.read_csv(train_csv, usecols=["time"])["time"],
        utc=True,
        errors="raise",
        format="mixed",
    )
    all_ns = _time_ns(parsed)
    boundaries = {}
    for fold, item in parts.items():
        cutoff_ns = pd.Timestamp(item["cutoff"]).value
        prefix = np.sort(np.unique(all_ns[all_ns <= cutoff_ns]))
        position = max(0, int(len(prefix) * fraction) - 1)
        boundary_ns = int(prefix[position])
        if not int(all_ns.min()) <= boundary_ns <= cutoff_ns:
            raise RuntimeError(f"cutoff boundary invalid: {fold}")
        boundaries[fold] = {
            "cutoff_utc": pd.Timestamp(cutoff_ns, tz="UTC").isoformat(),
            "boundary_utc": pd.Timestamp(boundary_ns, tz="UTC").isoformat(),
            "boundary_ns": boundary_ns,
        }
    if len({item["boundary_ns"] for item in boundaries.values()}) != len(boundaries):
        raise RuntimeError("fold boundaries are not cutoff-specific")
    return {"status": "PASS_NS_CUTOFF_DISTINCT", "folds": boundaries}
'''
    if source.count(marker) != 1:
        raise RuntimeError("time helper insertion point drifted")
    source = source.replace(marker, marker + helper)
    replacements = {
        'group["_time"].astype("int64").unique()': 'np.unique(_time_ns(group["_time"]))',
        'group["_time"].astype("int64").to_numpy()': '_time_ns(group["_time"])',
        'frame["_time"].astype("int64").to_numpy()': '_time_ns(frame["_time"])',
        'frame.loc[frame["_time"].astype("int64") <= cutoff_ns, "_time"].astype("int64").unique()': 'np.unique(_time_ns(frame.loc[_time_ns(frame["_time"]) <= cutoff_ns, "_time"]))',
    }
    expected_counts = {
        'group["_time"].astype("int64").unique()': 1,
        'group["_time"].astype("int64").to_numpy()': 1,
        'frame["_time"].astype("int64").to_numpy()': 2,
        'frame.loc[frame["_time"].astype("int64") <= cutoff_ns, "_time"].astype("int64").unique()': 1,
    }
    for old, new in replacements.items():
        if source.count(old) != expected_counts[old]:
            raise RuntimeError(f"time replacement count drifted: {old}")
        source = source.replace(old, new)
    if '.astype("int64")' in source:
        raise RuntimeError("ambiguous time integer conversion remains")
    return_marker = '    return {\n        "schema_version": "p1.v5.vertical_graph.preflight.v1",'
    injected_return = (
        '    time_contract = _boundary_contract('
        'train, parts, config["selection"]["inner_train_fraction"]'
        ')\n'
        + return_marker
    )
    if source.count(return_marker) != 1:
        raise RuntimeError("preflight boundary insertion point drifted")
    source = source.replace(return_marker, injected_return)
    receipt_marker = '        "vertical_contract": contract,\n'
    if source.count(receipt_marker) != 1:
        raise RuntimeError("time receipt insertion point drifted")
    source = source.replace(
        receipt_marker, receipt_marker + '        "time_contract": time_contract,\n'
    )
    qa_marker = '        "zero_operation": all(value == 0 for value in ready["counters"].values()),\n'
    if source.count(qa_marker) != 1:
        raise RuntimeError("QA insertion point drifted")
    source = source.replace(
        qa_marker,
        qa_marker
        + '        "time_unit_ns": ready["time_contract"]["status"] == "PASS_NS_CUTOFF_DISTINCT",\n',
    )
    return source


def _load_impl() -> types.ModuleType:
    name = "p1_v5_vertical_graph_v1r1_impl"
    module = types.ModuleType(name)
    module.__file__ = str(Path(__file__).resolve())
    sys.modules[name] = module
    exec(compile(_patched_source(), module.__file__, "exec"), module.__dict__)
    return module


def main() -> None:
    _load_impl().main()


if __name__ == "__main__":
    main()
