"""Prospectively sealed group and day guards for P1 v29."""

from __future__ import annotations

import numpy as np

from p1_qc.causal_scar_pu import ContractError, binary_f1


def eligible_groups(
    labels: np.ndarray,
    anchor: np.ndarray,
    proposed: np.ndarray,
    station: np.ndarray,
    layer: np.ndarray,
    *,
    minimum_support: int = 20,
) -> set[tuple[str, int]]:
    truth = np.asarray(labels, dtype=np.int8)
    incumbent = np.asarray(anchor, dtype=np.int8)
    add = np.asarray(proposed, dtype=bool) & (incumbent == 0)
    if not (truth.shape == incumbent.shape == add.shape == station.shape == layer.shape):
        raise ContractError("group guard arrays are not aligned")
    if minimum_support <= 0:
        raise ContractError("minimum group support must be positive")
    precision_floor = binary_f1(truth, incumbent) / 2.0
    allowed: set[tuple[str, int]] = set()
    keys = np.array(
        [f"{station_name}|{int(layer_value)}" for station_name, layer_value in zip(station, layer, strict=True)],
        dtype=object,
    )
    for key in np.unique(keys):
        group = keys == key
        additions = group & add
        support = int(additions.sum())
        if support < minimum_support:
            continue
        precision = float(truth[additions].mean())
        candidate = np.maximum(incumbent[group], add[group].astype(np.int8))
        if precision > precision_floor and binary_f1(truth[group], candidate) >= binary_f1(truth[group], incumbent[group]):
            name, value = str(key).rsplit("|", 1)
            allowed.add((name, int(value)))
    return allowed


def apply_group_guard(proposed: np.ndarray, station: np.ndarray, layer: np.ndarray, allowed: set[tuple[str, int]]) -> np.ndarray:
    add = np.asarray(proposed, dtype=bool)
    return add & np.fromiter(
        (
            (str(station_name), int(layer_value)) in allowed
            for station_name, layer_value in zip(station, layer, strict=True)
        ),
        dtype=bool,
        count=len(add),
    )


def day_cap_mask(proposed: np.ndarray, score: np.ndarray, day: np.ndarray, *, maximum_fraction: float = 0.005) -> np.ndarray:
    add = np.asarray(proposed, dtype=bool)
    value = np.asarray(score, dtype=np.float64)
    if add.shape != value.shape or add.shape != day.shape or not np.isfinite(value).all():
        raise ContractError("day cap arrays are invalid")
    if not 0 < maximum_fraction <= 0.005:
        raise ContractError("day cap must be in (0, 0.005]")
    kept = np.zeros(len(add), dtype=bool)
    indices = np.arange(len(add))
    for current in np.unique(day):
        rows = indices[day == current]
        candidates = rows[add[rows]]
        slots = int(np.floor(maximum_fraction * len(rows)))
        if slots <= 0 or candidates.size == 0:
            continue
        order = np.lexsort((candidates, -value[candidates]))
        kept[candidates[order[:slots]]] = True
    return kept


__all__ = ["apply_group_guard", "day_cap_mask", "eligible_groups"]
