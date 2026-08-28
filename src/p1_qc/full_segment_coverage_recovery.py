"""Coverage-complete inference helpers for a frozen P1 sequence encoder.

This module does not train or alter the encoder.  It applies a same-length
convolutional encoder to every eligible contiguous segment, including segments
shorter than the 512-row windows used during the parent label-free fit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def contiguous_segment_bounds(
    segments: np.ndarray, eligible: np.ndarray
) -> np.ndarray:
    """Return every maximal eligible run without imposing a minimum length."""
    segment_ids = np.asarray(segments)
    keep = np.asarray(eligible, dtype=bool)
    if len(segment_ids) != len(keep):
        raise ValueError("segments and eligible must have equal length")
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < len(keep):
        if not keep[start]:
            start += 1
            continue
        stop = start + 1
        while (
            stop < len(keep)
            and keep[stop]
            and segment_ids[stop] == segment_ids[start]
        ):
            stop += 1
        bounds.append((start, stop))
        start = stop
    return np.asarray(bounds, dtype=np.int64).reshape(-1, 2)


def infer_complete_segments(
    model: torch.nn.Module,
    values: np.ndarray,
    segments: np.ndarray,
    eligible: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Infer each complete segment once and return an explicit coverage audit."""
    bounds = contiguous_segment_bounds(segments, eligible)
    width = int(model.output_projection.out_channels)
    output = np.full((len(values), width), np.nan, dtype=np.float32)
    covered = np.zeros(len(values), dtype=bool)
    lengths: list[int] = []
    model.eval()
    with torch.inference_mode():
        for start, stop in bounds:
            tensor = torch.from_numpy(values[start:stop][None]).to(device)
            embedding = model(tensor)[0].cpu().numpy()
            output[start:stop] = embedding
            covered[start:stop] = True
            lengths.append(int(stop - start))
    eligible_rows = int(np.asarray(eligible, dtype=bool).sum())
    covered_rows = int((covered & np.asarray(eligible, dtype=bool)).sum())
    audit = {
        "segment_count": int(len(bounds)),
        "eligible_rows": eligible_rows,
        "covered_rows": covered_rows,
        "coverage": float(covered_rows / eligible_rows) if eligible_rows else 0.0,
        "minimum_segment_rows": min(lengths) if lengths else 0,
        "maximum_segment_rows": max(lengths) if lengths else 0,
        "segments_below_parent_window_rows": int(sum(length < 512 for length in lengths)),
        "rows_below_parent_window_rows": int(sum(length for length in lengths if length < 512)),
    }
    return output, covered, audit


def vectorized_conditional_prototype_scores(
    state: Any,
    embeddings: np.ndarray,
    day_sin: np.ndarray,
    day_cos: np.ndarray,
    cells: np.ndarray,
    *,
    knn_k: int,
    prototype_weight: float,
    device: torch.device,
    batch_rows: int = 4096,
) -> np.ndarray:
    """Evaluate the frozen prototype-plus-kNN score in bounded GPU batches."""
    z = np.asarray(embeddings, dtype=np.float32)
    design = np.column_stack([np.ones(len(z)), day_sin, day_cos]).astype(np.float32)
    seasonal = design @ state.harmonic
    cell_values = np.asarray(cells, dtype=str)
    mapping = {cell: index for index, cell in enumerate(state.cells)}
    scores = np.full(len(z), np.nan, dtype=np.float32)
    global_scale = np.median(state.scale, axis=0).astype(np.float32)
    root_width = float(np.sqrt(z.shape[1]))
    for cell in sorted(set(cell_values.tolist())):
        rows = np.flatnonzero((cell_values == cell) & np.isfinite(z).all(axis=1))
        if not len(rows):
            continue
        index = mapping.get(cell)
        if index is None:
            center = seasonal[rows]
            scale = global_scale
            prototype = np.sqrt(np.mean(((z[rows] - center) / scale) ** 2, axis=1))
            scores[rows] = prototype_weight * prototype
            continue
        scale = np.asarray(state.scale[index], dtype=np.float32)
        center = seasonal[rows] + float(state.cell_weight[index]) * np.asarray(
            state.cell_residual[index], dtype=np.float32
        )
        prototype = np.sqrt(np.mean(((z[rows] - center) / scale) ** 2, axis=1))
        bank = torch.from_numpy(
            np.asarray(state.banks[index], dtype=np.float32) / scale
        ).to(device)
        bank_scores = np.empty(len(rows), dtype=np.float32)
        for offset in range(0, len(rows), batch_rows):
            local = rows[offset : offset + batch_rows]
            query = torch.from_numpy((z[local] - seasonal[local]) / scale).to(device)
            distances = torch.cdist(query, bank) / root_width
            take = min(int(knn_k), int(bank.shape[0]))
            nearest = torch.topk(distances, take, dim=1, largest=False).values
            bank_scores[offset : offset + len(local)] = nearest.mean(dim=1).cpu().numpy()
        scores[rows] = (
            prototype_weight * prototype
            + (1.0 - prototype_weight) * bank_scores
        )
    return scores


__all__ = [
    "contiguous_segment_bounds",
    "infer_complete_segments",
    "vectorized_conditional_prototype_scores",
]
