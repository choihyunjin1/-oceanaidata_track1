"""Pure, saved-weight inference for the frozen P2 submission candidate.

The final path intentionally starts from the immutable P2 observations and
saved model/checkpoint artifacts.  It does not read any intermediate
submission CSV.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable, Mapping
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import P2Data, load_p2_data, resolve_data_dir
from .deep_data import build_panel
from .deep_training import predict_full_checkpoint
from .features import build_test_features
from .profile_projection import project_profiles, project_profiles_vectorized, public_endpoint_frame
from .regime_gate import build_public_state_features, predict_soft_gate
from .research import append_public_dynamics, append_public_m2_harmonics, select_lean_m2_dynamics
from .submission import build_submission, validate_submission

FINAL_LAYER_FACTORS = {2: 10.0, 3: 0.0, 4: 2.0}
DEEP_CONTRIBUTORS = (
    "router_400",
    "depth_query_bitcn",
    "lsti_style",
    "timemixerpp_style",
    "moment_units_scratch",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_float_roundtrip(values: np.ndarray) -> np.ndarray:
    """Recreate a historical CSV numeric boundary entirely in memory."""

    buffer = io.StringIO()
    pd.DataFrame({"value": np.asarray(values, dtype=np.float64)}).to_csv(
        buffer, index=False, lineterminator="\n"
    )
    buffer.seek(0)
    return pd.read_csv(buffer)["value"].to_numpy(dtype=np.float64)


def _test_rows(index: pd.DataFrame, times: pd.DatetimeIndex, panel: np.ndarray) -> np.ndarray:
    positions = times.get_indexer(pd.to_datetime(index["time"], utc=True))
    if (positions < 0).any():
        raise ValueError("test_index time is absent from the deep panel")
    return panel[positions, index["layer"].to_numpy(int) - 2]


def predict_router_component(data: P2Data, model: object) -> np.ndarray:
    """Rebuild the frozen 400-round layer router from raw observations."""

    base = build_test_features(data)
    dynamic = append_public_dynamics(base, data.observations)
    lean = select_lean_m2_dynamics(base, dynamic)
    phase = append_public_m2_harmonics(lean, data.observations)
    components = model.predict_components(base, lean, phase)
    prediction = np.asarray(components["router"], dtype=np.float64)
    if prediction.shape != (len(data.test_index),) or not np.isfinite(prediction).all():
        raise ValueError("saved P2 router produced invalid predictions")
    return prediction


def predict_deep_components(
    data: P2Data,
    result: Mapping[str, object],
    *,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, np.ndarray]:
    """Run each of the eight frozen deep checkpoints exactly once."""

    panel = build_panel(data.observations)
    full_models = result["full_models"]
    total = sum(len(full_models[name]) for name in DEEP_CONTRIBUTORS if name != "router_400")
    done = 0
    output: dict[str, np.ndarray] = {}
    for name in DEEP_CONTRIBUTORS:
        if name == "router_400":
            continue
        values: list[np.ndarray] = []
        for entry in full_models[name]:
            checkpoint = Path(entry["checkpoint"])
            expected = str(entry["checkpoint_sha256"])
            actual = sha256_file(checkpoint)
            if actual != expected:
                raise ValueError(f"P2 checkpoint hash changed: {checkpoint}")
            if progress is not None:
                progress(name, done, total)
            values.append(
                _test_rows(
                    data.test_index,
                    panel.times,
                    predict_full_checkpoint(checkpoint, panel),
                )
            )
            done += 1
        output[name] = np.mean(values, axis=0)
    if progress is not None:
        progress("complete", total, total)
    return output


def _weighted_stack(frame: pd.DataFrame, weights: Mapping[str, Mapping[str, float]]) -> np.ndarray:
    prediction = np.full(len(frame), np.nan, dtype=np.float64)
    for layer in (2, 3, 4):
        keep = frame["layer"].to_numpy(int) == layer
        vector = np.array([weights[str(layer)][name] for name in DEEP_CONTRIBUTORS])
        prediction[keep] = frame.loc[keep, DEEP_CONTRIBUTORS].to_numpy(float) @ vector
    if not np.isfinite(prediction).all():
        raise ValueError("P2 stack produced non-finite predictions")
    return prediction


def compose_final_prediction(
    data: P2Data,
    *,
    router_prediction: np.ndarray,
    deep_predictions: Mapping[str, np.ndarray],
    stack_weights: Mapping[str, Mapping[str, float]],
    gate_model: object,
    layer_factors: Mapping[int, float] = FINAL_LAYER_FACTORS,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Apply the frozen Deep Stack, soft route and physical projections."""

    # The frozen research lineage persisted the router component before the
    # deep stack and persisted three subsequent candidates.  Mirror those
    # deterministic float boundaries in memory so this pure path is byte exact
    # while remaining independent of every intermediate submission CSV.
    router_serialized = csv_float_roundtrip(router_prediction)
    component_frame = pd.DataFrame(
        {
            "time": pd.to_datetime(data.test_index["time"], utc=True),
            "layer": data.test_index["layer"].to_numpy(int),
            "router_400": router_serialized,
            **{name: np.asarray(deep_predictions[name]) for name in DEEP_CONTRIBUTORS[1:]},
        }
    )
    deep_stack = csv_float_roundtrip(_weighted_stack(component_frame, stack_weights))
    endpoints = public_endpoint_frame(data.observations)
    base_raw = project_profiles(data.test_index, deep_stack, endpoints)
    base_prediction = csv_float_roundtrip(base_raw.prediction)

    public_state = build_public_state_features(
        data.observations, component_frame[["time", "layer"]]
    )
    gate_frame = pd.concat(
        [
            component_frame.reset_index(drop=True),
            public_state.drop(columns=["time", "layer"]).reset_index(drop=True),
        ],
        axis=1,
    )
    raw_gate = csv_float_roundtrip(predict_soft_gate(gate_model, gate_frame))
    routed_input = base_prediction.copy()
    use_gate = data.test_index["layer"].isin((2, 4)).to_numpy()
    routed_input[use_gate] = raw_gate[use_gate]
    routed = project_profiles_vectorized(data.test_index, routed_input, endpoints).prediction

    layers = data.test_index["layer"].to_numpy(int)
    scale = np.array([float(layer_factors[int(layer)]) for layer in layers])
    final_input = base_prediction + scale * (routed - base_prediction)
    final = project_profiles_vectorized(data.test_index, final_input, endpoints).prediction
    if not np.isfinite(final).all():
        raise ValueError("P2 final composition produced non-finite predictions")
    diagnostics: dict[str, float | int] = {
        "rows": len(final),
        "deep_projection_active_rows": int(base_raw.active_mask.sum()),
        "soft_route_rows": int(use_gate.sum()),
        "minimum": float(final.min()),
        "maximum": float(final.max()),
    }
    return final, diagnostics


def reproduce_final_submission(
    *,
    data_dir: str | Path,
    router_model_path: str | Path,
    deep_result_path: str | Path,
    gate_model_path: str | Path,
    output_path: str | Path,
    expected_sha256: str | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, object]:
    data = load_p2_data(resolve_data_dir(Path(data_dir)))
    router = joblib.load(router_model_path)
    deep_result = json.loads(Path(deep_result_path).read_text(encoding="utf-8"))
    gate = joblib.load(gate_model_path)
    router_prediction = predict_router_component(data, router)
    deep_predictions = predict_deep_components(data, deep_result, progress=progress)
    prediction, diagnostics = compose_final_prediction(
        data,
        router_prediction=router_prediction,
        deep_predictions=deep_predictions,
        stack_weights=deep_result["weights_by_layer"],
        gate_model=gate,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    build_submission(data.test_index, prediction).to_csv(
        target, index=False, encoding="utf-8", lineterminator="\n"
    )
    validation = validate_submission(target, data.test_index)
    actual = sha256_file(target)
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError(f"P2 reproduced SHA differs: {actual} != {expected_sha256}")
    return {**validation, **diagnostics, "sha256": actual}
