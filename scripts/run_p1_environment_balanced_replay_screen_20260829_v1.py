"""Run the preflight-approved low-fidelity environment-balanced replay screen."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_environment_balanced_replay_screen_20260829_v1"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
PREFLIGHT_PATH = ROOT / "artifacts" / EXPERIMENT_ID / "preflight.json"


def _load_base() -> Any:
    path = ROOT / "scripts" / "run_p1_mstcn_type_boundary_cascade_shadow_20260829_v1.py"
    spec = importlib.util.spec_from_file_location("p1_environment_balanced_replay_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load shared shadow runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rank(window: Any, seed: int, stratum: tuple[str, int, bool]) -> bytes:
    payload = (
        f"{seed}|{stratum[0]}|{stratum[1]}|{int(stratum[2])}|"
        f"{window.segment_id}|{window.start}|{window.valid_length}"
    ).encode()
    return hashlib.sha256(payload).digest()


def _repeat_to_target(windows: list[Any], target: int) -> tuple[Any, ...]:
    if not windows or target < 1:
        return ()
    return tuple(windows[index % len(windows)] for index in range(target))


def _balanced_windows(
    runner: Any,
    training: Any,
    capacity: dict[str, Any],
    config: dict[str, Any],
    phase: str,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    import numpy as np

    all_windows = runner._all_windows(training, capacity)  # noqa: SLF001
    keys = training.surface.keys.reset_index(drop=True)
    labels = np.asarray(training.surface.labels, dtype=np.int8)
    strata: dict[tuple[str, int, bool], list[Any]] = defaultdict(list)
    for window in all_windows:
        rows = window.row_ids
        first = int(rows[0])
        stratum = (
            str(keys.iloc[first]["station"]),
            int(keys.iloc[first]["layer"]),
            bool(labels[rows].any()),
        )
        strata[stratum].append(window)

    positive_target = int(config["budget"]["positive_windows_per_station_layer"])
    negative_target = int(config["budget"]["negative_windows_per_station_layer"])
    selected: list[Any] = []
    details: list[dict[str, Any]] = []
    seed = int(config["fixed_seed"])
    for stratum in sorted(strata):
        available = sorted(strata[stratum], key=lambda window: _rank(window, seed, stratum))
        target = positive_target if stratum[2] else negative_target
        chosen = _repeat_to_target(available, target)
        selected.extend(chosen)
        details.append(
            {
                "station": stratum[0],
                "layer": stratum[1],
                "positive_window": stratum[2],
                "available": len(available),
                "selected_with_replay": len(chosen),
                "unique_selected": len({(w.segment_id, w.start) for w in chosen}),
            }
        )
    if not selected:
        raise RuntimeError("balanced replay selected no windows")
    return tuple(selected), {
        "mode": "fixed_station_layer_positive_negative_balance_with_replay",
        "phase": phase,
        "all_windows": len(all_windows),
        "selected_windows": len(selected),
        "positive_target_per_station_layer": positive_target,
        "negative_target_per_station_layer": negative_target,
        "strata": details,
    }


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not PREFLIGHT_PATH.exists():
        raise FileNotFoundError("value preflight must run before low-fidelity training")
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    if preflight.get("decision") != "PASS_TO_LOW_FIDELITY":
        raise RuntimeError("proposal did not pass the value preflight")
    if config["budget"]["full_fidelity_epochs"] != 0:
        raise RuntimeError("screen must not authorize full-fidelity training")
    base = _load_base()
    base.EXPERIMENT_ID = EXPERIMENT_ID
    base.CONFIG_PATH = CONFIG_PATH
    base.ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID / "run"
    args = base._parse_args()  # noqa: SLF001
    if not args.execute:
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "status": "CHECK_ONLY",
                    "preflight": preflight["decision"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    base.execute(
        candidate_module_name="p1_qc.ms_tcn_environment_balanced_replay",
        window_selector=_balanced_windows,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
