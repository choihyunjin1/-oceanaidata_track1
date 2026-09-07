"""Independent process aggregate, model replay and complete-key QA; zero fits."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "bracketb_frozen_runner", ROOT / "scripts/p1_champion_bracketb_20260907_v1.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def qa(data):
    from sklearn.metrics import confusion_matrix, f1_score

    r = m.runtime()
    start = time.monotonic()
    cfg = m.read(ROOT / "configs/experiments" / f"{m.ID}.json")
    result = m.read(m.OUT / "terminal_result.json")
    checks = {
        "complete": result["status"] == "TRAIN_AND_INTERNAL_COMPLETE_QA_PENDING",
        "fresh_process": result["pid"] != os.getpid(),
        "fits_17": len(result["fits"]) == 17,
        "training_source": m.sha(data / "train.csv") == cfg["train_sha256"],
    }
    pins = m.read(m.OUT / "source-seal.json")
    checks["source_seal"] = all(m.sha(ROOT / p) == value for p, value in pins.items())
    checks["runner_hash"] = (
        m.sha(ROOT / "scripts/p1_champion_bracketb_20260907_v1.py") == result["runner_sha256"]
    )
    checks["paired_hash"] = m.sha(m.OUT / "paired.parquet") == result["paired_sha256"]
    paired = r.pd.read_parquet(m.OUT / "paired.parquet")
    checks["full_population"] = (
        len(paired) == 287862 and not paired.duplicated(r.composition.KEYS).any()
    )
    for fit in result["fits"]:
        checks[fit["model_file"]] = (
            m.sha(m.OUT / "models" / fit["model_file"]) == fit["model_sha256"]
        )
    truth = r.pd.read_csv(data / "train.csv", usecols=r.scope["old"].RAW + ["label"])
    truth.time = r.pd.to_datetime(truth.time, utc=True)
    source_index = r.pd.MultiIndex.from_frame(truth[r.composition.KEYS])
    paired.time = r.pd.to_datetime(paired.time, utc=True)
    positions = source_index.get_indexer(r.pd.MultiIndex.from_frame(paired[r.composition.KEYS]))
    checks["truth_keys_exact"] = bool(
        (positions >= 0).all() and r.np.array_equal(truth.label.to_numpy()[positions], paired.label)
    )
    proposal_root = ROOT / "artifacts/p1_champion_reconstruction_20260906_v1_historical_proposals"
    source_ms = r.pd.read_parquet(proposal_root / "proposals.parquet")
    source_ms.time = r.pd.to_datetime(source_ms.time, utc=True)
    ms_index = r.pd.MultiIndex.from_frame(source_ms[r.composition.KEYS])
    msqa = m.read(proposal_root / "qa.json")
    checks["MS_QA_hash"] = m.sha(proposal_root / "qa.json") == cfg["proposal_qa_sha256"]
    checks["MS_QA_pass"] = msqa["status"] == "PASS" and all(msqa["checks"].values())
    full = r.joblib.load(m.OUT / "models/full_bracket.joblib")
    full_probe = r.pd.read_csv(data / "train.csv", usecols=r.scope["old"].RAW, nrows=256)
    matrix = full["encoder"].transform(m.bundle(r, full_probe, True, cfg))
    checks["full_B_three_original_seeds"] = full["seeds"] == cfg["seeds"]
    for item in full["packages"]:
        separate = r.joblib.load(m.OUT / "models" / f"full_bracket_{item['seed']}.joblib")
        checks[f"full_B_{item['seed']}_model_probe"] = bool(
            r.np.array_equal(
                separate.predict_proba(matrix)[:, 1], item["global"].predict_proba(matrix)[:, 1]
            )
        )
        checks[f"full_B_{item['seed']}_700_trees"] = separate.booster_.num_trees() == 700
    contract_path = ROOT / "reports/p1_champion_reconstruction_20260906_v1/union-contract-v2.json"
    uc = m.read(contract_path)
    committed = subprocess.run(
        [
            "git",
            "show",
            "HEAD:reports/p1_champion_reconstruction_20260906_v1/union-contract-v2.json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    checks["union_contract_matches_committed_content"] = json.loads(committed) == uc
    supports = []
    for fold in uc["folds"]:
        part = paired.loc[paired.fold.eq(fold["id"])].reset_index(drop=True)
        ids = source_index.get_indexer(r.pd.MultiIndex.from_frame(part[r.composition.KEYS]))
        valid = truth.iloc[ids].reset_index(drop=True)
        mi = ms_index.get_indexer(r.pd.MultiIndex.from_frame(part[r.composition.KEYS]))
        checks[fold["id"] + "_whole_MS_keys"] = bool(
            (mi >= 0).all() and len(part) == fold["source_rows"]
        )
        probs = {}
        for arm in ("O", "control", "bracket"):
            package = r.joblib.load(m.OUT / "models" / f"{fold['id']}_{arm}.joblib")
            features = m.bundle(r, valid, arm == "bracket", cfg)
            x = package["encoder"].transform(features)
            probs[arm] = r.np.mean(
                r.np.vstack([p["global"].predict_proba(x)[:, 1] for p in package["packages"]]),
                axis=0,
            )
            for p in package["packages"]:
                original = r.joblib.load(
                    m.OUT / "models" / f"{fold['id']}_{arm}_{p['seed']}.joblib"
                )
                checks[f"{fold['id']}/{arm}/{p['seed']}_individual_combined"] = bool(
                    r.np.array_equal(
                        original.predict_proba(x)[:, 1], p["global"].predict_proba(x)[:, 1]
                    )
                )
        proposal = source_ms.proposal.to_numpy()[mi]
        for arm, b in (("control", "control"), ("candidate", "bracket")):
            rebuilt = m.compose(r, valid, probs["O"], probs[b], proposal, cfg)
            checks[f"{fold['id']}/{arm}_fresh_replay"] = bool(r.np.array_equal(rebuilt, part[arm]))
        train = truth.loc[truth.time.le(r.pd.Timestamp(fold["training_max_utc"]))]
        known = set(zip(train.station, train.layer, strict=True))
        missing = sum(key not in known for key in zip(valid.station, valid.layer, strict=True))
        checks[fold["id"] + "_MS_cutoff_exact"] = (
            msqa["folds"][fold["phase"]]["training_max_time_utc"] == fold["training_max_utc"]
        )
        supports.append(
            {
                "fold": fold["id"],
                "rows": len(valid),
                "unsupported_rows": missing,
                "unsupported_share": missing / len(valid),
                "cutoff": fold["training_max_utc"],
                "MS_cutoff_exact": msqa["folds"][fold["phase"]]["training_max_time_utc"]
                == fold["training_max_utc"],
            }
        )
    for arm in ("control", "candidate"):
        tn, fp, fn, tp = confusion_matrix(paired.label, paired[arm], labels=[0, 1]).ravel()
        expected = result["metrics"][arm]
        checks[arm + "_confusion"] = (int(tp), int(fp), int(fn)) == (
            expected["tp"],
            expected["fp"],
            expected["fn"],
        )
        checks[arm + "_f1"] = abs(f1_score(paired.label, paired[arm]) - expected["f1"]) < 1e-12
    paired["block"] = (
        paired.time.dt.tz_convert("Asia/Seoul").dt.normalize()
        - r.pd.Timestamp("2025-01-01", tz="Asia/Seoul")
    ).dt.days // 7
    rng = r.np.random.default_rng(20260906)
    total = r.np.zeros((2000, 2, 3), dtype=r.np.int64)
    slices = []
    for fold, group in paired.groupby("fold", sort=True):
        arrays = []
        for block, part in group.groupby("block", sort=True):
            row = []
            for arm in ("control", "candidate"):
                y, p = part.label.to_numpy(), part[arm].to_numpy()
                row.append(
                    [
                        int(((y == 1) & (p == 1)).sum()),
                        int(((y == 0) & (p == 1)).sum()),
                        int(((y == 1) & (p == 0)).sum()),
                    ]
                )
            arrays.append(row)
            slices.append(
                {
                    "fold": fold,
                    "block": int(block),
                    "rows": len(part),
                    "delta_f1": f1_score(part.label, part.candidate, zero_division=0)
                    - f1_score(part.label, part.control, zero_division=0),
                }
            )
        array = r.np.asarray(arrays)
        draw = rng.integers(0, len(array), size=(2000, len(array)))
        total += array[draw].sum(axis=1)
    numerator = 2 * total[:, :, 0]
    denominator = numerator + total[:, :, 1] + total[:, :, 2]
    scores = r.np.divide(
        numerator, denominator, out=r.np.zeros_like(numerator, dtype=float), where=denominator != 0
    )
    deltas = scores[:, 1] - scores[:, 0]
    station_layers = [
        {
            "station": str(station),
            "layer": int(layer),
            "rows": len(part),
            "control_f1": float(f1_score(part.label, part.control, zero_division=0)),
            "candidate_f1": float(f1_score(part.label, part.candidate, zero_division=0)),
            "delta_f1": float(
                f1_score(part.label, part.candidate, zero_division=0)
                - f1_score(part.label, part.control, zero_division=0)
            ),
        }
        for (station, layer), part in paired.groupby(["station", "layer"], sort=True)
    ]
    output = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "new_fits": 0,
        "pid": os.getpid(),
        "training_pid": result["pid"],
        "runtime_seconds": time.monotonic() - start,
        "metrics": result["metrics"],
        "full_B_aggregate_sha256": m.sha(m.OUT / "models/full_bracket.joblib"),
        "delta_f1": result["delta_f1"],
        "support": supports,
        "bootstrap": {
            "ci90": r.np.quantile(deltas, [0.05, 0.95]).tolist(),
            "P_improvement_descriptive": float((deltas > 0).mean()),
            "resamples": 2000,
        },
        "worst_block": min(slices, key=lambda x: x["delta_f1"]),
        "station_layer_slices": station_layers,
        "worst_station_layer": min(station_layers, key=lambda x: x["delta_f1"]),
        "contract_sha256_independently_recorded": m.sha(contract_path),
        "original_runner_seal_contract_omission": "union-contract path not pinned in runner seal; independently hashed and checked against unchanged source history in QA",
        "scope": cfg["comparison"],
        "official_rows": 0,
        "hidden_rows": 0,
        "uploads": 0,
    }
    m.write(m.OUT / "independent-qa.json", output)
    m.write(m.REPORT / "independent-qa.json", output)
    if not all(checks.values()):
        raise ValueError("Independent QA failed")
    print(output["status"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    qa(parser.parse_args().data.resolve())
