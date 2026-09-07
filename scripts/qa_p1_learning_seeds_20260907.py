"""Independent full historical replay and arithmetic QA of two frozen P1 changes."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import p1_original_learning_seed_ablation_20260907_v2 as s
from sklearn.metrics import confusion_matrix, f1_score


def main():
    start = time.monotonic()
    m, r = s.m, s.m.runtime()
    data = Path(os.environ["P1_DATA_DIR"])
    cfg = m.read(m.ROOT / "configs/experiments/p1_champion_bracketb_20260907_v1.json")
    result = m.read(s.OUT / "terminal_result.json")
    checks = {"terminal": result["status"] == "TRAIN_INTERNAL_COMPLETE_QA_PENDING",
              "fits9": result["new_fits"] == len(result["fits"]) == 9,
              "separate_pid": result["pid"] != os.getpid(),
              "training_hash": m.sha(data / "train.csv") == cfg["train_sha256"],
              "source_pins": all(m.sha(Path(p)) == h for p, h in m.read(s.OUT / "source-seal.json").items()),
              "paired_sha": m.sha(s.OUT / "paired.parquet") == result["paired_sha256"]}
    for fit in result["fits"]:
        checks[fit["model"]] = m.sha(s.OUT / "models" / fit["model"]) == fit["sha256"]
    paired = r.pd.read_parquet(s.OUT / "paired.parquet")
    paired.time = r.pd.to_datetime(paired.time, utc=True)
    train = r.pd.read_csv(data / "train.csv", usecols=r.scope["old"].RAW+["label"])
    train.time = r.pd.to_datetime(train.time, utc=True)
    index = r.pd.MultiIndex.from_frame(train[r.composition.KEYS])
    ids = index.get_indexer(r.pd.MultiIndex.from_frame(paired[r.composition.KEYS]))
    checks["whole_population_truth"] = bool(len(paired) == 287862 and (ids >= 0).all() and r.np.array_equal(train.label.to_numpy()[ids], paired.label))
    proposals = r.pd.read_parquet(m.ROOT / "artifacts/p1_champion_reconstruction_20260906_v1_historical_proposals/proposals.parquet")
    proposals.time = r.pd.to_datetime(proposals.time, utc=True)
    pindex = r.pd.MultiIndex.from_frame(proposals[r.composition.KEYS])
    for fold, part in paired.groupby("fold", sort=True):
        part = part.reset_index(drop=True)
        keys = r.pd.MultiIndex.from_frame(part[r.composition.KEYS])
        valid = train.iloc[index.get_indexer(keys)].reset_index(drop=True)
        ms = proposals.proposal.to_numpy()[pindex.get_indexer(keys)]
        op, bp = [s.saved_predict(r, valid, fold, arm, cfg) for arm in ["O", "control"]]
        probs = {}
        for arm, seed in [("O_slow", 20260813), ("B_extra", 20260861), ("B_extra", 20260875)]:
            package = r.joblib.load(s.OUT / "models" / f"{fold}_{arm}_{seed}.joblib")
            x = package["encoder"].transform(m.bundle(r, valid, False, cfg))
            model = package["packages"][0]["global"]
            probs[seed, arm] = model.predict_proba(x)[:, 1]
        b5 = (bp*3 + probs[20260861, "B_extra"] + probs[20260875, "B_extra"]) / 5
        for arm, a, b in [("control", op, bp), ("O_slow", probs[20260813, "O_slow"], bp), ("B5", op, b5)]:
            checks[f"{fold}/{arm}/exact_bits"] = bool(r.np.array_equal(m.compose(r, valid, a, b, ms, cfg), part[arm]))
    for arm in ["control", "O_slow", "B5"]:
        _, fp, fn, tp = confusion_matrix(paired.label, paired[arm], labels=[0, 1]).ravel()
        recorded = result["metrics"][arm]
        checks[arm + "/counts"] = (int(tp), int(fp), int(fn)) == (recorded["tp"], recorded["fp"], recorded["fn"])
        checks[arm + "/f1"] = abs(f1_score(paired.label, paired[arm])-recorded["f1"]) < 1e-12
    paired["week"] = (paired.time.dt.tz_convert("Asia/Seoul").dt.normalize()-r.pd.Timestamp("2025-01-01", tz="Asia/Seoul")).dt.days // 7
    arms = {}
    for arm in ["O_slow", "B5"]:
        rng = r.np.random.default_rng(20260907)
        counts = r.np.zeros((2000, 2, 3), dtype=r.np.int64)
        weeks = []
        for fold, block in paired.groupby("fold", sort=True):
            items = []
            for week, part in block.groupby("week", sort=True):
                y = part.label.to_numpy()
                items.append([[(int(((y == 1) & (part[a].to_numpy() == 1)).sum())),
                               int(((y == 0) & (part[a].to_numpy() == 1)).sum()),
                               int(((y == 1) & (part[a].to_numpy() == 0)).sum())] for a in ["control", arm]])
                weeks.append({"fold": fold, "week": int(week), "rows": len(part), "delta_f1": float(f1_score(y, part[arm], zero_division=0)-f1_score(y, part.control, zero_division=0))})
            array = r.np.asarray(items)
            counts += array[rng.integers(0, len(array), size=(2000, len(array)))].sum(axis=1)
        numerator = 2*counts[:, :, 0]
        denominator = numerator+counts[:, :, 1]+counts[:, :, 2]
        f1 = r.np.divide(numerator, denominator, out=r.np.zeros_like(numerator, dtype=float), where=denominator != 0)
        delta = f1[:, 1]-f1[:, 0]
        cells = [{"station": str(st), "layer": int(layer), "rows": len(part),
                  "delta_f1": float(f1_score(part.label, part[arm], zero_division=0)-f1_score(part.label, part.control, zero_division=0))}
                 for (st, layer), part in paired.groupby(["station", "layer"], sort=True)]
        arms[arm] = {"delta_pooled_f1": result["metrics"][arm]["f1"]-result["metrics"]["control"]["f1"],
                     "CI90": r.np.quantile(delta, [.05, .95]).tolist(), "P_improvement": float((delta > 0).mean()),
                     "bootstrap": "2000 paired 7-day blocks stratified by fold, seed20260907; retrospective",
                     "worst_week": min(weeks, key=lambda x: x["delta_f1"]), "station_layer": cells,
                     "worst_station_layer": min(cells, key=lambda x: x["delta_f1"])}
    output = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
              "arms": arms, "metrics": result["metrics"], "by_fold": result["by_fold"],
              "result_sha256": m.sha(s.OUT / "terminal_result.json"), "pid": os.getpid(),
              "new_fits": 0, "official_rows": 0, "hidden_rows": 0, "seconds": time.monotonic()-start,
              "scope": "Q3/Q4 exact common MS-owned rows, no Q2 composed evidence"}
    m.write(s.OUT / "independent-qa.json", output)
    m.write(s.REPORT / "independent-qa.json", output)
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}
    print(json.dumps({k: output[k] for k in ["status", "metrics", "arms", "seconds"]}), flush=True)


if __name__ == "__main__":
    main()
