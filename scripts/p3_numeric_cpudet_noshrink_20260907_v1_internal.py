"""No-fit paired diagnostic of the fixed CPU no-shrink arm; aggregate output only."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def undo(final, persistence, lead):
    out = np.asarray(final, dtype=float).copy()
    active = np.isin(lead, [12, 18, 24])
    out[active] = (out[active] - 0.2 * np.asarray(persistence)[active]) / 0.8
    return out


def compare(frame):
    y, a, z = [frame[x].to_numpy(float) for x in ["target_hs", "final_prediction", "noshrink"]]
    sa, sz = float(np.sum((y-a)**2)), float(np.sum((y-z)**2))
    ra, rz = float(np.sqrt(sa/len(y))), float(np.sqrt(sz/len(y)))
    return {"rows": len(y), "base_sse": sa, "candidate_sse": sz,
            "base_rmse": ra, "candidate_rmse": rz, "delta": rz-ra}


def bootstrap(frame, cfg):
    f = frame.copy()
    f["a_sse"] = (f.target_hs-f.final_prediction)**2
    f["z_sse"] = (f.target_hs-f.noshrink)**2
    g = f.groupby(["station", "episode_id"], sort=True).agg(
        n=("target_hs", "size"), a=("a_sse", "sum"), z=("z_sse", "sum"))
    n, a, z = [g[c].to_numpy() for c in ["n", "a", "z"]]
    rng = np.random.default_rng(cfg["seed"])
    d = []
    for _ in range(cfg["resamples"]):
        ix = rng.integers(0, len(g), len(g))
        d.append(float(np.sqrt(z[ix].sum()/n[ix].sum())-np.sqrt(a[ix].sum()/n[ix].sum())))
    return {"unit": "station x episode_id", "blocks": len(g),
            "resamples": cfg["resamples"], "seed": cfg["seed"],
            "ci90": np.quantile(d, [.05, .95]).tolist(), "p_improve": float(np.mean(np.array(d)<0)),
            "retrospective": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["evaluate", "qa"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    cfg = json.loads(args.config.read_text())
    base = Path(cfg["base"])
    sys.path.insert(0, str(base / "02_code"))
    u = importlib.import_module("unbounded")
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    u.b.access_guard(source, False)
    u.settings()
    receipt = u.training_receipt()
    q = u.b.read(base / "06_docs/cpudet-training-qa.json")
    assert q["status"] == "PASS" and q["training_sha256"] == sha(base / "06_docs/cpudet-training.json")
    oof_path = base / "04_logs/candidate_oof.parquet"
    assert sha(oof_path) == receipt["oof_sha256"]
    f = pd.read_parquet(oof_path)
    assert len(f) == 103602 and not f.duplicated(u.E.KEYS).any()
    assert f.episode_id.notna().all() and f.groupby("anchor_id").size().eq(6).all()
    f["noshrink"] = undo(f.final_prediction, f.persistence, f.lead_h)
    assert np.isfinite(f.noshrink).all() and f.noshrink.between(-1e-12, 30+1e-12).all()
    assert np.array_equal(f.loc[f.lead_h.isin([3, 6, 9]), "final_prediction"],
                          f.loc[f.lead_h.isin([3, 6, 9]), "noshrink"])
    if args.mode == "evaluate":
        result = {"status": "INTERNAL_EVALUATION_COMPLETE", "pid": os.getpid(),
                  "oof_sha256": sha(oof_path), "config_sha256": sha(args.config),
                  "training_receipt_sha256": sha(base / "06_docs/cpudet-training.json"),
                  "new_fits": 0, "official_rows": 0, "hidden_rows": 0,
                  "pooled": compare(f), "bootstrap": bootstrap(f, cfg["bootstrap"]),
                  "low_wave_hs0_lt_1_7": compare(f.loc[f.current_hs < 1.7]),
                  "tables": {key: {str(v): compare(part) for v, part in f.groupby(key)}
                             for key in ["fold", "station", "lead_h"]}}
        groups = [(f"{station}/{ep}", compare(p)) for (station, ep), p in f.groupby(["station", "episode_id"])]
        result["worst_episode"] = max(groups, key=lambda x: x[1]["delta"])
        result["seconds"] = time.monotonic()-start
        save(args.out / "internal-evaluation.json", result)
    else:
        old = json.loads((args.out / "internal-evaluation.json").read_text())
        assert old["pid"] != os.getpid() and old["oof_sha256"] == sha(oof_path)
        assert old["pooled"] == compare(f) and old["bootstrap"] == bootstrap(f, cfg["bootstrap"])
        # Rebuild the raw routed forecasts from frozen historical routers and components.
        features = pd.read_parquet(base / "04_logs/features.parquet")
        anchors = pd.read_parquet(base / "04_logs/anchors.parquet")
        u.E.OUT = base / "03_model/routers"
        original = u.E.apply_long_lead_persistence_shrink
        def identity(routed, persistence, leads, *, config):
            assert config.weight == .2 and config.active_leads == (12, 18, 24)
            return np.asarray(routed).copy()
        u.E.apply_long_lead_persistence_shrink = identity
        raw, _ = u.E.fixed_policy(f.drop(columns="noshrink"), features, anchors,
            u.b.read(base / "02_code/configs/evaluation/ocean_forward_v5.json"), arm="candidate",
            fit=False, expected_receipts=receipt["historical_routers"])
        u.E.apply_long_lead_persistence_shrink = original
        assert f[u.E.KEYS].equals(raw[u.E.KEYS])
        maxdiff = float(np.max(np.abs(f.noshrink-raw.final_prediction)))
        assert maxdiff < 1e-12
        restored = original(raw.final_prediction.to_numpy(), f.persistence.to_numpy(),
                            f.lead_h.to_numpy(), config=u.E.LongLeadPersistenceShrink(.2, (12, 18, 24)))
        assert np.array_equal(restored, f.final_prediction.to_numpy())
        save(args.out / "independent-qa.json", {"status": "PASS", "pid": os.getpid(),
            "evaluation_pid": old["pid"], "internal_evaluation_sha256": sha(args.out / "internal-evaluation.json"),
            "oof_rows": len(f), "inverse_vs_saved_router_max_abs": maxdiff,
            "saved_router_reapplied_shrink_exact": True, "short_leads_unchanged": True,
            "bootstrap_independently_replayed": True, "new_fits": 0, "official_rows": 0,
            "hidden_rows": 0, "seconds": time.monotonic()-start})
    print(json.dumps({"stage": args.mode, "status": "PASS", "seconds": time.monotonic()-start}))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
