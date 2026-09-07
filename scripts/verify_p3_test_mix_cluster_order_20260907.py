"""Explain the finite-bootstrap discrepancy without changing predictions or fits."""
import argparse
import json
import os
import time
from pathlib import Path

from verify_p3_test_mix_20260907 import np, pd, sha, threadpool_limits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    with threadpool_limits(limits=2):
        f = pd.read_parquet(args.cpu, columns=["station", "episode_id", "current_hs",
                            "lead_h", "target_hs", "persistence", "final_prediction"])
        truth, base = f.target_hs.to_numpy(), f.final_prediction.to_numpy()
        candidate = base.copy()
        mask = f.lead_h.isin([12, 18, 24]).to_numpy()
        candidate[mask] = (base[mask] - .2 * f.persistence.to_numpy()[mask]) / .8
        bins = np.searchsorted([1.6, 1.7, 1.9, 2.4], f.current_hs, side="right")
        # These observed case proportions were independently checked in test-mix-check.json.
        weights = (np.array([.4, .2, .165, .135, .1]) / (np.bincount(bins) / len(f)))[bins]
        grouped = pd.DataFrame({"key": f.station.astype(str) + "_" + f.episode_id.astype(str),
                                "weight": weights, "base": weights * (base - truth)**2,
                                "candidate": weights * (candidate - truth)**2})
        clusters = grouped.groupby("key", sort=True)[["weight", "base", "candidate"]].sum().to_numpy()
        rng, deltas = np.random.default_rng(20260907), []
        for _ in range(2000):
            sample = clusters[rng.integers(0, len(clusters), len(clusters))].sum(axis=0)
            deltas.append(float(np.sqrt(sample[2] / sample[0]) - np.sqrt(sample[1] / sample[0])))
        ci = np.quantile(deltas, [.05, .95])
        p_improve = float(np.mean(np.asarray(deltas) < 0))
        assert len(clusters) == 951
        assert np.allclose(ci, [.002354, .007736], rtol=0, atol=5.1e-7)
        assert p_improve == .0005
    result = {"status": "BOOTSTRAP_ORDER_DIFFERENCE_RESOLVED", "new_fits": 0,
              "thread_limit": 2, "pid": os.getpid(), "script_sha256": sha(Path(__file__)),
              "input_sha256": sha(args.cpu), "rows": len(f), "clusters": len(clusters),
              "sort": "lexicographic station + '_' + str(episode_id), matching Fable code",
              "resamples": 2000, "seed": 20260907, "ci90": ci.tolist(),
              "improving_resamples": 1, "p_improve_exact": p_improve,
              "p_improve_formatted_3dp": format(p_improve, ".3f"),
              "explanation": "Same row errors/weights. Numeric tuple vs lexical cluster ordering changes seeded finite resamples, not the estimand. Original numeric-order result is preserved.",
              "caveat": "Descriptive OOF transport diagnostic, not an official test result or proof of impossibility.",
              "seconds": time.monotonic() - started}
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
