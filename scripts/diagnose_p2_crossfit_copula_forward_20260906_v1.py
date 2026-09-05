"""Read-only post-mortem of the consumed v1 invariant assertion; no fitting."""

import json
import os
from pathlib import Path

import numpy as np
import run_p2_crossfit_copula_forward_20260906_v1 as m


def main():
    failure = json.loads((m.OUT / "terminal_failure.json").read_text())
    assert failure["status"] == "TERMINAL_TECHNICAL_FAILURE"
    cfg, contract, _ = m.settings()
    seal = json.loads(m.SEAL.read_text())
    assert seal["hashes"] == m.fingerprints()
    source = m.install_guard()
    m.torch.set_num_threads(2)
    with m.threadpool_limits(2):
        frame, _ = m.load_population(cfg)
        frozen = np.load(m.OUT / "baseline_oof.npz", allow_pickle=False)
        spec = next(s for s in contract["P2"]["folds"] if s["id"] == "B2")
        _, valid = m.masks(frame.time, spec)
        local = frame.loc[valid].reset_index(drop=True)
        altered, selected = m.outage_frame(local, spec)
        c, ac = frozen["natural_C3"][valid], frozen["outage_C3"][valid]
        nx = m.profile.physical_features(local, c)
        ax = m.profile.physical_features(altered, ac)
        fitted_path = m.OUT / "03_model/B2_crossfit_full.npz"
        fitted = dict(np.load(fitted_path, allow_pickle=False))
        nc = m.profile.predict_copula(fitted, nx)
        oc = m.profile.predict_copula(fitted, ax)
        difference = (c + nc)[~selected] - (ac + oc)[~selected]
        correction_difference = nc[~selected] - oc[~selected]
        # Equal input row subsets must produce equal output when batch shape is equal.
        subset_equal = np.array_equal(
            m.profile.predict_copula(fitted, nx[~selected]),
            m.profile.predict_copula(fitted, ax[~selected]),
        )
        present = np.isfinite(nx)
        changed = np.flatnonzero(~selected)[difference != 0]
        pattern_sizes = []
        for pattern in np.unique(present[changed], axis=0):
            pattern_sizes.append(
                {
                    "natural_batch_rows": int(np.all(np.isfinite(nx) == pattern, axis=1).sum()),
                    "outage_batch_rows": int(np.all(np.isfinite(ax) == pattern, axis=1).sum()),
                }
            )
        payload = {
            "status": "POST_MORTEM_READ_ONLY_COMPLETE",
            "failure": failure,
            "failure_line": 569,
            "fold": "B2",
            "arm": "crossfit_full",
            "outside_outage_rows": int((~selected).sum()),
            "outside_baseline_exact_equal": bool(np.array_equal(c[~selected], ac[~selected])),
            "outside_physical_features_exact_equal_including_nan": bool(
                np.array_equal(nx[~selected], ax[~selected], equal_nan=True)
            ),
            "outside_prediction_different_rows": int(np.count_nonzero(difference)),
            "outside_prediction_max_abs_difference": float(np.max(np.abs(difference))),
            "outside_correction_max_abs_difference": float(np.max(np.abs(correction_difference))),
            "same_shape_subset_predictions_exact_equal": subset_equal,
            "affected_missingness_pattern_batch_sizes": pattern_sizes,
            "serialized_backbone_models": len(list((m.OUT / "03_model").glob("*.pt"))),
            "serialized_copula_models": len(list((m.OUT / "03_model").glob("*.npz"))),
            "recorded_post_invariant_copula_receipts": failure["completed_copula_fits"],
            "actual_completed_fit_calls": 40,
            "new_fits": 0,
            "performance_metrics_read": 0,
            "official_access_rows": 0,
            "csv_written": 0,
            "upload": 0,
            "source_sha256_unchanged": m.base.file_hash(source) == cfg["source_sha256"],
            "sealed_hashes_unchanged": m.fingerprints() == seal["hashes"],
            "diagnostic_runner_sha256": m.base.file_hash(Path(__file__)),
            "pid": os.getpid(),
        }
    m.save(m.REPORT / "technical-diagnostic.json", payload)
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
