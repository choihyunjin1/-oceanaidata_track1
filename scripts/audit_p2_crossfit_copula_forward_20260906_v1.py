"""Aggregate source-date explanation for fixed empty B4/B8 outage diagnostics."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import run_p2_crossfit_copula_forward_20260906_v1 as m


def iso_max(times):
    return pd.Timestamp(times.max()).tz_convert("Asia/Seoul").isoformat() if len(times) else None


def summarize(observations, contract):
    source = observations.copy()
    source.time = pd.to_datetime(source.time, utc=True)
    public = source.loc[source.layer.isin(m.base.PUBLIC_LAYERS)].copy()
    public["available"] = np.isfinite(public.temp)
    counts = public.groupby(["station", "time"]).available.sum()
    target = source.loc[source.layer.isin([2, 3, 4])].copy()
    target = target.join(counts.rename("public_count"), on=["station", "time"])
    target["eligible"] = np.isfinite(target.temp) & target.public_count.ge(2)
    rows = []
    for fold in contract["P2"]["folds"]:
        if fold["id"] not in {"B4", "B8"}:
            continue
        start, end = m.base.utc(fold["start"]), m.base.utc(fold["end"])
        onset = end - pd.Timedelta(days=17)
        source_fold = source.loc[source.time.ge(start) & source.time.lt(end)]
        target_fold = target.loc[target.time.ge(start) & target.time.lt(end)]
        source_outage = source_fold.loc[source_fold.time.ge(onset)]
        target_outage = target_fold.loc[target_fold.time.ge(onset)]
        eligible = target_fold.loc[target_fold.eligible]
        eligible_outage = target_outage.loc[target_outage.eligible]
        rows.append(
            {
                "fold": fold["id"],
                "fold_start": fold["start"],
                "fold_end_exclusive": fold["end"],
                "fixed_outage_start": onset.tz_convert("Asia/Seoul").isoformat(),
                "source_fold_rows": len(source_fold),
                "source_fold_last_timestamp": iso_max(source_fold.time),
                "target_fold_rows": len(target_fold),
                "finite_target_fold_rows": int(np.isfinite(target_fold.temp).sum()),
                "eligible_target_fold_rows": len(eligible),
                "eligible_target_last_timestamp": iso_max(eligible.time),
                "source_outage_rows": len(source_outage),
                "target_outage_rows": len(target_outage),
                "finite_target_outage_rows": int(np.isfinite(target_outage.temp).sum()),
                "eligible_target_outage_rows": len(eligible_outage),
                "status": "NOT_ESTIMABLE_NO_ROWS"
                if not len(eligible_outage)
                else "HAS_ELIGIBLE_ROWS",
                "reason": "source_grid_ends_before_fixed_outage"
                if not len(source_outage)
                else "source_grid_exists_but_finite_supported_target_labels_absent"
                if not len(eligible_outage)
                else "supported",
                "date_moved": False,
            }
        )
    return rows


def main():
    cfg, contract, _ = m.settings()
    if (
        not (m.REPORT / "result.json").exists()
        and not (m.REPORT / "terminal-failure.json").exists()
    ):
        raise RuntimeError("metadata explanation deferred until sealed training terminal")
    source = m.install_guard()
    before = m.base.file_hash(source)
    assert before == cfg["source_sha256"]
    obs = pd.read_csv(source, usecols=["station", "time", "layer", "temp"])
    rows = summarize(obs, contract)
    assert all(row["eligible_target_outage_rows"] == 0 for row in rows)
    assert m.base.file_hash(source) == before
    report = {
        "status": "SOURCE_DATE_EXPLANATION_COMPLETE",
        "experiment_id": m.ID,
        "source_sha256_before_after": before,
        "source_rows": len(obs),
        "folds": rows,
        "source_values_printed": 0,
        "new_fits": 0,
        "official_access_rows": 0,
        "csv_written": 0,
        "upload": 0,
        "runner_sha256": m.base.file_hash(Path(__file__)),
        "source_env": "P2_DATA_DIR",
        "pid": os.getpid(),
    }
    m.save(m.REPORT / "evaluation-gap-source-metadata.json", report)
    print(json.dumps({"status": report["status"], "folds": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
