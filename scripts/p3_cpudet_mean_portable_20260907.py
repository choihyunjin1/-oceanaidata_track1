"""Portable equal-component ablation; original CPU training recipe remains pinned."""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import unbounded as u  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

ROOT = CODE.parent


class EqualRouter:
    def predict_weights(self, frame):
        return np.tile([0.5, 0.5, 0.0], (len(frame), 1))


class Ensemble(u.SeedEnsemble):
    def predict(self, frame, thread_count=2):
        current = frame["hs_current" if self.multi else "current_hs_for_residual"].to_numpy()
        if self.multi:
            current = current[:, None]
        return np.mean([np.clip(current + m.predict(frame, thread_count=2), 0, 30)
                        for m in self.models], axis=0) - current


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["adopt-trained", "infer", "replay"])
    args = p.parse_args()
    start = time.monotonic()
    cfg, _ = u.settings()
    for name, digest in u.b.read(CODE / "mean-source-manifest.json").items():
        assert u.b.sha(ROOT / name) == digest, name
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    u.b.access_guard(source, args.stage != "adopt-trained")
    if args.stage == "adopt-trained":
        receipt = u.training_receipt()
        qa = u.b.read(ROOT / "06_docs/cpudet-training-qa.json")
        assert qa["status"] == "PASS"
        assert qa["training_sha256"] == u.b.sha(ROOT / "06_docs/cpudet-training.json")
        files = {k: v for k, v in receipt["models"].items() if k.startswith("03_model/full/")}
        assert len(files) == 6
        u.b.save(ROOT / "06_docs/mean-models.json", {"files": files,
                 "columns_sha256": u.b.sha(ROOT / "04_logs/columns.json"),
                 "training_sha256": u.b.sha(ROOT / "06_docs/cpudet-training.json")})
        return
    seal = u.b.read(ROOT / "06_docs/mean-models.json")
    for name, digest in seal["files"].items():
        assert u.b.sha(ROOT / name) == digest, name
    assert u.b.sha(ROOT / "04_logs/columns.json") == seal["columns_sha256"]
    columns = u.b.read(ROOT / "04_logs/columns.json")["columns"]
    assert len(columns) == 591
    hashes = {n: u.b.sha(source / n) for n in ["test_context.parquet", "test_index.csv"]}
    context = pd.read_parquet(source / "test_context.parquet",
        columns=["case_id", "station", "step_minute", *u.E.BASE_COLUMNS, *u.E.DIRECTION_COLUMNS])
    index = pd.read_csv(source / "test_index.csv", usecols=u.b.materializer.KEYS)
    assert len(context) == 57800 and len(index) == 1200 and not index.duplicated().any()
    records = []
    for case_id, group in context.groupby("case_id", sort=False):
        group = group.sort_values("step_minute")
        assert len(group) == 289 and group.station.nunique() == 1
        assert np.array_equal(group.step_minute, np.arange(-2880, 1, 10))
        records.append({"case_id": case_id, "station": str(group.station.iloc[0]),
                        **u.E.summarize_context(group)})
    cases = index[["case_id", "station"]].drop_duplicates().merge(pd.DataFrame(records),
        on=["case_id", "station"], how="outer", validate="one_to_one", indicator=True)
    assert len(cases) == 200 and cases._merge.eq("both").all()
    paths = u.paths("full", cfg)
    models = (Ensemble([u.CatBoostRegressor().load_model(a) for a, _ in paths], False),
              Ensemble([u.CatBoostRegressor().load_model(b) for _, b in paths], True), EqualRouter())
    # Unchanged materializer: fixed short-lead weights, routing, then original long shrink.
    keys, values = u.b.materializer.predict_cases(u.E, {}, cases.drop(columns="_merge"), columns, models)
    rows = index.merge(keys.assign(hs_pred=values), on=u.b.materializer.KEYS, how="left", validate="one_to_one")
    assert rows[u.b.materializer.KEYS].equals(index)
    assert np.isfinite(rows.hs_pred).all() and rows.hs_pred.between(0, 30).all()
    payload = rows.to_csv(index=False, lineterminator="\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    answer = ROOT / "05_answer/submission_p3_cpudet_mean.csv"
    if args.stage == "infer":
        with answer.open("xb") as stream:
            stream.write(payload)
    else:
        prior = u.b.read(ROOT / "06_docs/mean-answer.json")
        assert prior["pid"] != os.getpid() and prior["sha256"] == digest
        assert u.b.sha(answer) == digest
    assert hashes == {n: u.b.sha(source / n) for n in hashes}
    for name, expected in seal["files"].items():
        assert u.b.sha(ROOT / name) == expected
    receipt = {"status": "EXACT_REPLAY_PASS" if args.stage == "replay" else "ANSWER_QA_PASS",
        "pid": os.getpid(), "rows": 1200, "bytes": len(payload), "sha256": digest,
        "seconds": time.monotonic() - start, "new_fits": 0, "hidden_rows": 0,
        "old_answer_values_read": 0, "input_sha256": hashes, "inference_threads": 2,
        "full_fresh_cold_proof": "pending base fresh_cold_2", "public_coefficients": 0}
    u.b.save(ROOT / ("06_docs/mean-replay.json" if args.stage == "replay" else "06_docs/mean-answer.json"), receipt)
    print(receipt)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
