"""Frozen CPU model inference with only the preregistered long-lead shrink removed."""
from __future__ import annotations

import argparse
import hashlib
import json
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


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def pins():
    cfg, _ = u.settings()  # Verify the original CPU recipe without changing it.
    for name, digest in u.b.read(CODE / "noshrink-source-manifest.json").items():
        assert u.b.sha(ROOT / name) == digest, name
    assert cfg["shrink"] == {"leads": [12, 18, 24], "weight": .2}
    return cfg


def adopt():
    r = u.training_receipt()
    q = u.b.read(ROOT / "06_docs/cpudet-training-qa.json")
    assert q["status"] == "PASS" and q["training_sha256"] == u.b.sha(ROOT / "06_docs/cpudet-training.json")
    files = {n: h for n, h in r["models"].items()
             if n.startswith("03_model/full/") or n == "03_model/full_router.joblib"}
    assert len(files) == 7
    save(ROOT / "06_docs/noshrink-models.json", {
        "files": files, "columns_sha256": u.b.sha(ROOT / "04_logs/columns.json"),
        "training_receipt_sha256": u.b.sha(ROOT / "06_docs/cpudet-training.json"),
        "training_qa_sha256": u.b.sha(ROOT / "06_docs/cpudet-training-qa.json"),
        "expected_base_answer_sha256": None, "cold_status": r["run_kind"]})


class Ensemble(u.SeedEnsemble):
    def predict(self, frame, thread_count=2):
        current = frame["hs_current" if self.multi else "current_hs_for_residual"].to_numpy()
        if self.multi:
            current = current[:, None]
        return np.mean([np.clip(current+m.predict(frame, thread_count=2), 0, 30)
                        for m in self.models], axis=0)-current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["adopt-trained", "infer", "replay"])
    args = parser.parse_args()
    start = time.monotonic()
    cfg = pins()
    source = Path(os.environ["P3_DATA_DIR"]).resolve()
    u.b.access_guard(source, args.stage != "adopt-trained")
    if args.stage == "adopt-trained":
        adopt()
        return
    seal = u.b.read(ROOT / "06_docs/noshrink-models.json")
    for name, h in seal["files"].items():
        assert u.b.sha(ROOT / name) == h, name
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
              Ensemble([u.CatBoostRegressor().load_model(b) for _, b in paths], True),
              u.joblib.load(ROOT / "03_model/full_router.joblib"))
    original = u.E.apply_long_lead_persistence_shrink
    captured = {}
    def capture(routed, persistence, leads, *, config):
        assert config.weight == .2 and config.active_leads == (12, 18, 24)
        captured["routed"] = np.asarray(routed).copy()
        captured["leads"] = np.asarray(leads).copy()
        return original(routed, persistence, leads, config=config)
    u.E.apply_long_lead_persistence_shrink = capture
    keys, baseline = u.b.materializer.predict_cases(u.E, {}, cases.drop(columns="_merge"), columns, models)
    u.E.apply_long_lead_persistence_shrink = original
    candidate = captured["routed"]
    assert np.array_equal(baseline[~np.isin(captured["leads"], [12, 18, 24])],
                          candidate[~np.isin(captured["leads"], [12, 18, 24])])
    digests, payloads = {}, {}
    for name, values in [("base", baseline), ("candidate", candidate)]:
        rows = keys.assign(hs_pred=values)
        rows = index.merge(rows, on=u.b.materializer.KEYS, how="left", validate="one_to_one")
        assert rows[u.b.materializer.KEYS].equals(index)
        assert np.isfinite(rows.hs_pred).all() and rows.hs_pred.between(0, 30).all()
        payloads[name] = rows.to_csv(index=False, lineterminator="\n").encode()
        digests[name] = hashlib.sha256(payloads[name]).hexdigest()
    expected = seal["expected_base_answer_sha256"]
    if expected:
        assert digests["base"] == expected, "base inference differs; addon cannot attribute change"
    answer = ROOT / "05_answer/submission_p3_numeric_cpudet_noshrink.csv"
    if args.stage == "infer":
        with answer.open("xb") as stream:
            stream.write(payloads["candidate"])
    else:
        before = u.b.read(ROOT / "06_docs/noshrink-answer.json")
        assert before["pid"] != os.getpid() and before["candidate_sha256"] == digests["candidate"]
        assert u.b.sha(answer) == digests["candidate"]
    assert hashes == {n: u.b.sha(source / n) for n in hashes}
    for name, h in seal["files"].items():
        assert u.b.sha(ROOT / name) == h
    r = {"status": "EXACT_REPLAY_PASS" if args.stage == "replay" else "LOCAL_NOT_UPLOADED",
         "pid": os.getpid(), "rows": 1200, "bytes": len(payloads["candidate"]),
         "candidate_sha256": digests["candidate"], "base_sha256": digests["base"],
         "base_exact_reconstruction": bool(expected and expected == digests["base"]),
         "short_leads_exact_unchanged": True, "changed_rows": int(np.sum(candidate != baseline)),
         "model_manifest_sha256": u.b.sha(ROOT / "06_docs/noshrink-models.json"),
         "input_sha256": hashes, "inference_threads": 2, "seconds": time.monotonic()-start,
         "new_fits": 0, "old_answer_values_read": 0, "hidden_rows": 0, "uploads": 0,
         "base_cold_status": seal["cold_status"], "fresh_two_cold_proof": False}
    save(ROOT / ("06_docs/noshrink-replay.json" if args.stage == "replay" else "06_docs/noshrink-answer.json"), r)
    print(json.dumps(r))


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
