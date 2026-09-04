from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.p1_qc.logit_shrunk_label_shift import (  # noqa: E402
    correct_to_prior,
    shrink_lambda,
    shrunk_target_prevalence,
)
from src.p1_qc.prequential_label_shift_em import (  # noqa: E402
    frozen_logit_matrix,
    label_shift_em,
)

CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v31.json"
V28_CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v28.json"
GATE_V4 = ROOT / "configs/goals/p1_prospective_transport_gate_20260831_v4.json"
CALIBRATION = ROOT / "reports/public_transport_calibration_20260831_v3/calibration.json"
REPORT = ROOT / "reports/p1_public_transport_repair_cycle_20260831_v31/preflight-report.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract() -> dict:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    old = json.loads(V28_CONFIG.read_text(encoding="utf-8"))
    checks = {"C": cfg["model"]["C"] == old["model"]["C"] == 0.1, "solver": cfg["model"]["solver"] == old["model"]["solver"], "em_iter": cfg["em"]["maximum_iterations"] == old["outer_label_shift"]["maximum_iterations"] == 200, "fits": cfg["model"]["maximum_fits"] == 2, "v3_raw": cfg["decision_policy"]["minimum_raw_expected_point_delta_inclusive"] == 0.015383691373120248, "v4_hash": cfg["prospective_gate_v4"]["sha256"] == sha256(GATE_V4), "v4_day_diagnostic": cfg["safety"]["maximum_changed_fraction_any_kst_day"] == "diagnostic_only", "v4_group_diagnostic": cfg["safety"]["minimum_each_supported_station_layer_delta_f1"] == "diagnostic_only", "history_off": cfg["authorization"]["historical_execution"] is False, "lock_off": cfg["authorization"]["attempt_lock_creation"] is False, "calibration": cfg["transport"]["calibration_sha256"] == sha256(CALIBRATION)}
    if not all(checks.values()):
        raise RuntimeError(f"v31 contract mismatch: {checks}")
    return cfg


def preflight() -> dict:
    cfg = load_contract()
    rng = np.random.default_rng(20260931)
    rows = 1200
    latent = rng.normal(size=rows)
    labels = (latent + rng.normal(scale=0.8, size=rows) > 1.7).astype(np.int8)
    sources = []
    for scale, noise in ((1.0, 1.0), (0.8, 1.2), (1.25, 0.9)):
        score = scale * latent + rng.normal(scale=noise, size=rows) - 2.0
        sources.append(1.0 / (1.0 + np.exp(-score)))
    design = frozen_logit_matrix(*sources)
    model = LogisticRegression(C=0.1, solver="lbfgs", max_iter=500, tol=1e-8)
    model.fit(design[:900], labels[:900])
    source_prevalence = float(np.clip(labels[900:].mean(), 1e-6, 1 - 1e-6))
    target_design = frozen_logit_matrix(*(np.clip(value[900:] * 0.8, 1e-6, 1 - 1e-6) for value in sources))
    source_probability = model.predict_proba(target_design)[:, 1]
    em_probability, em = label_shift_em(source_probability, source_prevalence, maximum_iterations=200, tolerance=1e-10, epsilon=1e-6)
    observed = float(np.clip(labels[900:].mean() * 0.9, 1e-6, 1 - 1e-6))
    shrink = shrink_lambda(source_prevalence, em.target_prevalence, observed)
    target = shrunk_target_prevalence(source_prevalence, em.target_prevalence, shrink)
    corrected = correct_to_prior(source_probability, source_prevalence, target)
    checks = {"em_converged": em.converged, "shrink_bounded": 0 <= shrink <= 1, "target_between_source_and_em": min(source_prevalence, em.target_prevalence) - 1e-12 <= target <= max(source_prevalence, em.target_prevalence) + 1e-12, "finite": bool(np.isfinite(corrected).all()), "prediction_differs_from_full_em": bool(not np.array_equal(corrected, em_probability)), "historical_zero": cfg["authorization"]["historical_execution"] is False, "lock_zero": cfg["authorization"]["attempt_lock_creation"] is False, "official_zero": True}
    return {"schema_version": "p1.v31.preflight.1", "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "synthetic": {"lambda": shrink, "source_prevalence": source_prevalence, "em_target": em.target_prevalence, "shrunk_target": target}, "hashes": {"config_sha256": sha256(CONFIG), "source_sha256": sha256(ROOT / "src/p1_qc/logit_shrunk_label_shift.py"), "v28_config_sha256": sha256(V28_CONFIG), "gate_v4_sha256": sha256(GATE_V4), "calibration_sha256": sha256(CALIBRATION)}, "access": {"historical_truth_reads": 0, "locks": 0, "official_reads": 0, "hidden_truth_reads": 0, "csv": 0, "uploads": 0}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        raise SystemExit("historical execution is not authorized")
    if not args.preflight:
        raise SystemExit("only --preflight is authorized")
    result = preflight()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
