"""Independent QA entrypoint for P3 v56 real cepstral periodicity."""

from __future__ import annotations

import json

from qa_p3_cycle_generic import run_qa


def main() -> int:
    payload = run_qa(experiment_id="p3_real_cepstral_periodicity_residual_cycle_20260901_v56", feature_count=72, no_go_decision="NO_GO_ALL_REAL_CEPSTRAL_CANDIDATES", schema_version="p3.real_cepstral_periodicity_residual.independent_qa.v56")
    print(json.dumps({"decision": payload["decision"], "checks": payload["check_count"], "passed": payload["passed"], "failed": payload["failed"], "model_fits": payload["model_fits"], "official_rows": 0, "csv_materializations": 0, "uploads": 0}, ensure_ascii=False))
    return 0 if payload["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
