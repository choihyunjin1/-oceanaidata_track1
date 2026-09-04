"""Append-only r3 correction of the full-cycle interpretation.

The r2 charts, quantitative datasets, metrics, fold comparisons, model/candidate
identities, existing sources, and operational receipts remain unchanged.  A new
independent meaningful-improvement audit controls the narrower conclusion:
point-estimate improvement is not confirmed meaningful generalization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

KST = timezone(timedelta(hours=9))
BASE_SCRIPT = Path(__file__).with_name("build_full_improvement_cycle_report_v2.py")
SPEC = importlib.util.spec_from_file_location("build_full_improvement_cycle_report_v2", BASE_SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import bootstrap guard
    raise RuntimeError(f"Could not import r2 builder: {BASE_SCRIPT}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

REPORT_TITLE = "Full Improvement Cycle — confirmed meaningful improvements: 0/3"
REPORT_ID = "full-improvement-cycle-2026-08-23-r3"
DEFAULT_OUTPUT = Path("reports/generated/full_improvement_cycle_2026-08-23_r3/artifact.json")
R2_ARTIFACT = Path("reports/generated/full_improvement_cycle_2026-08-22_r2/artifact.json")
EXPECTED_R2_ARTIFACT_SHA256 = "599526473805684067b02e7a282f42ee0aa8a72dfdf07bae2ecfabd63254ce5c"
EXPECTED_REGISTRY_SHA256 = "69fd2896a3df8626df6c80996683309236ad2953aa422beb7f9b0a99a157327c"

AUDIT_RELATIVE_PATH = Path(
    "artifacts/full_improvement_cycle_20260822/meaningful_improvement_audit.json"
)
EXPECTED_AUDIT_SHA256 = "cb5f60227cb7d7dfe8adca63ca5c300b82689ee00f5c1a5f5de146b7b9039427"
AUDIT_SOURCE_ID = "meaningful_improvement_audit"
SYNTHESIS_SOURCE_ID = "meaningful_decision_synthesis"

DECISION_EVIDENCE = {
    "P1": (
        "FAIL — confirmed meaningful improvement; CI90 includes zero, effect is tiny, and the "
        "119-row strict-clock caveat limits the validation claim; research_only / do_not_promote"
    ),
    "P2": (
        "PROMISING LOCAL ONLY / FAIL CONFIRMED — strong effect and CI on the same corrected OOF, "
        "but adaptive/repeatedly exposed, inner candidate remains above baseline, and 1/3 folds "
        "regress; research_only / do_not_promote"
    ),
    "P3": (
        "FAIL — confirmed meaningful improvement; CI90 includes zero and effect is tiny; "
        "research_only / do_not_promote"
    ),
}


def _read_and_pin_audit(root: Path) -> dict[str, Any]:
    base.base._require(not AUDIT_RELATIVE_PATH.is_absolute(), "absolute audit path forbidden")
    base.base._require(".." not in AUDIT_RELATIVE_PATH.parts, "audit path traversal forbidden")
    base.base._require(AUDIT_RELATIVE_PATH.suffix.lower() == ".json", "audit must be JSON")
    base.base._require(
        re.fullmatch(r"[0-9a-f]{64}", EXPECTED_AUDIT_SHA256) is not None
        and EXPECTED_AUDIT_SHA256 != "0" * 64,
        "meaningful-improvement audit SHA is not finalized",
    )
    path = root / AUDIT_RELATIVE_PATH
    base.base._require(
        path.is_file(), f"meaningful-improvement audit missing: {AUDIT_RELATIVE_PATH}"
    )
    actual = base.base._sha256(path)
    base.base._require(
        actual == EXPECTED_AUDIT_SHA256,
        f"meaningful-improvement audit SHA drifted: {actual}",
    )
    audit = base.base._read_json(path)
    _validate_audit(audit)
    return audit


def _validate_audit(audit: dict[str, Any]) -> None:
    base.base._require(
        audit["schema_version"] == "meaningful_improvement_audit.v1", "audit schema drifted"
    )
    base.base._require(
        audit["status"]
        == "COMPLETE__CONFIRMED_MEANINGFUL_GENERALIZATION_IMPROVEMENTS_0_OF_3__RESEARCH_ONLY__DO_NOT_PROMOTE",
        "audit status drifted",
    )
    summary = audit["summary"]
    base.base._require(
        summary["candidate_point_estimate_improvement_count"] == 3,
        "point-estimate count drifted",
    )
    base.base._require(
        summary["confirmed_meaningful_generalization_improvement_count"] == 0,
        "confirmed count drifted",
    )
    base.base._require(
        summary["failed_confirmatory_problems"] == ["P1", "P2", "P3"],
        "confirmatory failure list drifted",
    )
    base.base._require(
        summary["nominal_ci90_excludes_zero_problems"] == ["P2"],
        "nominal CI list drifted",
    )
    for problem in ("P1", "P2", "P3"):
        record = audit["problems"][problem]
        base.base._require(record["overall_verdict"] == "FAIL", f"{problem} verdict drifted")
        base.base._require(
            record["confirmed_meaningful_generalization_improvement"] is False,
            f"{problem} confirmation drifted",
        )
        base.base._require(
            record["candidate_status_recommendation"] == "research_only",
            f"{problem} research status drifted",
        )
        base.base._require(record["do_not_promote"] is True, f"{problem} promotion drifted")
    p1 = audit["problems"]["P1"]
    p2 = audit["problems"]["P2"]
    p3 = audit["problems"]["P3"]
    base.base._require(
        p1["criteria"]["b_practical_effect"]["verdict"] == "FAIL", "P1 effect gate drifted"
    )
    base.base._require(
        p1["criteria"]["c_uncertainty_excludes_no_effect"]["verdict"] == "FAIL",
        "P1 CI gate drifted",
    )
    base.base._require(
        p2["criteria"]["b_practical_effect"]["verdict"] == "PASS", "P2 effect gate drifted"
    )
    base.base._require(
        p2["criteria"]["c_uncertainty_excludes_no_effect"]["verdict"] == "FAIL",
        "P2 CI gate drifted",
    )
    base.base._require(
        p2["criteria"]["d_fresh_or_honest_nested_evaluation"]["verdict"] == "FAIL",
        "P2 exposure gate drifted",
    )
    base.base._require(
        p3["criteria"]["b_practical_effect"]["verdict"] == "FAIL", "P3 effect gate drifted"
    )
    base.base._require(
        p3["criteria"]["c_uncertainty_excludes_no_effect"]["verdict"] == "FAIL",
        "P3 CI gate drifted",
    )
    serialized = json.dumps(audit, ensure_ascii=False)
    source_pins = {item["path"]: item["sha256"] for item in audit["source_pins"]}
    base.base._require(
        source_pins[base.base.DEFAULT_REGISTRY.as_posix()] == EXPECTED_REGISTRY_SHA256,
        "audit registry pin drifted",
    )
    for source_id in ("p1_metrics", "p2_metrics", "p3_metrics"):
        path = base.base.RELATIVE_PATHS[source_id].as_posix()
        base.base._require(
            source_pins[path] == base.base.EXPECTED_SHA256[source_id],
            f"audit aggregate pin drifted: {source_id}",
        )
    for frozen_sha in base.base.FROZEN_SHA256.values():
        base.base._require(frozen_sha in serialized, f"audit frozen pin missing: {frozen_sha}")
    base.base._require("0.8603708380408055" in serialized, "P1 baseline missing from audit")
    base.base._require("1.042512377552349" in serialized, "P2 candidate metric missing from audit")
    base.base._require("0.7786608799293823" in serialized, "P3 candidate metric missing from audit")


def collect_inputs(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    evidence, registry, r1 = base.collect_inputs(root)
    base.base._require(
        base.EXPECTED_REGISTRY_SHA256 == EXPECTED_REGISTRY_SHA256,
        "r2/registry pin contract drifted",
    )
    r2_path = root / R2_ARTIFACT
    base.base._require(r2_path.is_file(), "r2 artifact is missing")
    r2_actual = base.base._sha256(r2_path)
    base.base._require(r2_actual == EXPECTED_R2_ARTIFACT_SHA256, f"r2 SHA drifted: {r2_actual}")
    r2 = base.base._read_json(r2_path)
    reconstructed_r2 = base.build_artifact(r1)
    base.base._require(reconstructed_r2 == r2, "reconstructed r2 artifact drifted")
    audit = _read_and_pin_audit(root)
    return evidence, registry, r2, audit


def _block(artifact: dict[str, Any], block_id: str) -> dict[str, Any]:
    return next(block for block in artifact["manifest"]["blocks"] if block["id"] == block_id)


def _update_summary_table(artifact: dict[str, Any]) -> None:
    rows = artifact["snapshot"]["datasets"]["full_cycle_exact_registry"]
    base.base._require([row["problem"] for row in rows] == ["P1", "P2", "P3"], "table rows drifted")
    for row in rows:
        row["evidence"] = DECISION_EVIDENCE[row["problem"]]
    table = artifact["manifest"]["tables"][0]
    fields = [
        "sequence",
        "problem",
        "metric_unit",
        "baseline",
        "candidate",
        "signed_delta",
        "ci90",
        "ci_zero",
        "evidence",
        "full_fit",
        "candidate_sha256",
        "frozen_unchanged",
        "uploads",
    ]
    table["title"] = "Exact full-cycle results and meaningful-improvement decisions"
    table["subtitle"] = (
        "Point estimates and full-fit provenance preserved; confirmed meaningful improvements: 0/3"
    )
    table["sourceId"] = SYNTHESIS_SOURCE_ID
    table["source"]["query"]["sql"] = base.base._rows_to_union_sql(rows, fields)
    table["source"]["query"]["description"] = (
        "Exact unit-specific results joined to the independent meaningful-improvement decision audit."
    )
    table["source"]["query"]["tables_used"] = [AUDIT_RELATIVE_PATH.as_posix()]
    table["source"]["query"]["filters"] = [
        "P1/P2/P3",
        "confirmed meaningful improvement decision",
        "raw rows excluded",
    ]
    table["source"]["query"]["metric_definitions"]["evidence"] = (
        "independent decision status; point-estimate improvement alone is insufficient"
    )


def build_artifact(
    r2: dict[str, Any], audit: dict[str, Any], *, generated_at: str
) -> dict[str, Any]:
    _validate_audit(audit)
    artifact = copy.deepcopy(r2)
    manifest = artifact["manifest"]
    manifest["title"] = REPORT_TITLE
    manifest["description"] = (
        "Technical correction: zero of three point-estimate improvements are confirmed meaningful "
        "generalization improvements"
    )
    manifest["generatedAt"] = generated_at
    artifact["snapshot"]["generatedAt"] = generated_at
    _block(artifact, "title")["body"] = f"# {REPORT_TITLE}"

    technical_summary = _block(artifact, "technical_summary")
    technical_summary["sourceId"] = AUDIT_SOURCE_ID
    technical_summary["body"] = (
        "## Technical Summary — confirmed meaningful improvements: 0/3\n\n"
        "세 문제 모두 실제 full fit과 candidate 생성까지 완료했고 local point estimate는 좋아졌다. "
        "그러나 **점추정 개선은 유의미한 일반화 개선의 확인이 아니다. 최종 판정은 0/3이다.** "
        "P1과 P3는 작은 효과의 CI90이 0을 포함해 FAIL이다. P2는 같은 corrected OOF에서 효과와 "
        "CI가 강해 promising local only이지만, adaptive/repeatedly exposed surface, inner candidate의 "
        "baseline 초과, 1/3 fold regression 때문에 confirmed meaningful improvement에는 FAIL이다. "
        "세 candidate는 모두 `research_only / do_not_promote`이며 current frozen 세 파일을 유지하고 "
        "upload는 0으로 둔다. 이 r3는 r2의 ‘개선 완료’ 해석을 명시적으로 supersede한다."
    )

    key_findings = _block(artifact, "key_findings")
    key_findings["sourceId"] = SYNTHESIS_SOURCE_ID
    key_findings["body"] = (
        "## Key Findings — local movement는 있었지만 confirmed meaningful gain은 없었다\n\n"
        "- **P1 — FAIL:** F1 0.8603708380 → 0.8609416446, Δ +0.0005708065로 효과가 작고 "
        "CI90 [-0.0010006259, +0.0019853049]가 0을 포함한다. 또한 event-protected Q3 I-ORS "
        "layer-1 positive-event tail 119 rows가 global Q4 earliest timestamp 뒤까지 남는 strict-clock "
        "caveat가 있다. 그 119 rows를 calibration에서 제외해도 Q4 params 0.75/0.15와 predictions는 "
        "불변이지만, globally strict wall-clock generalization 근거는 아니다.\n"
        "- **P2 — PROMISING LOCAL ONLY / FAIL CONFIRMED:** RMSE 1.1158878560 → 1.0425123776°C, "
        "Δ -0.0733754784°C이고 CI90 [-0.1016906444, -0.0446891572]로 같은 corrected OOF에서는 "
        "강하다. 그러나 adaptive/repeatedly exposed evaluation이며 inner candidate는 baseline보다 "
        "+0.0276402153°C 높고 Sep–Oct fold는 +0.0770355591°C 악화했다.\n"
        "- **P3 — FAIL:** 유의파고(hs) RMSE 0.7791048400 → 0.7786608799m, "
        "Δ -0.0004439600m로 작고 CI90 [-0.0020481402, +0.0011163551]가 0을 포함한다.\n\n"
        "세 signed chart는 unit별 local movement를 정확히 보존할 뿐 meaningful generalization을 "
        "증명하지 않는다. 최종 decision은 아래 exact table의 evidence status다."
    )

    scope = _block(artifact, "scope_definitions")
    scope["sourceId"] = AUDIT_SOURCE_ID
    scope["body"] = (
        "## Scope, Data, and Metric Definitions\n\n"
        "`Point-estimate improvement`는 지정된 local validation surface에서 candidate metric이 baseline보다 "
        "좋았다는 기술적 사실이다. `Confirmed meaningful improvement`는 효과 크기·불확실성·독립성·"
        "fold/inner robustness를 함께 통과해 일반화 개선으로 승격할 수 있는 상태다. 이 r3에서는 후자가 "
        "0/3이다. P1은 outer micro F1(클수록 좋음), P2는 fold-equal official-layer-weighted RMSE(°C, "
        "작을수록 좋음), P3는 181-case corrected OOF의 유의파고(hs) RMSE(m, 작을수록 좋음)다. "
        "모든 Δ는 candidate − baseline이다. `research_only / do_not_promote`는 candidate를 보존하되 "
        "official pool/current frozen을 바꾸거나 upload하지 않는다는 뜻이다."
    )

    limitations = _block(artifact, "limitations")
    limitations["sourceId"] = SYNTHESIS_SOURCE_ID
    limitations["body"] = (
        "## Limitations, Uncertainty, and Robustness Checks\n\n"
        "P1과 P3는 CI90이 0을 포함하고 효과가 작아 confirmed meaningful improvement가 아니다. P1은 "
        "fold order와 station-layer chronology는 안전하지만 119-row boundary 때문에 globally strict "
        "wall-clock label scope라는 주장을 할 수 없다. P2는 CI가 0을 배제하더라도 같은 corrected OOF를 "
        "adaptive하게 반복 노출한 결과다. inner candidate가 baseline보다 높고 한 fold가 악화했으므로 "
        "independent generalization confirmation으로 승격할 수 없다. 세 chart의 bar와 CI는 local evidence "
        "범위만 나타낸다. hidden-test absolute calibration이나 leaderboard gain은 식별하지 못한다."
    )

    next_steps = _block(artifact, "next_steps")
    next_steps["sourceId"] = AUDIT_SOURCE_ID
    next_steps["body"] = (
        "## Recommended Next Steps — freeze 유지, promotion 중단\n\n"
        "1. P1·P2·P3 candidate를 모두 `research_only / do_not_promote`로 등록한다.\n"
        "2. 세 current frozen SHA를 그대로 유지하고 official pool 변경과 upload를 수행하지 않는다.\n"
        "3. P1·P3는 새 independent labeled surface에서 CI-zero crossing과 effect size를 다시 평가한다.\n"
        "4. P2는 untouched seasonal block을 사전 고정해 inner-baseline 열세와 Sep–Oct regression을 함께 "
        "검증한다. 같은 corrected OOF 재사용 결과는 confirmation으로 세지 않는다.\n"
        "5. 향후 promotion은 독립 surface와 사전 정의된 stop rule을 모두 통과한 뒤 별도 승인한다."
    )

    _update_summary_table(artifact)
    audit_source = {
        "id": AUDIT_SOURCE_ID,
        "label": "Independent meaningful-improvement decision audit",
        "path": AUDIT_RELATIVE_PATH.as_posix(),
        "sha256": EXPECTED_AUDIT_SHA256,
        "note": (
            "Decision authority for r3: confirmed meaningful improvements 0/3; all candidates "
            "research_only/do_not_promote; current frozen retained."
        ),
    }
    manifest["sources"].append(audit_source)
    synthesis_source = {
        "id": SYNTHESIS_SOURCE_ID,
        "label": "Meaningful-decision synthesis with strict-clock attestation",
        "note": (
            "Decision status comes from the SHA-pinned meaningful-improvement audit. The P1 119-row "
            "strict-clock boundary comes from the existing message-only full-cycle independent QA "
            "attestation; no receipt was created or invented."
        ),
    }
    manifest["sources"].append(synthesis_source)
    artifact["sources"].append(
        {
            "id": audit_source["id"],
            "label": audit_source["label"],
            "path": audit_source["path"],
            "sha256": audit_source["sha256"],
        }
    )
    artifact["sources"].append({"id": synthesis_source["id"], "label": synthesis_source["label"]})
    artifact["package_info"]["originUrl"] = f"artifact://{REPORT_ID}"
    _validate_correction(artifact, r2, generated_at=generated_at)
    return artifact


def _validate_correction(
    artifact: dict[str, Any], r2: dict[str, Any], *, generated_at: str
) -> None:
    manifest = artifact["manifest"]
    serialized = json.dumps(artifact, ensure_ascii=False)
    for phrase in (
        "confirmed meaningful improvements: 0/3",
        "P1 — FAIL",
        "P2 — PROMISING LOCAL ONLY / FAIL CONFIRMED",
        "P3 — FAIL",
        "research_only / do_not_promote",
        "r3는 r2",
        "119-row",
        "adaptive/repeatedly exposed",
        "current frozen",
        "upload는 0",
    ):
        base.base._require(phrase in serialized, f"r3 decision phrase missing: {phrase}")

    base.base._require(manifest["charts"] == r2["manifest"]["charts"], "charts changed in r3")
    for dataset_id in (
        "p1_signed_validation_delta",
        "p2_signed_validation_delta",
        "p3_signed_validation_delta",
    ):
        base.base._require(
            artifact["snapshot"]["datasets"][dataset_id] == r2["snapshot"]["datasets"][dataset_id],
            f"chart dataset changed: {dataset_id}",
        )
    old_rows = r2["snapshot"]["datasets"]["full_cycle_exact_registry"]
    new_rows = artifact["snapshot"]["datasets"]["full_cycle_exact_registry"]
    for old, new in zip(old_rows, new_rows, strict=True):
        normalized = dict(new)
        normalized["evidence"] = old["evidence"]
        base.base._require(normalized == old, f"non-decision table field changed: {old['problem']}")
        base.base._require(
            new["evidence"] == DECISION_EVIDENCE[new["problem"]], "decision cell drifted"
        )

    unchanged_blocks = {
        "full_cycle_exact_table",
        "p1_signed_chart",
        "p2_signed_chart",
        "p3_signed_chart",
        "methodology",
        "operation_failures_resume",
        "further_questions",
        "source_method_note",
    }
    for block_id in unchanged_blocks:
        base.base._require(
            _block(artifact, block_id) == _block(r2, block_id), f"block changed: {block_id}"
        )
    base.base._require(manifest["generatedAt"] == generated_at, "manifest timestamp drifted")
    base.base._require(
        artifact["snapshot"]["generatedAt"] == generated_at, "snapshot timestamp drifted"
    )
    base.base._require(
        len(manifest["sources"]) == len(r2["manifest"]["sources"]) + 2, "source count drifted"
    )
    base.base._require(
        manifest["sources"][:-2] == r2["manifest"]["sources"], "existing sources changed"
    )
    audit_source = manifest["sources"][-2]
    base.base._require(audit_source["id"] == AUDIT_SOURCE_ID, "audit source id drifted")
    base.base._require(audit_source["sha256"] == EXPECTED_AUDIT_SHA256, "audit source SHA drifted")
    base.base._require(audit_source["path"] == AUDIT_RELATIVE_PATH.as_posix(), "audit path drifted")
    base.base._require(
        manifest["sources"][-1]["id"] == SYNTHESIS_SOURCE_ID, "synthesis source drifted"
    )
    base.base._require(
        artifact["package_info"]["originUrl"] == f"artifact://{REPORT_ID}", "origin drifted"
    )

    source_ids = {source["id"] for source in manifest["sources"]}
    for block in manifest["blocks"]:
        if "sourceId" in block:
            base.base._require(
                block["sourceId"] in source_ids, f"missing block source: {block['id']}"
            )
    for visual in [*manifest["charts"], *manifest["tables"]]:
        base.base._require(
            visual["sourceId"] in source_ids, f"missing visual source: {visual['id']}"
        )
    table = manifest["tables"][0]
    declared = {column["field"] for column in table["columns"]}
    base.base._require(table["defaultSort"]["field"] in declared, "table sort field undeclared")
    base.base._require(table["sourceId"] == SYNTHESIS_SOURCE_ID, "decision table source drifted")
    for forbidden in ("C:/Users/", "C:\\Users\\", "api_key", "access_token", "password"):
        base.base._require(
            forbidden.lower() not in serialized.lower(), f"unsafe r3 content: {forbidden}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generated-at", default=None)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    base.base._require(
        output.resolve() == (root / DEFAULT_OUTPUT).resolve(), "r3 output path is frozen"
    )
    _, _, r2, audit = collect_inputs(root)
    generated_at = args.generated_at or datetime.now(KST).isoformat()
    artifact = build_artifact(r2, audit, generated_at=generated_at)
    artifact_sha = hashlib.sha256(base.base._canonical_bytes(artifact)).hexdigest()
    if args.check_only:
        print(
            f"PASS: validated r3 meaningful-improvement correction ({artifact_sha}); no files written"
        )
        return 0
    base.base._write_new(output, artifact)
    print(f"PASS: wrote r3 canonical artifact to {DEFAULT_OUTPUT.as_posix()} ({artifact_sha})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
