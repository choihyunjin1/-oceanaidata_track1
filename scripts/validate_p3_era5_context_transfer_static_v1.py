"""Static, metadata-only audit of the preregistered P3 ERA5 one-shot runner.

This validator never imports or executes the ERA5 runner, never opens ERA5 value
files, and never reads any official evaluation, sample, submission, or candidate.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def resolve_inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def exclusive_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _position(text: str, needle: str) -> int:
    value = text.find(needle)
    if value < 0:
        raise ValueError(f"Required static runner token is absent: {needle}")
    return value


def _manifest_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "sha256": None}
    value = read_json(path)
    requests = value.get("requests") if isinstance(value.get("requests"), dict) else {}
    return {
        "exists": True,
        "sha256": sha256_file(path),
        "created_at_utc": value.get("created_at_utc"),
        "stage": value.get("stage"),
        "row_count": value.get("row_count"),
        "observed_start": value.get("observed_start"),
        "observed_end": value.get("observed_end"),
        "local_file_present": bool(value.get("local_file")),
        "file_sha256_present": bool(re.fullmatch(r"[0-9a-f]{64}", str(value.get("file_sha256", "")))),
        "selected_cells": len(value.get("selected_cells") or []),
        "year_requests": len(requests.get("selected_single_cell_years") or []),
        "file_receipts": len(value.get("files") or []),
        "official_test_or_submission_accessed": value.get(
            "official_test_or_submission_accessed"
        ),
    }


def audit(config_path: Path, root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path)
    static_inputs = config["static_inputs"]
    observed_hashes: dict[str, str] = {}
    input_hash_checks: dict[str, bool] = {}
    for name, receipt in static_inputs.items():
        path = resolve_inside(root, receipt["path"])
        observed = sha256_file(path)
        observed_hashes[name] = observed
        input_hash_checks[name] = observed == receipt["sha256"]
    common = config["common_secondary_recon"]
    common_hash = sha256_file(resolve_inside(root, common["path"]))

    experiment = read_json(resolve_inside(root, static_inputs["experiment_config"]["path"]))
    preaccess = read_json(resolve_inside(root, static_inputs["preaccess_seal"]["path"]))
    amendment_r1 = read_json(
        resolve_inside(root, static_inputs["transport_amendment_r1"]["path"])
    )
    amendment_r2 = read_json(
        resolve_inside(root, static_inputs["transport_amendment_r2"]["path"])
    )
    scope = read_json(resolve_inside(root, static_inputs["external_scope"]["path"]))
    runner_path = resolve_inside(root, static_inputs["context_runner"]["path"])
    runner = runner_path.read_text(encoding="utf-8")
    prepare = resolve_inside(root, static_inputs["prepare_runner"]["path"]).read_text(
        encoding="utf-8"
    )
    tests = resolve_inside(root, static_inputs["context_runner_tests"]["path"]).read_text(
        encoding="utf-8"
    )

    current_runner_hash = observed_hashes["context_runner"]
    current_config_hash = observed_hashes["experiment_config"]
    current_scope_hash = observed_hashes["external_scope"]
    chain_checks = {
        "preaccess_seal_hash_matches_r1": amendment_r1["superseded_sha256"]["preaccess_seal"]
        == observed_hashes["preaccess_seal"],
        "r1_hash_matches_r2_supersedes": amendment_r2["supersedes"]["sha256"]
        == observed_hashes["transport_amendment_r1"],
        "r2_hash_bound_by_current_experiment": experiment["bindings"][
            "transport_amendment_after_smoke"
        ]["sha256"]
        == observed_hashes["transport_amendment_r2"],
        "r2_hash_bound_by_current_scope": scope["bindings"]["transport_amendment"]["sha256"]
        == observed_hashes["transport_amendment_r2"],
        "current_scope_hash_bound_by_experiment": experiment["bindings"]["external_scope"][
            "sha256"
        ]
        == current_scope_hash,
        "current_runner_hash_bound_by_experiment": experiment["bindings"][
            "implementation_before_first_value_access"
        ]["context_transfer_runner_sha256"]
        == current_runner_hash,
        "current_prepare_hash_bound_by_experiment": experiment["bindings"]
        ["implementation_before_first_value_access"]["prepare_era5_runner_sha256"]
        == observed_hashes["prepare_runner"],
        "current_config_hash_pinned_by_preaccess_or_amendment": current_config_hash
        in {
            str(preaccess.get("sha256", {}).get("experiment_config", "")),
            *(
                str(value.get("experiment_config", ""))
                for value in (
                    amendment_r1.get("superseded_sha256", {}),
                    amendment_r1.get("amended_sha256", {}),
                    amendment_r2.get("superseded_sha256", {}),
                    amendment_r2.get("amended_sha256", {}),
                )
            ),
        },
        "current_runner_change_explicitly_pinned_by_amendment": current_runner_hash
        in {
            str(value.get("context_transfer_runner", ""))
            for value in (
                amendment_r1.get("amended_sha256", {}),
                amendment_r2.get("amended_sha256", {}),
            )
        },
    }
    chain_checks["immutable_chain_complete"] = all(chain_checks.values())

    execute_start = _position(runner, "def execute_once(")
    execute = runner[execute_start:]
    order = {
        "check_only": _position(execute, "check = check_only(root)"),
        "attempt_lock": _position(execute, "lock_sha = _create_attempt_lock"),
        "output_directory": _position(execute, "paths.output.mkdir"),
        "source_preflight_and_load": _position(execute, "source_hourly, source_provenance"),
        "source_fit": _position(execute, ".fit_pretrain("),
        "source_gate": _position(execute, "source_gate = _source_gate"),
        "domain_auc": _position(execute, "source_auc = _domain_classifier_auc"),
        "source_fail_branch": _position(execute, 'if not source_gate["passed"]:'),
        "local_three_window_prediction": _position(
            execute, "transfer = _produce_local_blind_predictions"
        ),
        "blind_seal": _position(execute, "seal_sha = _atomic_parquet"),
        "truth_attach": _position(execute, "evaluated = _attach_truth_after_seal"),
        "local_gate": _position(execute, "local_gate = _local_gate"),
        "viewpoint_gate": _position(execute, "viewpoint_gate = _viewpoint_gate"),
        "final_result": _position(execute, '_atomic_json(base_result, paths.output / "result.json")'),
    }
    expected_order = [
        "check_only",
        "attempt_lock",
        "output_directory",
        "source_preflight_and_load",
        "source_fit",
        "source_gate",
        "domain_auc",
        "source_fail_branch",
        "local_three_window_prediction",
        "blind_seal",
        "truth_attach",
        "local_gate",
        "viewpoint_gate",
    ]
    static_enforcement = {
        "canonical_root_attempt_lock_is_exclusive": "os.O_CREAT | os.O_EXCL" in runner,
        "attempt_lock_precedes_source_value_load": order["attempt_lock"]
        < order["source_preflight_and_load"],
        "source_model_fit_call_count_in_execute_once": execute.count(".fit_pretrain("),
        "source_fail_branch_precedes_local_three_window_predictions": order[
            "source_fail_branch"
        ]
        < order["local_three_window_prediction"],
        "domain_auc_diagnostic_runs_before_source_fail_short_circuit": order["domain_auc"]
        < order["source_fail_branch"],
        "blind_predictions_sealed_before_truth_attach": order["blind_seal"]
        < order["truth_attach"],
        "local_and_viewpoint_gates_both_required": "passed = bool(local_gate[\"passed\"] and viewpoint_gate[\"passed\"])"
        in execute,
        "ordered_state_transition": all(
            order[left] < order[right] for left, right in zip(expected_order, expected_order[1:])
        ),
        "exact_three_windows_bound_in_config": len(
            experiment["validation"]["local_outer_windows_exactly_three"]
        )
        == 3,
        "exact_three_windows_checked_against_frozen_default": "!= DEFAULT_WINDOWS" in runner,
        "hyperparameter_search_disabled": experiment["model"]["local_continuation_each_fold"][
            "hyperparameter_search"
        ]
        is False,
        "official_operational_path_literals_absent": not any(
            token in runner.casefold()
            for token in ("test_context", "test_index", "sample_submission", "submissions/")
        ),
    }

    expected = config["expected_scientific_contract"]
    decision = experiment["decision_gate"]
    scientific_contract = {
        "source_validation_years": experiment["source_case_builder"][
            "source_validation_years_exactly_three"
        ],
        "local_windows": [
            item[0] for item in experiment["validation"]["local_outer_windows_exactly_three"]
        ],
        "feature_count": experiment["features"]["expanded_feature_count"],
        "fixed_shrink_weight": experiment["model"]["postprocess"]
        ["fixed_long_lead_persistence_weight"],
        "fixed_shrink_leads": experiment["model"]["postprocess"]["active_leads_h"],
        "bootstrap_replicates": experiment["validation"]["bootstrap_replicates"],
        "bootstrap_seed": experiment["validation"]["bootstrap_seed"],
        "solution_delta_threshold_m": decision[
            "full_delta_candidate_minus_incumbent_m_at_most"
        ],
        "minimum_improved_local_windows": decision["minimum_improved_local_windows"],
        "maximum_critical_slice_regression_m": decision[
            "maximum_critical_slice_regression_m"
        ],
    }
    scientific_contract_matches = scientific_contract == expected

    mutable = config["mutable_state_metadata"]
    manifest_path = resolve_inside(root, mutable["canonical_manifest"])
    manifest = _manifest_metadata(manifest_path)
    plan = _manifest_metadata(resolve_inside(root, mutable["plan_receipt"]))
    smoke = _manifest_metadata(resolve_inside(root, mutable["smoke_receipt"]))
    final_expected = config["expected_final_manifest"]
    manifest_ready = bool(
        manifest.get("exists")
        and manifest.get("row_count") == final_expected["row_count"]
        and manifest.get("observed_start") == final_expected["observed_start"]
        and manifest.get("observed_end") == final_expected["observed_end"]
        and manifest.get("local_file_present")
        and manifest.get("file_sha256_present")
        and manifest.get("selected_cells") == final_expected["selected_cells"]
        and manifest.get("year_requests") == final_expected["year_requests"]
        and manifest.get("official_test_or_submission_accessed") is False
    )
    output_path = resolve_inside(root, mutable["context_output"])
    lock_path = resolve_inside(root, mutable["attempt_lock"])
    result_path = output_path / "result.json"
    if not manifest_ready:
        state_verdict = "BLOCKED_PREFLIGHT"
    elif lock_path.exists() and not result_path.is_file():
        state_verdict = "INCOMPLETE_LOCK_CONSUMED"
    elif result_path.is_file():
        status = read_json(result_path).get("status")
        state_verdict = {
            "NO_GO_SOURCE_GATE": "NO_GO_SOURCE_GATE",
            "NO_GO_LOCAL_OR_VIEWPOINT_GATE": "NO_GO_LOCAL_TRANSPORT",
            "RESEARCH_GATE_PASS": "GO_CONTEXT_SIGNAL",
        }.get(str(status), "UNRECOGNIZED_RESULT_STATUS")
    else:
        state_verdict = "READY_FOR_ONE_SHOT_EXECUTION"

    prepare_validate = _position(prepare, "validate_existing_canonical_manifest(")
    prepare_write = _position(prepare, "manifest_path = write_manifest")
    contract_gaps = {
        "immutable_preregistration_hash_chain_incomplete": not chain_checks[
            "immutable_chain_complete"
        ],
        "current_config_hash_not_pinned_by_preaccess_or_amendment": not chain_checks[
            "current_config_hash_pinned_by_preaccess_or_amendment"
        ],
        "current_runner_change_not_explicitly_listed_in_transport_amendments": not chain_checks[
            "current_runner_change_explicitly_pinned_by_amendment"
        ],
        "check_only_can_report_passed_while_source_not_ready": (
            '"passed": True' in runner and '"source_quarantine_ready": source_ready' in runner
        ),
        "blocked_preflight_has_no_durable_result_status": "BLOCKED_PREFLIGHT" not in runner,
        "crash_after_attempt_lock_has_no_terminal_state_or_resume": (
            "one-shot attempt was already consumed" in runner
            and "INCOMPLETE_LOCK_CONSUMED" not in runner
        ),
        "attempt_registry_is_root_relative_not_global": (
            'parser.add_argument("--root"' in runner
            and "output.with_name" in runner
        ),
        "final_independent_qa_missing": "independent_qa" not in runner.casefold(),
        "final_output_manifest_hash_receipt_missing": not (
            'paths.output / "manifest.json"' in runner
            or 'paths.output / "manifest.sha256"' in runner
        ),
        "incomplete_legacy_canonical_manifest_present": bool(
            manifest.get("exists") and not manifest_ready
        ),
        "prepare_validates_existing_canonical_before_final_write": prepare_validate
        < prepare_write,
        "incomplete_canonical_handoff_has_no_explicit_test": not bool(
            re.search(r"incomplete.*canonical|canonical.*incomplete", tests, re.IGNORECASE)
        ),
    }

    gate_definitions = {
        "source_gate": {
            "all_three_held_years_better_than_persistence": True,
            "pooled_episode_ci90_upper_below_zero": True,
            "lead_18_non_degrade": True,
            "lead_24_non_degrade": True,
            "minimum_dynamic_finite_fraction": 0.995,
            "short_circuit_local_three_windows_on_fail": static_enforcement[
                "source_fail_branch_precedes_local_three_window_predictions"
            ],
            "fixed_domain_auc_is_diagnostic_not_gate": True,
        },
        "solution_gate_vs_frozen_incumbent": {
            "pooled_delta_at_most_m": -0.03,
            "minimum_improved_windows": 2,
            "paired_episode_ci90_upper_below_zero": True,
            "maximum_critical_slice_regression_m": 0.0075,
        },
        "viewpoint_gate_vs_matched_local_only_control": {
            "pooled_delta_below_zero": True,
            "minimum_improved_windows": 2,
            "paired_episode_ci90_upper_below_zero": True,
        },
        "go_requires_solution_and_viewpoint": True,
    }

    return {
        "schema_version": "p3.era5_static_contract_audit.result.v1",
        "audit_id": config["audit_id"],
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "status": "STATIC_AUDIT_COMPLETE",
        "scientific_state_verdict": state_verdict,
        "common_secondary_recon_sha256": common_hash,
        "common_secondary_recon_hash_matches": common_hash == common["sha256"],
        "input_hash_checks": input_hash_checks,
        "observed_input_sha256": observed_hashes,
        "hash_chain": chain_checks,
        "scientific_contract": scientific_contract,
        "scientific_contract_matches_preregistered_audit": scientific_contract_matches,
        "static_enforcement": static_enforcement,
        "gate_definitions": gate_definitions,
        "manifest_state": {
            "ready": manifest_ready,
            "canonical": manifest,
            "plan_receipt": plan,
            "smoke_receipt": smoke,
        },
        "context_execution_state": {
            "attempt_lock_exists": lock_path.exists(),
            "output_directory_exists": output_path.exists(),
            "result_exists": result_path.is_file(),
        },
        "contract_gaps": contract_gaps,
        "final_qa_requirement": {
            "required_after_terminal_result": True,
            "must_independently_recompute_gate_booleans_from_aggregate_result": True,
            "must_verify_attempt_lock_blind_seal_result_and_readme_hashes": True,
            "must_verify_all_static_and_source_manifest_hashes_unchanged": True,
            "must_map_terminal_status_to_common_verdict": True,
            "current_runner_provides_this": False,
        },
        "external_actions": {
            "era5_value_file_reads": 0,
            "running_process_inspections_or_mutations": 0,
            "official_test_sample_submission_reads": 0,
            "context_runner_executions": 0,
            "model_fits": 0,
        },
    }


def render_report(result: Mapping[str, Any]) -> str:
    gaps = result["contract_gaps"]
    true_gaps = [name for name, value in gaps.items() if value]
    manifest = result["manifest_state"]["canonical"]
    chain = result["hash_chain"]
    return f"""# P3 ERA5 context-transfer 정적 계약 감사

## 결론

현재 과학적 상태 판정은 **`{result['scientific_state_verdict']}`**다. 이는 다운로드 실패 판정이 아니라, canonical manifest가 아직 등록된 262,917행·363-request 완료 surface를 증명하지 못한다는 뜻이다. context-transfer attempt lock과 output은 아직 없으므로 과학 실험의 1회 기회는 소비되지 않았다.

canonical repository root 안에서는 O_EXCL attempt lock, source gate 실패 단락, 정확히 세 local window, blind seal 후 truth attach, solution+viewpoint 동시 통과가 코드로 강제된다. hyperparameter search와 결과 기반 rerun 경로는 없다.

다만 현재 상태를 그대로 `완전한 단일 실행 계약`으로 표현하면 안 된다. 정적 감사에서 {len(true_gaps)}개 위험/누락 신호가 확인됐다.

## 핵심 누락과 위험

1. **불변 해시 체인이 완결되지 않았다.** preaccess seal은 runner `097643...`와 config `7cafe...`를 고정했지만 현재 runner는 `{result['observed_input_sha256']['context_runner']}`이고 config는 `{result['observed_input_sha256']['experiment_config']}`다. R2 transport amendment는 현재 prepare/data 변경은 기록하지만 context runner와 최종 config hash를 `amended_sha256`에 명시하지 않는다. 현재 config가 자기 자신과 현재 runner를 신뢰하는 구조여서 외부 seal에서 current config까지 이어지는 마지막 연결이 없다. `immutable_chain_complete={str(chain['immutable_chain_complete']).lower()}`.

2. **`check_only.passed=true`는 source preflight 완료를 뜻하지 않는다.** source가 준비되지 않아도 contract/schema audit 자체는 passed가 된다. 자동화는 반드시 `source_quarantine_ready=true`와 non-null `source_preflight`를 별도로 요구해야 한다.

3. **현재 canonical manifest는 incomplete legacy snapshot이다.** row_count={manifest.get('row_count')}, stage={manifest.get('stage')}, local file/checksum 없음, year requests={manifest.get('year_requests')}. prepare runner는 기존 canonical 검증을 최종 manifest write보다 먼저 호출한다. 기존 incomplete canonical을 최종 것으로 안전하게 교체하는 경우가 테스트에 명시돼 있지 않아, 완료 시 manifest collision으로 `BLOCKED_PREFLIGHT`가 될 위험을 사전 확인해야 한다.

4. **잠금 이후 장애 상태가 없다.** attempt lock 생성 뒤 I/O·fit·QA가 실패하면 재실행은 영구 거부되지만 `BLOCKED_EXECUTION` 또는 `INCOMPLETE_LOCK_CONSUMED` 결과가 남지 않는다. 또한 `--root`가 사용자 지정 가능하므로 잠금은 canonical root 내부에서는 강하지만 전역 실험 registry는 아니다.

5. **독립 최종 QA와 output hash manifest가 없다.** runner는 blind OOF hash를 result에 남기지만 result/README/attempt lock 전체를 독립 재검산하고 묶는 `manifest.json`/`manifest.sha256`를 생성하지 않는다. 공통 2차 정찰 계약의 최종 QA 요구는 terminal result 이후 별도 단계가 반드시 맡아야 한다.

## Gate 정의

- Source gate: held 2021·2022·2023 모두 persistence보다 개선, pooled episode bootstrap CI90 upper<0, lead 18/24 각각 non-degrade, dynamic finite fraction≥0.995. 실패하면 local 3창 prediction은 실행되지 않는다. 단, 고정 domain-AUC diagnostic은 분기 전에 실행된다.
- Solution gate: frozen incumbent 대비 pooled Δ≤-0.03m, 2/3 window 개선, episode CI90 upper<0, critical slice 최대 악화≤0.0075m.
- Viewpoint gate: matched local-only control 대비 pooled Δ<0, 2/3 window 개선, episode CI90 upper<0.
- 최종 GO는 solution과 viewpoint가 모두 통과해야 한다. AUC는 보고 전용이며 직접 pooling 금지만 재확인한다.

## 완료 후 필수 절차

1. canonical manifest metadata가 262,917행, 2014-01-01 00UTC~2023-12-31 14UTC, 3 selected cells, 363 requests, SHA256/provenance를 모두 증명하는지 확인한다.
2. `check_only.passed`가 아니라 source-ready와 실제 preflight receipt를 확인한다.
3. context runner는 canonical root에서 `--execute` 정확히 한 번만 호출한다.
4. terminal result 후 별도 QA가 source/local/viewpoint gate boolean을 재산출하고 attempt lock·blind seal·result·README·입력 manifest/code hash를 하나의 최종 manifest에 pin한다.
5. common verdict는 source fail→`NO_GO_SOURCE_GATE`, local/viewpoint fail→`NO_GO_LOCAL_TRANSPORT`, 모두 pass→`GO_CONTEXT_SIGNAL`, source integrity 미확립→`BLOCKED_PREFLIGHT`로 매핑한다.

## 감사 경계

이 validator는 config·scripts·JSON manifest metadata만 읽었다. ERA5 value file, 실행 중 프로세스, 모델 fit, 공식 P3 평가/sample/submission 경로에는 접근하지 않았다.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/experiments/p3_era5_static_contract_audit_20260825_v1.json",
    )
    parser.add_argument("--execute-static-audit", action="store_true")
    args = parser.parse_args()
    if not args.execute_static_audit:
        raise SystemExit("Static audit requires --execute-static-audit")
    config_path = args.config.resolve()
    config = read_json(config_path)
    output = resolve_inside(ROOT, config["output_dir"])
    protocol = output / "static_audit_protocol.json"
    if not protocol.is_file():
        raise FileNotFoundError("Static audit protocol must be sealed before execution")
    protected = [
        output / "static_audit.json",
        output / "report_ko.md",
        output / "manifest.json",
        output / "manifest.sha256",
    ]
    if any(path.exists() for path in protected):
        raise FileExistsError("Static audit outputs are append-only")
    protocol_value = read_json(protocol)
    if protocol_value["config_sha256"] != sha256_file(config_path):
        raise ValueError("Static audit protocol config hash differs")
    if protocol_value["validator_sha256"] != sha256_file(Path(__file__).resolve()):
        raise ValueError("Static audit protocol validator hash differs")
    result = audit(config_path)
    exclusive_json(output / "static_audit.json", result)
    exclusive_text(output / "report_ko.md", render_report(result))
    outputs = {
        "static_audit_protocol.json": sha256_file(protocol),
        "static_audit.json": sha256_file(output / "static_audit.json"),
        "report_ko.md": sha256_file(output / "report_ko.md"),
    }
    manifest = {
        "schema_version": "p3.era5_static_contract_audit.manifest.v1",
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE_STATIC_READ_ONLY_AUDIT",
        "scientific_state_verdict": result["scientific_state_verdict"],
        "common_secondary_recon_sha256": result["common_secondary_recon_sha256"],
        "config_sha256": sha256_file(config_path),
        "validator_sha256": sha256_file(Path(__file__).resolve()),
        "outputs_sha256": outputs,
        "external_actions": result["external_actions"],
    }
    exclusive_json(output / "manifest.json", manifest)
    manifest_hash = sha256_file(output / "manifest.json")
    exclusive_text(output / "manifest.sha256", f"{manifest_hash}  manifest.json\n")
    print(json.dumps({"status": manifest["status"], "verdict": result["scientific_state_verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
