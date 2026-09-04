"""Validate and seal the aggregate-only P2 nested-surrogate decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p2_restore.authoritative_nested_surrogate_contract import validate_contract  # noqa: E402

KST = ZoneInfo("Asia/Seoul")
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/p2_authoritative_nested_surrogate_recipe_20260825_v1.json"
)
EXPECTED_CONFIG_SHA256 = (
    "8b442b31ec1b2af8d1f356d728cc83aa94636c34528e26adad48ff268f2ce2b2"
)
FORBIDDEN_PATH_TOKENS = (
    "test_index",
    "sample_submission",
    "baseline_interp",
    "p2_submission",
    "/output/",
    "\\output\\",
)


def _now_kst() -> str:
    return datetime.now(KST).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError(f"expected JSON object: {path}")
    return parsed


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")


def _write_text_new(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value.rstrip())
        handle.write("\n")


def _repo_file(relative: str) -> Path:
    normalized = relative.replace("\\", "/").lower()
    if any(token in normalized for token in FORBIDDEN_PATH_TOKENS):
        raise RuntimeError(f"forbidden evidence path: {relative}")
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT) or not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".json":
        raise RuntimeError(f"only aggregate JSON evidence is allowed: {relative}")
    return path


def _validate_recipe_header(path: Path, recipe: dict[str, Any]) -> None:
    if _sha256(path) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("sealed recipe config hash drift")
    if recipe.get("schema_version") != "p2_authoritative_nested_surrogate_recipe.v1":
        raise ValueError("unexpected recipe schema")
    if recipe.get("status") != "SEALED_DRY_RUN_BEFORE_ANY_NEW_SCORE_OR_FIT":
        raise ValueError("recipe is not sealed")
    if recipe.get("training_authorized") or recipe.get("promotion_authorized"):
        raise ValueError("recipe improperly authorizes training or promotion")
    output = recipe["output"]
    if not output["aggregate_contract_only"] or output["prediction_files_written"]:
        raise ValueError("aggregate-only output contract changed")


def _verify_evidence(
    recipe: dict[str, Any],
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths: dict[str, Path] = {}
    manifest: dict[str, dict[str, Any]] = {}
    for name, spec in recipe["evidence"].items():
        path = _repo_file(str(spec["path"]))
        digest = _sha256(path)
        if digest != str(spec["sha256"]):
            raise RuntimeError(f"immutable evidence drift: {name}")
        paths[name] = path
        manifest[name] = {
            "path": str(spec["path"]),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
    return paths, manifest


def _artifact_dir(recipe: Mapping[str, Any]) -> Path:
    path = (PROJECT_ROOT / str(recipe["output"]["directory"])).resolve()
    if not path.is_relative_to(PROJECT_ROOT / "artifacts"):
        raise RuntimeError("output must remain below repository artifacts")
    return path


def _render_report(
    decision: Mapping[str, Any],
    state_matrix: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> str:
    evidence = decision["matched_evidence"]
    dimension_lines = []
    for row in state_matrix["exactness_dimensions"]:
        dimension_lines.append(
            "| {dimension} | {historical_exact_state} | {current_implementation_conformance} |".format(
                **row
            )
        )
    blockers = "\n".join(
        f"{index}. `{item['id']}` — {item['consequence']}"
        for index, item in enumerate(state_matrix["blocking_recipe_gaps"], start=1)
    )
    return f"""# P2 exact causal prefix refit 2차 정찰

## 결론

판정은 **`NEW_AUTHORITATIVE_SURROGATE_REQUIRED`**, 선택지는 **(A) 새 authoritative
nested train-only surrogate recipe**입니다. 현재 exact incumbent의 causal prefix
refit은 재현할 수 없으며 exact cell은 15개 중 0개입니다. 중앙 계약 문구만 바꾸는
(B)는 빠진 학습 상태를 만들지 못하므로 선택하지 않았습니다.

이 문서가 봉인한 recipe는 향후 모든 family가 같은 population, outer fold, cutoff,
complete-pipeline seed, postprocess, metric을 공유하게 만드는 비교 기준입니다. 이는
**공식 incumbent의 exact 재현이 아니며**, 현재 실행·학습·승격 권한도 없습니다.

## 왜 현재 exact refit이 불가능한가

{blockers}

저장 가중치 추론과 deterministic postprocess는 재현되지만, 이는 causal fold-train-only
refit 의미를 소유하지 않습니다. 특히 May–June component OOF가 없고, 기존 겹치는
OOF도 future-complement split이며 7일 embargo가 없습니다.

## Exactness 상태표

| 차원 | historical exact 상태 | 현재 conformance 증거 |
|---|---|---|
{chr(10).join(dimension_lines)}

새 계약은 8개 차원의 의미를 모두 소유하지만, prefix mask·component OOF·epoch/meta-refit
3개 차원은 구현 및 별도 승인 전까지 `PENDING`입니다. 따라서 dry-run PASS는 학습 GO가
아닙니다.

## 기존 matched 결과가 요구하는 결정성

- exact frozen-lineage fallback gain: {evidence['exact_frozen_fallback_gain_c']:+.6f}℃
- time-safe surrogate 전체 cutoff fallback gain: {evidence['surrogate_all_cutoff_fallback_gain_c']:+.6f}℃
- time-safe surrogate full-prefix fallback gain: {evidence['surrogate_full_prefix_fallback_gain_c']:+.6f}℃
- 방향 충돌 확인: `{str(evidence['direction_conflict_confirmed']).lower()}`
- causal correction supported rows: {evidence['causal_correction_supported_rows']}

즉 기존 surrogate 자체로 승격할 수 없습니다. 공통 nested recipe가 필요한 이유는 점수를
높이는 새 후보를 만들기 위해서가 아니라, lineage에 따라 부호가 바뀌는 비교를 하나의
causal surface로 고정하기 위해서입니다.

## 봉인된 향후 비교

- 3 outer folds × 5 chronological cutoffs = 15 cells
- 3 complete-pipeline seeds = {len(comparison['seeded_execution_plan'])} seeded cells
- prefix unique timestamps의 오래된 쪽부터 `ceil(fraction*N)`, outer 시작 7일 embargo
- prefix 내부 3개 expanding inner folds에서 component OOF 생성
- epoch, layer stack, public-state gate를 prefix 내부에서만 선택·refit
- frozen official stack/gate/epoch 재사용 금지
- 가족별 후보 수와 기본값은 기존 matched-budget 봉인과 동일
- Public 점수는 모든 선택 이후 transport audit에서만 사용

## 실행 정지 조건

새 학습은 별도 승인이 있기 전까지 금지됩니다. 구현 후에도 ordered key/truth digest,
cutoff receipt, seed fan-out, mask audit, 45-cell dry-run 중 하나라도 family 간 다르면 즉시
중지해야 합니다. surrogate를 exact official incumbent라고 표현하거나 공식 test/sample/
submission/Public 점수를 선택에 사용해도 즉시 중지합니다.
"""


def _output_manifest(paths: Sequence[Path]) -> dict[str, Any]:
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in paths
    }


def run(config_path: Path, *, execute: bool) -> dict[str, Any]:
    config_path = config_path.resolve()
    recipe = _read_json(config_path)
    _validate_recipe_header(config_path, recipe)
    evidence_paths, evidence_manifest = _verify_evidence(recipe)
    preflight = {
        "status": "PASS",
        "config_sha256": _sha256(config_path),
        "aggregate_json_evidence_count": len(evidence_paths),
        "new_model_fits": 0,
        "new_score_reads": 0,
        "official_test_reads": 0,
        "sample_submission_reads": 0,
        "submission_candidate_reads": 0,
    }
    if not execute:
        evidence = {name: _read_json(path) for name, path in evidence_paths.items()}
        validation = validate_contract(recipe, evidence)
        return {**preflight, "verdict": validation.decision["verdict"]}

    output_dir = _artifact_dir(recipe)
    output_dir.mkdir(parents=True, exist_ok=False)
    runner_path = Path(__file__).resolve()
    module_path = (
        PROJECT_ROOT
        / "src/p2_restore/authoritative_nested_surrogate_contract.py"
    )
    seal_path = output_dir / "contract_seal.json"
    seal = {
        "schema_version": "p2_authoritative_nested_surrogate_contract.seal.v1",
        "sealed_at_kst": _now_kst(),
        "status": "SEALED_BEFORE_AGGREGATE_EVIDENCE_INTERPRETATION",
        "config": {
            "path": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": _sha256(config_path),
        },
        "decisive_recon_preregistration": recipe["evidence"][
            "decisive_recon_preregistration"
        ],
        "runner_sha256": _sha256(runner_path),
        "module_sha256": _sha256(module_path),
        "chosen_verdict": "NEW_AUTHORITATIVE_SURROGATE_REQUIRED",
        "central_contract_mutations": 0,
        "new_model_fits": 0,
        "new_score_reads": 0,
        "official_public_score_available_to_decision": False,
    }
    _write_json_new(seal_path, seal)

    evidence = {name: _read_json(path) for name, path in evidence_paths.items()}
    validation = validate_contract(recipe, evidence)
    decision_path = output_dir / "decision.json"
    matrix_path = output_dir / "required_state_matrix.json"
    comparison_path = output_dir / "comparison_preregistration.json"
    qa_path = output_dir / "qa.json"
    report_path = output_dir / "report.md"
    qa = {
        **validation.qa,
        "config_hash_pass": True,
        "evidence_hashes_pass": True,
        "aggregate_json_evidence_count": len(evidence_paths),
    }
    _write_json_new(decision_path, validation.decision)
    _write_json_new(matrix_path, validation.state_matrix)
    _write_json_new(comparison_path, validation.comparison_preregistration)
    _write_json_new(qa_path, qa)
    _write_text_new(
        report_path,
        _render_report(
            validation.decision,
            validation.state_matrix,
            validation.comparison_preregistration,
        ),
    )
    output_paths = [
        seal_path,
        decision_path,
        matrix_path,
        comparison_path,
        qa_path,
        report_path,
    ]
    manifest = {
        "schema_version": "p2_authoritative_nested_surrogate_contract.manifest.v1",
        "experiment_id": recipe["experiment_id"],
        "created_at_kst": _now_kst(),
        "status": qa["status"],
        "verdict": validation.decision["verdict"],
        "config": seal["config"],
        "implementation": {
            "runner": {
                "path": str(runner_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(runner_path),
            },
            "module": {
                "path": str(module_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(module_path),
            },
        },
        "evidence": evidence_manifest,
        "outputs": _output_manifest(output_paths),
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "external_actions": {
            "central_contract_mutations": 0,
            "new_model_fits": 0,
            "new_score_reads": 0,
            "official_test_reads": 0,
            "sample_submission_reads": 0,
            "submission_candidate_reads": 0,
            "submission_files_generated": 0,
            "uploads": 0,
            "p3_era5_process_mutations": 0,
        },
    }
    manifest_path = output_dir / "manifest.json"
    _write_json_new(manifest_path, manifest)
    return {
        "status": qa["status"],
        "verdict": validation.decision["verdict"],
        "output_dir": str(output_dir),
        "manifest_sha256": _sha256(manifest_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run(args.config, execute=bool(args.execute))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
