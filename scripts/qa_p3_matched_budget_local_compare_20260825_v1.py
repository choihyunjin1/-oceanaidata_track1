from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p3_wave.matched_budget_local_compare import (  # noqa: E402
    KEY_COLUMNS,
    exclusive_write_json,
    exclusive_write_text,
    read_json,
    sha256_file,
)


def _kst_now() -> str:
    return datetime.now().astimezone().isoformat()


def _resolve(relative: str) -> Path:
    return (ROOT / relative).resolve()


def _rmse(frame: pd.DataFrame, prediction: str) -> float:
    residual = frame[prediction].to_numpy(dtype=np.float64) - frame["target_hs"].to_numpy(
        dtype=np.float64
    )
    return float(math.sqrt(float(np.dot(residual, residual)) / residual.size))


def _bootstrap(
    frame: pd.DataFrame,
    candidate: str,
    reference: str,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    work = frame.loc[:, ["fold", "station", "anchor_id", "target_hs", candidate, reference]].copy()
    work["c"] = np.square(work[candidate] - work["target_hs"])
    work["r"] = np.square(work[reference] - work["target_hs"])
    grouped = work.groupby(["fold", "station", "anchor_id"], sort=True)[["c", "r"]].sum()
    ncase = len(grouped)
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, ncase, size=(replicates, ncase))
    denominator = float(len(frame))
    delta = np.sqrt(grouped["c"].to_numpy()[sampled].sum(axis=1) / denominator) - np.sqrt(
        grouped["r"].to_numpy()[sampled].sum(axis=1) / denominator
    )
    low, high = np.quantile(delta, [0.05, 0.95])
    return float(low), float(high), float(np.mean(delta < 0.0))


def _assert_close(observed: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if abs(float(observed) - float(expected)) > tolerance:
        raise AssertionError(f"{label} differs: {observed} != {expected}")


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _format_ci(receipt: dict[str, Any]) -> str:
    low, high = receipt["delta_candidate_minus_reference_ci90_m"]
    return f"[{low:+.6f}, {high:+.6f}]"


def execute(config_path: Path) -> dict[str, Any]:
    started = perf_counter()
    config = read_json(config_path)
    output_dir = _resolve(config["output_dir"])
    metrics_path = output_dir / "metrics.json"
    execution_path = output_dir / "execution_receipt.json"
    seal_path = output_dir / "preexecution_seal.json"
    for required in (metrics_path, execution_path, seal_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    for forbidden in (
        output_dir / "independent_qa.json",
        output_dir / "technical_report_ko.md",
        output_dir / "manifest.json",
        output_dir / "manifest.sha256",
    ):
        if forbidden.exists():
            raise FileExistsError(f"Independent QA output already exists: {forbidden}")

    metrics = read_json(metrics_path)
    checks: dict[str, bool] = {}
    checks["common_protocol_hash_pinned"] = (
        sha256_file(_resolve(config["common_protocol"]["path"]))
        == config["common_protocol"]["sha256"]
        == metrics["common_protocol_sha256"]
    )
    checks["config_hash_pinned_before_results"] = (
        read_json(seal_path)["config_sha256"] == sha256_file(config_path)
    )
    checks["all_inputs_unchanged"] = all(
        sha256_file(_resolve(value["path"])) == value["sha256"]
        for value in config["input_files"].values()
    ) and all(
        sha256_file(_resolve(family["oof_path"])) == family["oof_sha256"]
        and sha256_file(_resolve(family["metrics_path"])) == family["metrics_sha256"]
        for family in config["structural_families"]
    )

    canonical = pd.read_parquet(_resolve(config["input_files"]["canonical_oof"]["path"]))
    checks["surface_181_cases_1086_rows"] = (
        len(canonical) == 1086
        and canonical.groupby(["fold", "station", "anchor_id"]).ngroups == 181
        and not canonical.duplicated(list(KEY_COLUMNS)).any()
    )
    for prediction, summary in metrics["canonical_family"]["settings"].items():
        _assert_close(_rmse(canonical.assign(**{prediction: canonical[prediction]}) if prediction in canonical else _rebuild(canonical, prediction), prediction), summary["rmse_m"], f"canonical {prediction}")
    checks["canonical_pooled_metrics_independently_reproduced"] = True

    bootstrap_config = config["evaluation"]
    rebuilt = _rebuild_all(canonical)
    for candidate, comparison in metrics["canonical_family"]["comparisons_vs_current_20pct"].items():
        low, high, probability = _bootstrap(
            rebuilt,
            candidate,
            "current_shrink_20pct",
            replicates=bootstrap_config["bootstrap_replicates"],
            seed=bootstrap_config["bootstrap_seed"],
        )
        receipt = comparison["complete_case_bootstrap"]
        _assert_close(low, receipt["delta_candidate_minus_reference_ci90_m"][0], f"{candidate} bootstrap low")
        _assert_close(high, receipt["delta_candidate_minus_reference_ci90_m"][1], f"{candidate} bootstrap high")
        _assert_close(probability, receipt["probability_candidate_improves_descriptive"], f"{candidate} bootstrap probability")
    checks["canonical_bootstrap_independently_reproduced"] = True

    for family in config["structural_families"]:
        full = pd.read_parquet(_resolve(family["oof_path"]))
        full = full.loc[full["prefix_fraction"] == family["prefix_fraction"]].copy()
        receipt = metrics["structural_families"][family["id"]]
        _assert_close(_rmse(full, "incumbent_prediction"), receipt["incumbent_matched_refit"]["rmse_m"], f"{family['id']} incumbent")
        _assert_close(_rmse(full, "challenger_prediction"), receipt["challenger"]["rmse_m"], f"{family['id']} challenger")
        low, high, probability = _bootstrap(
            full,
            "challenger_prediction",
            "incumbent_prediction",
            replicates=bootstrap_config["bootstrap_replicates"],
            seed=bootstrap_config["bootstrap_seed"],
        )
        boot = receipt["complete_case_bootstrap"]
        _assert_close(low, boot["delta_candidate_minus_reference_ci90_m"][0], f"{family['id']} bootstrap low")
        _assert_close(high, boot["delta_candidate_minus_reference_ci90_m"][1], f"{family['id']} bootstrap high")
        _assert_close(probability, boot["probability_candidate_improves_descriptive"], f"{family['id']} bootstrap probability")
    checks["all_structural_metrics_and_bootstraps_independently_reproduced"] = True
    checks["density_no_go_not_overridden"] = (
        metrics["density_gate"]["selection_reason"] == "NO_GO_LABEL_FREE_DOMAIN_GATE"
        and metrics["density_gate"]["override_attempted"] is False
        and metrics["density_gate"]["density_model_trained_in_this_run"] is False
    )
    checks["no_additional_windows_added"] = metrics["additional_windows"]["count"] == 0
    checks["no_model_fit_or_external_action"] = (
        metrics["resource_isolation"]["model_fits"] == 0
        and metrics["resource_isolation"]["era5_path_or_process_accesses"] == 0
        and all(value == 0 for value in read_json(execution_path)["external_actions"].values())
    )
    checks["no_prediction_artifact_generated"] = not any(
        path.suffix.lower() in {".csv", ".parquet", ".npy"}
        for path in output_dir.iterdir()
    )
    passed = all(checks.values())
    qa = {
        "schema_version": "p3.matched_budget_local_compare.independent_qa.v1",
        "created_at_kst": _kst_now(),
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "metrics_sha256": sha256_file(metrics_path),
        "execution_receipt_sha256": sha256_file(execution_path),
        "elapsed_seconds": float(perf_counter() - started),
    }
    exclusive_write_json(output_dir / "independent_qa.json", qa)
    if not passed:
        raise AssertionError("Independent QA failed")

    current = metrics["canonical_family"]["settings"]["current_shrink_20pct"]
    round_b = metrics["canonical_family"]["settings"]["round_b_shrink_22p5pct"]
    round_a = metrics["canonical_family"]["settings"]["round_a_shrink_25pct"]
    comparisons = metrics["canonical_family"]["comparisons_vs_current_20pct"]
    structure_rows = []
    for family_id, receipt in metrics["structural_families"].items():
        delta = receipt["delta_challenger_minus_incumbent_m"]["pooled"]
        ci = _format_ci(receipt["complete_case_bootstrap"])
        structure_rows.append(
            f"| {family_id} | {receipt['incumbent_matched_refit']['rmse_m']:.6f} | "
            f"{receipt['challenger']['rmse_m']:.6f} | {delta:+.6f} | {ci} | "
            f"{receipt['residual_correlation']:.6f} |"
        )
    coefficient_rows = []
    for label, receipt in (("현행 20%", current), ("Round B 22.5%", round_b), ("Round A 25%", round_a)):
        delta = receipt["rmse_m"] - current["rmse_m"]
        coefficient_rows.append(f"| {label} | {receipt['rmse_m']:.6f} | {delta:+.6f} |")
    report = f"""# P3 matched-budget 로컬 비교 기술 보고서

## 결론

**새 구조로 문제를 바꿔야 한다는 증거는 이 고정 로컬 비교에서 나오지 않았다.** 동등한 3-seed/45-cell 예산으로 비교 가능한 세 구조는 모두 matched CatBoost 기준보다 악화했다. 반면 20%→22.5% 또는 25% 장기-lead 수축은 방향상 소폭 개선했지만 최대 폭이 0.000444 m에 불과하고 complete-case bootstrap 90% 구간이 0을 포함한다. 따라서 이 결과는 새 구조 승격도, 추가 계수 튜닝도 정당화하지 않는다.

## 고정 surface와 격리

- historical window: `2024_h2_storm`, `winter_transition`, `2025_h1`
- 181 unique complete cases / 1,086 rows / leads 3·6·9·12·18·24 h
- 공통 규약 SHA256: `{metrics['common_protocol_sha256']}`
- 공식 평가값·submission 값·ERA5 접근 0회, 신규 모델 fit 0회, 결과 기반 재실행 0회
- density-ratio 모델은 기존 label-free ESS `NO_GO`를 그대로 유지했다. 최소 ratio ESS={metrics['density_gate']['minimum_ratio_ess_fraction']:.6f}, combined ESS={metrics['density_gate']['minimum_combined_ess_fraction']:.6f}, station ESS={metrics['density_gate']['minimum_station_combined_ess_fraction']:.6f}.

## 계수 효과: 동일 CatBoost OOF

| 설정 | pooled RMSE (m) | 현행 20% 대비 ΔRMSE (m) |
|---|---:|---:|
{os.linesep.join(coefficient_rows)}

- Round B bootstrap 90% CI: {_format_ci(comparisons['round_b_shrink_22p5pct']['complete_case_bootstrap'])}, 개선확률 {comparisons['round_b_shrink_22p5pct']['complete_case_bootstrap']['probability_candidate_improves_descriptive']:.3f}
- Round A bootstrap 90% CI: {_format_ci(comparisons['round_a_shrink_25pct']['complete_case_bootstrap'])}, 개선확률 {comparisons['round_a_shrink_25pct']['complete_case_bootstrap']['probability_candidate_improves_descriptive']:.3f}
- 세 점만 사전 고정해 계수 민감도를 계산했으며 추가 grid/최적점 탐색은 하지 않았다.

## 구조 효과: 동등한 3-seed/45-cell 기본 예산

각 구조의 challenger와 comparator는 동일 prefix·fold·metric·clip·고정 20% 후처리를 썼다. 아래 Δ는 challenger−matched incumbent이므로 양수가 악화다.

| 구조 | matched incumbent | challenger | ΔRMSE (m) | complete-case CI90 | residual corr. |
|---|---:|---:|---:|---:|---:|
{os.linesep.join(structure_rows)}

모든 구조의 pooled Δ가 양수이므로 `structure_gain_supported=false`다. residual correlation은 오류 방향의 유사성을 보여 주지만, RMSE 열세를 뒤집는 독립 보정 신호는 확인되지 않았다.

## slice와 QA

전체 설정에 대해 pooled/fold/station/lead RMSE가 `metrics.json`에 기록되어 있다. 독립 프로세스가 모든 pooled RMSE와 5,000회 complete-case bootstrap을 다시 계산했고 QA는 `{qa['status']}`다. 25% 재구성은 기존 sealed Round A OOF와 최대 절대차 {metrics['canonical_family']['round_a_25pct_reference_prediction_max_abs_m']:.3e} m로 일치했다.

## 해석과 한계

- coefficient 효과와 structure 효과는 서로 다른 공정한 기준으로 분리했다. 계수 계열은 동일 canonical OOF를, 구조 계열은 각 실험의 fresh 3-seed matched comparator를 사용한다.
- 구조 comparator의 RMSE는 canonical 단일 fold-seed incumbent와 정확히 같은 예측이 아니다. 이는 불공정이 아니라 seed/fit 예산을 맞추기 위한 별도 기준이며 두 숫자를 직접 섞어 구조 gain으로 해석하지 않았다.
- 세 window 밖의 추가 비중첩 sealed support가 없어 새 window를 사후 추가하지 않았다. 따라서 181-case 불확실성과 시간대 수송 한계가 남는다.
- 공식 점수는 후보 선택·튜닝에 사용하지 않았고 이 보고서에도 선택 근거로 넣지 않았다.

## 다음 단계

현 단계에서는 현행 모델을 유지하고 P3 구조 탐색을 중지하는 것이 타당하다. 재개 조건은 결과를 보고 고른 수축계수가 아니라, 완전히 새로운 causal signal과 사전 봉인된 추가 독립 historical support가 함께 확보되는 경우다. 본 결과는 promotion·submission 생성·업로드를 승인하지 않는다.
"""
    exclusive_write_text(output_dir / "technical_report_ko.md", report)

    implementation_paths = {
        "config": config_path,
        "module": ROOT / "src/p3_wave/matched_budget_local_compare.py",
        "runner": ROOT / "scripts/run_p3_matched_budget_local_compare_20260825_v1.py",
        "qa_runner": Path(__file__).resolve(),
        "test": ROOT / "tests/test_p3_matched_budget_local_compare_20260825_v1.py",
    }
    output_paths = {
        "preexecution_seal": seal_path,
        "metrics": metrics_path,
        "execution_receipt": execution_path,
        "independent_qa": output_dir / "independent_qa.json",
        "technical_report_ko": output_dir / "technical_report_ko.md",
    }
    manifest = {
        "schema_version": "p3.matched_budget_local_compare.manifest.v1",
        "created_at_kst": _kst_now(),
        "status": "COMPLETE_QA_PASS_NO_PROMOTION",
        "append_only_generation": True,
        "common_protocol_sha256": config["common_protocol"]["sha256"],
        "git_head": _git_head(),
        "implementation_sha256": {
            key: sha256_file(path) for key, path in implementation_paths.items()
        },
        "input_sha256": {
            key: value["sha256"] for key, value in config["input_files"].items()
        }
        | {
            family["id"] + "_oof": family["oof_sha256"]
            for family in config["structural_families"]
        }
        | {
            family["id"] + "_metrics": family["metrics_sha256"]
            for family in config["structural_families"]
        },
        "output_sha256": {key: sha256_file(path) for key, path in output_paths.items()},
        "decision": {
            "structure_gain_supported": metrics["summary"]["structure_gain_supported"],
            "maturity_bias_supported": metrics["summary"]["maturity_bias_supported"],
            "density_gate_override": False,
            "promotion_authorized": False,
        },
        "external_actions": read_json(execution_path)["external_actions"],
    }
    exclusive_write_json(output_dir / "manifest.json", manifest)
    manifest_hash = sha256_file(output_dir / "manifest.json")
    exclusive_write_text(
        output_dir / "manifest.sha256", f"{manifest_hash}  manifest.json\n"
    )
    return qa


def _rebuild(frame: pd.DataFrame, prediction: str) -> pd.DataFrame:
    return _rebuild_all(frame)


def _rebuild_all(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    base = output["routed_prediction"].to_numpy(dtype=np.float64)
    persistence = output["persistence"].to_numpy(dtype=np.float64)
    lead = output["lead_h"].to_numpy(dtype=np.int64)
    output["incumbent_router_unshrunk"] = base
    active = np.isin(lead, np.array([12, 18, 24], dtype=np.int64))
    for name, weight in (
        ("current_shrink_20pct", 0.2),
        ("round_b_shrink_22p5pct", 0.225),
        ("round_a_shrink_25pct", 0.25),
    ):
        prediction = base.copy()
        prediction[active] = (1.0 - weight) * base[active] + weight * persistence[active]
        output[name] = prediction
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiments/p3_matched_budget_local_compare_20260825_v1.json",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Refusing independent QA without --execute")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "2")
    qa = execute(args.config.resolve())
    print(json.dumps({"status": qa["status"], "elapsed_seconds": qa["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
