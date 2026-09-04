"""Build the aggregate-only external-data breakthrough technical report.

Only sealed aggregate JSON receipts are loaded.  The builder never reads raw
external observations, competition rows, OOF parquet files, submissions, or
model checkpoints.  Mandatory evidence is pinned by SHA-256.  The optional P1
point-residual result is discovered only after it exists, and the exact bytes
used for the report are content-addressed in the report itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

KST = timezone(timedelta(hours=9))
REPORT_TITLE = "외부 데이터 돌파 실험 — 검증 결과와 다음 관문"
DEFAULT_OUTPUT = Path("reports/generated/external_data_breakthrough_2026-08-21/artifact.json")

RELATIVE_PATHS = {
    "permission": Path("configs/external_data/official_faq_permission.json"),
    "catalog": Path("configs/external_data/catalog.toml"),
    "p1_v1": Path("artifacts/p1_iors_external_precheck_v1/20260821T171120+0900/result.json"),
    "p1_v2": Path(
        "artifacts/p1_iors_external_profile_conformal_v2/20260821T171756+0900/result.json"
    ),
    "p2_nasa": Path("artifacts/p2_nasa_power_residual_meta_v1/result.json"),
    "p2_era5": Path("artifacts/p2_era5_primary_scaffold_v1/arco_metadata_receipt.json"),
    "p3_kma": Path("external_data/p3_kma_buoy_pre2024/status.json"),
}

EXPECTED_SHA256 = {
    "permission": "f4d2ea55461a8af2c1cf1ed1a1642224a93349a414dfa0af7b254606b7f64f62",
    "catalog": "08645af1c6238fff256d60580f99154ac89070d655bff5e70ca11925c1cb52a8",
    "p1_v1": "357ead1749a93e9c67701e3c8db8215d28f2a911fc5500038d0d4abf5c61233e",
    "p1_v2": "af5820c8f2ed645c71f9be886d0cf37a6344acb3ad8918ee49851a1b30688aab",
    "p2_nasa": "846a246f6af7fadb0b33be78bb45a46049f92528534c8368c2ab7c14070db75f",
    "p2_era5": "5d30078b1e380db6a0dacbb82b135e23f86ff1a5e5a3144a16f3589d94ff47a2",
    "p3_kma": "3fec3569acdfeb74c0f14391e5de00570f1bdba52c9c4555722410aab34d1b9a",
}

POINT_RESULT_DIR = Path("artifacts/p1_iors_external_point_residual_oof_v1")
POINT_RESULT_PATH = POINT_RESULT_DIR / "result.json"


class ReportEvidenceError(RuntimeError):
    """Raised when report evidence violates its aggregate-only sealed contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportEvidenceError(message)


def _close(
    actual: object,
    expected: float,
    label: str,
    *,
    tolerance: float = 1e-12,
) -> None:
    try:
        numeric = float(actual)
    except (TypeError, ValueError) as exc:
        raise ReportEvidenceError(f"{label} is not numeric: {actual!r}") from exc
    if not math.isclose(numeric, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ReportEvidenceError(f"{label} drifted: {numeric!r} != {expected!r}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_bytes(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReportEvidenceError(f"unable to read evidence: {path}") from exc
    digest = _sha256_bytes(payload)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportEvidenceError(f"evidence is not valid UTF-8 JSON: {path}") from exc
    _require(isinstance(value, dict), f"evidence root must be an object: {path}")
    return value, digest


def _read_sealed_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing required evidence: {label} ({path})")
    value, actual = _read_json_bytes(path)
    _require(
        actual == expected_sha256,
        f"sealed SHA mismatch for {label}: {actual} != {expected_sha256}",
    )
    return value


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _nested(value: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _first(value: Mapping[str, Any], paths: Iterable[Sequence[str]]) -> Any:
    for path in paths:
        candidate = _nested(value, path)
        if candidate is not None:
            return candidate
    return None


def _contains_true_mutation_flag(value: Any, *, prefix: str = "") -> str | None:
    """Return the first true mutation/upload flag found in a result receipt."""

    forbidden_fragments = (
        "submission_created",
        "submission_written",
        "submission_modified",
        "upload_allowed",
        "uploaded",
        "frozen_model_modified",
        "frozen_submission_modified",
        "model_or_submission_modified",
    )
    if isinstance(value, Mapping):
        for key, child in value.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            normalized = str(key).lower()
            if child is True and any(fragment in normalized for fragment in forbidden_fragments):
                return dotted
            found = _contains_true_mutation_flag(child, prefix=dotted)
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _contains_true_mutation_flag(child, prefix=f"{prefix}[{index}]")
            if found:
                return found
    return None


def _discover_point_result(root: Path) -> dict[str, Any]:
    """Discover one optional point-residual aggregate result without ambiguity."""

    result_root = root / POINT_RESULT_DIR
    if not result_root.exists():
        return {
            "status": "running",
            "decision": "RUNNING — P1 OOF 결과 대기",
            "path": None,
            "sha256": None,
            "duplicate_count": 0,
            "result": None,
        }

    candidates: list[Path]
    direct = root / POINT_RESULT_PATH
    if direct.is_file():
        candidates = [direct]
    else:
        candidates = sorted(result_root.glob("**/result.json"))
    if not candidates:
        return {
            "status": "running",
            "decision": "RUNNING — P1 OOF 결과 대기",
            "path": None,
            "sha256": None,
            "duplicate_count": 0,
            "result": None,
        }

    for path in candidates:
        _require(path.is_file(), f"optional P1 result is not a regular file: {path}")
        _require(_path_within(path, result_root), "optional P1 result escaped its directory")

    loaded = [_read_json_bytes(path) for path in candidates]
    unique_sha = {digest for _, digest in loaded}
    _require(
        len(unique_sha) == 1,
        "multiple non-identical P1 point-residual results are ambiguous; refusing report build",
    )
    result, digest = loaded[-1]
    _require(
        result.get("experiment_id") == "p1_iors_external_point_residual_oof_v1",
        "unexpected optional P1 experiment_id",
    )
    mutation_flag = _contains_true_mutation_flag(result)
    _require(mutation_flag is None, f"optional P1 result reports forbidden action: {mutation_flag}")

    decision = _first(
        result,
        (
            ("decision",),
            ("promotion", "decision"),
            ("gate", "decision"),
        ),
    )
    _require(isinstance(decision, str) and decision.strip(), "optional P1 decision is missing")
    passed = _first(result, (("promotion", "passed"), ("gate", "passed")))
    status = "complete" if passed in {True, False} else str(result.get("status", "complete"))

    weighted_delta = _first(
        result,
        (
            ("metrics", "overall_weighted_f1_delta"),
            ("metrics", "weighted_f1_delta"),
            ("promotion", "metrics", "overall_weighted_f1_delta"),
            ("gate", "diagnostics", "overall_weighted_f1_delta"),
        ),
    )
    if weighted_delta is None:
        baseline = _first(
            result,
            (
                ("metrics", "baseline", "weighted", "f1"),
                ("metrics", "incumbent", "weighted", "f1"),
            ),
        )
        candidate = _first(result, (("metrics", "candidate", "weighted", "f1"),))
        if baseline is not None and candidate is not None:
            weighted_delta = float(candidate) - float(baseline)
    if weighted_delta is not None:
        _require(math.isfinite(float(weighted_delta)), "optional P1 weighted delta is non-finite")

    relative = candidates[-1].relative_to(root).as_posix()
    return {
        "status": status,
        "decision": decision,
        "passed": passed,
        "weighted_f1_delta": None if weighted_delta is None else float(weighted_delta),
        "path": relative,
        "sha256": digest,
        "duplicate_count": len(candidates),
        "result": result,
    }


def collect_evidence(root: Path) -> dict[str, Any]:
    """Load and validate only sealed aggregate evidence."""

    root = root.resolve()
    resolved = {name: root / path for name, path in RELATIVE_PATHS.items()}
    for name, path in resolved.items():
        _require(_path_within(path, root), f"evidence escaped repository root: {name}")

    permission = _read_sealed_json(
        resolved["permission"], EXPECTED_SHA256["permission"], "permission"
    )
    # The catalog is TOML, so only its bytes are needed for this aggregate report.
    _require(resolved["catalog"].is_file(), "missing required evidence: catalog")
    _require(
        _sha256(resolved["catalog"]) == EXPECTED_SHA256["catalog"],
        "sealed SHA mismatch for catalog",
    )
    p1_v1 = _read_sealed_json(resolved["p1_v1"], EXPECTED_SHA256["p1_v1"], "p1_v1")
    p1_v2 = _read_sealed_json(resolved["p1_v2"], EXPECTED_SHA256["p1_v2"], "p1_v2")
    p2_nasa = _read_sealed_json(resolved["p2_nasa"], EXPECTED_SHA256["p2_nasa"], "p2_nasa")
    p2_era5 = _read_sealed_json(resolved["p2_era5"], EXPECTED_SHA256["p2_era5"], "p2_era5")
    p3_kma = _read_sealed_json(resolved["p3_kma"], EXPECTED_SHA256["p3_kma"], "p3_kma")

    _require(permission.get("status") == "approved", "external-data permission not approved")
    _require(
        permission.get("organizer_channel") == "official public FAQ API, id=9",
        "organizer permission channel drifted",
    )

    _require(
        p1_v1.get("experiment_id") == "p1_iors_external_loo_precheck_v1",
        "unexpected P1 v1 experiment",
    )
    _require(p1_v1["gate"]["decision"] == "NO_GO_EXTERNAL_PROFILE", "P1 v1 decision drifted")
    _require(p1_v1["gate"]["passed"] is False, "P1 v1 gate state drifted")
    _require(
        p1_v1["scope"]["competition_labels_opened"] is False,
        "P1 v1 competition labels opened",
    )
    _close(p1_v1["metrics"]["baseline"]["rmse"], 1.9363519368704971, "P1 v1 baseline RMSE")
    _close(p1_v1["metrics"]["candidate"]["rmse"], 0.9597394211774888, "P1 v1 candidate RMSE")
    _close(
        p1_v1["metrics"]["rmse_relative_improvement"],
        0.5043569286642152,
        "P1 v1 RMSE relative improvement",
    )
    _close(p1_v1["metrics"]["q10_q90"]["coverage"], 0.6930222035486895, "P1 v1 coverage")

    _require(
        p1_v2.get("experiment_id") == "p1_iors_external_profile_conformal_v2",
        "unexpected P1 v2 experiment",
    )
    _require(p1_v2["gate"]["decision"] == "NO_GO_EXTERNAL_PROFILE", "P1 v2 decision drifted")
    _require(p1_v2["gate"]["passed"] is False, "P1 v2 gate state drifted")
    _close(
        p1_v2["point_metrics"]["candidate"]["rmse"],
        1.171231330440692,
        "P1 v2 candidate RMSE",
    )
    _close(p1_v2["conformal_test"]["coverage"], 0.8112148928606057, "P1 v2 coverage")
    _require(
        p1_v2["v1_provenance"]["v1_result_sha256"] == EXPECTED_SHA256["p1_v1"],
        "P1 v2 no longer binds the sealed v1 result",
    )

    _require(
        p2_nasa.get("experiment_id") == "p2_nasa_power_residual_meta_v1",
        "unexpected P2 NASA experiment",
    )
    _require(p2_nasa.get("status") == "complete", "P2 NASA result incomplete")
    _require(
        p2_nasa["leakage_contract"]["hidden_2025_sep_oct_target_values_used"] is False,
        "P2 hidden target values were used",
    )
    _require(
        p2_nasa["leakage_contract"]["submission_created_or_modified"] is False,
        "P2 submission was modified",
    )
    nasa_metrics = p2_nasa["metrics"]["external_incremental_candidate_vs_control"]
    _close(nasa_metrics["delta_rmse"], 0.0, "P2 NASA incremental delta")
    _require(
        all(fold["candidate"]["selected_alpha"] == 0.0 for fold in p2_nasa["outer_folds"].values()),
        "P2 NASA outer folds no longer all select alpha zero",
    )

    _require(
        p2_era5.get("decision") == "NO_GO_ANONYMOUS_ARCO_TRANSFER", "ERA5 transfer decision drifted"
    )
    _require(p2_era5.get("data_array_read") is False, "ERA5 array values were unexpectedly read")
    _require(
        p2_era5.get("model_or_submission_modified") is False,
        "ERA5 scaffold modified model or submission",
    )
    _close(p2_era5["transfer_gate"]["estimated_full_gib"], 112.98942985013127, "ERA5 transfer GiB")
    _close(
        p2_era5["transfer_gate"]["estimated_full_hours_at_50mbps"],
        5.142274496290419,
        "ERA5 transfer hours",
    )
    _require(p2_era5["transfer_gate"]["passed"] is False, "ERA5 transfer gate drifted")

    _require(p3_kma.get("status") == "awaiting_credential", "P3 KMA status drifted")
    _require(
        p3_kma.get("next_action") == "set KMA_API_KEY in the process environment",
        "P3 next action drifted",
    )
    _require(
        not any(bool(value) for value in p3_kma["safety_invariants"].values()),
        "P3 KMA scaffold crossed a safety invariant",
    )

    return {
        "permission": permission,
        "p1_v1": p1_v1,
        "p1_v2": p1_v2,
        "p1_point": _discover_point_result(root),
        "p2_nasa": p2_nasa,
        "p2_era5": p2_era5,
        "p3_kma": p3_kma,
        "hashes": dict(EXPECTED_SHA256),
    }


def _sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        _require(math.isfinite(float(value)), "non-finite value cannot enter report SQL")
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _values_sql(rows: list[dict[str, object]], columns: list[str]) -> str:
    return " UNION ALL ".join(
        "SELECT " + ", ".join(f"{_sql_literal(row.get(column))} AS {column}" for column in columns)
        for row in rows
    )


def _inline_source(
    rows: list[dict[str, object]],
    columns: list[str],
    *,
    description: str,
) -> dict[str, object]:
    return {
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "description": description,
            "sql": _values_sql(rows, columns),
        }
    }


def _table(
    *,
    table_id: str,
    title: str,
    subtitle: str,
    rows: list[dict[str, object]],
    columns: list[dict[str, object]],
    default_sort: tuple[str, str],
) -> dict[str, object]:
    fields = [str(column["field"]) for column in columns]
    return {
        "id": table_id,
        "title": title,
        "subtitle": subtitle,
        "showDescription": True,
        "dataset": table_id,
        "sourceId": "aggregate_receipt",
        "source": _inline_source(
            rows,
            fields,
            description="Reviewed aggregate rows assembled from the sealed receipts listed in source metadata.",
        ),
        "density": "spacious",
        "layout": "full",
        "defaultSort": {"field": default_sort[0], "direction": default_sort[1]},
        "columns": columns,
    }


def _point_effect(point: Mapping[str, Any]) -> str:
    if point["status"] == "running":
        return "결과 파일 없음; P1 nested OOF 실행 대기"
    delta = point.get("weighted_f1_delta")
    if delta is None:
        return "완료; headline delta는 결과 스키마에 없음"
    return f"overall weighted F1 delta {float(delta):+.6f}"


def build_artifact(evidence: Mapping[str, Any], *, generated_at: str) -> dict[str, Any]:
    """Build one canonical portable-report artifact from aggregate evidence."""

    p1_v1 = evidence["p1_v1"]
    point = evidence["p1_point"]

    layer_rows = []
    for layer, values in p1_v1["metrics"]["per_layer"].items():
        layer_rows.append(
            {
                "layer": f"Layer {layer}",
                "layer_number": int(layer),
                "rmse_relative_improvement": float(values["rmse_relative_improvement"]),
                "baseline_rmse_c": float(values["baseline"]["rmse"]),
                "candidate_rmse_c": float(values["candidate"]["rmse"]),
                "rows": int(values["rows"]),
                "direction": "improved"
                if float(values["rmse_relative_improvement"]) >= 0
                else "degraded",
            }
        )
    layer_rows.sort(key=lambda row: float(row["rmse_relative_improvement"]), reverse=True)

    decision_rows = [
        {
            "order": 1,
            "problem": "P1",
            "experiment": "I-ORS external profile q50/q10/q90 v1",
            "stage": "external-only 2023 holdout",
            "scope": "154,705 QC=1 rows; not P1 OOF",
            "headline": "RMSE 1.936352→0.959739°C; +50.44%",
            "gate_or_blocker": "coverage 0.6930 < 0.70",
            "decision": "NO_GO_EXTERNAL_PROFILE",
            "next_action": "q50 point signal만 P1 OOF에서 별도 검증",
        },
        {
            "order": 2,
            "problem": "P1",
            "experiment": "I-ORS conformal profile v2",
            "stage": "2022 calibrate / 2023 test",
            "scope": "154,705 test rows; non-independent follow-up",
            "headline": "coverage 0.8112; RMSE +39.51%",
            "gate_or_blocker": "4/6 layers non-worse; worst L2 −13.60%",
            "decision": "NO_GO_EXTERNAL_PROFILE",
            "next_action": "interval family는 중단; point-only 계약 유지",
        },
        {
            "order": 3,
            "problem": "P1",
            "experiment": "I-ORS point-residual nested OOF v1",
            "stage": "non-virgin P1 outer one-shot",
            "scope": "I-ORS eligible rows only; S/G unchanged",
            "headline": _point_effect(point),
            "gate_or_blocker": "8개 point-residual feature; fixed gate",
            "decision": str(point["decision"]),
            "next_action": "결과 gate 통과 전 동결 모델 변경 금지",
        },
        {
            "order": 4,
            "problem": "P2",
            "experiment": "NASA POWER residual-meta v1",
            "stage": "3-block nested OOF",
            "scope": "69,850 rows; 163 KST days",
            "headline": "incremental ΔRMSE 0.000000°C",
            "gate_or_blocker": "모든 outer fold alpha=0; CI90 [0,0]",
            "decision": "NO_GO_NASA_POWER_RESIDUAL_META_V1",
            "next_action": "현재 NASA residual-meta v1만 중단; ERA5는 별도 가설",
        },
        {
            "order": 5,
            "problem": "P2",
            "experiment": "ERA5 surface-flux primary scaffold",
            "stage": "metadata-only transfer gate",
            "scope": "4,900 planned hours; data arrays 0",
            "headline": "anonymous ARCO estimate 112.989 GiB / 5.142 h",
            "gate_or_blocker": "5 GiB·2 h cap 초과; CDS credential 없음",
            "decision": "NO_GO_ANONYMOUS_ARCO_TRANSFER; ERA5 모델 미평가",
            "next_action": "CDSAPI_KEY + ERA5_CDS_TERMS_ACCEPTED",
        },
        {
            "order": 6,
            "problem": "P3",
            "experiment": "KMA buoy pre-2024 scaffold",
            "stage": "credential-safe preflight",
            "scope": "30-minute WH_SIG; pre-2024 cutoff",
            "headline": "외부 값·모델·test context 접근 0",
            "gate_or_blocker": "KMA API credential 없음",
            "decision": "AWAITING_CREDENTIAL",
            "next_action": "KMA_API_KEY 설정 후 6시간 smoke",
        },
    ]

    source_rows = [
        {
            "order": 1,
            "source": "Organizer public external-data FAQ",
            "problem": "P1/P2/P3",
            "rights": "공개 외부자료 허용; 출처 표기 필요",
            "cutoff": "problem-specific receipt",
            "access": "approved",
            "value_accessed": "N/A",
            "current_use": "permission gate",
        },
        {
            "order": 2,
            "source": "KIOST I-ORS CTD v1.1.1",
            "problem": "P1",
            "rights": "CC BY 4.0; DOI 10.22808/DATA-2024-6",
            "cutoff": "2023-12-31 23:50 KST",
            "access": "downloaded + audited",
            "value_accessed": "yes",
            "current_use": "external profile + point-residual OOF",
        },
        {
            "order": 3,
            "source": "NASA POWER hourly meteorology",
            "problem": "P2",
            "rights": "free public data; attribution requested",
            "cutoff": "2025-12-31 14:00 UTC",
            "access": "17,535 hourly rows validated",
            "value_accessed": "yes",
            "current_use": "residual-meta v1 NO-GO",
        },
        {
            "order": 4,
            "source": "ERA5 hourly single levels",
            "problem": "P2",
            "rights": "Copernicus attribution; DOI 10.24381/cds.adbb2d47",
            "cutoff": "planned historical OOF chunks",
            "access": "metadata only",
            "value_accessed": "no",
            "current_use": "anonymous transfer NO-GO; CDS pending",
        },
        {
            "order": 5,
            "source": "KMA ocean meteorological buoy",
            "problem": "P3",
            "rights": "Korea Open Government Licence Type 1",
            "cutoff": "2023-12-31 23:59 KST",
            "access": "credential pending",
            "value_accessed": "no",
            "current_use": "preflight only",
        },
    ]

    evidence_rows = [
        {
            "evidence": key,
            "path": RELATIVE_PATHS[key].as_posix(),
            "sha256": evidence["hashes"][key],
            "seal": "fixed expected SHA-256",
        }
        for key in ("permission", "catalog", "p1_v1", "p1_v2", "p2_nasa", "p2_era5", "p3_kma")
    ]
    if point.get("path"):
        evidence_rows.append(
            {
                "evidence": "p1_point",
                "path": str(point["path"]),
                "sha256": str(point["sha256"]),
                "seal": "build-time content address; single unique result",
            }
        )

    chart = {
        "id": "p1_v1_layer_rmse_improvement",
        "title": "P1 v1 층별 RMSE 상대 개선",
        "subtitle": (
            "I-ORS 2023 external-only holdout, 154,705 QC=1 rows; "
            "양수는 depth-linear baseline 대비 개선"
        ),
        "showDescription": True,
        "intent": "comparison",
        "question": "외부 q50 profile model의 point reconstruction 이득이 층 전반에 유지되는가?",
        "rationale": "6개 긴 범주 라벨과 부호 있는 단일 지표의 순위를 읽기 위한 수평 막대.",
        "type": "horizontalBar",
        "dataset": "p1_v1_layer_rmse_improvement",
        "sourceId": "p1_v1_result",
        "source": _inline_source(
            layer_rows,
            [
                "layer",
                "layer_number",
                "rmse_relative_improvement",
                "baseline_rmse_c",
                "candidate_rmse_c",
                "rows",
                "direction",
            ],
            description="Layer-level aggregate metrics copied from the sealed P1 v1 result.",
        ),
        "valueFormat": "percent",
        "layout": "full",
        "maxRows": 6,
        "settings": {
            "orientation": "horizontal",
            "sort": "descending",
            "showValues": True,
            "groupMode": "single",
        },
        "referenceLines": [
            {
                "axis": "y",
                "value": 0,
                "label": "no change",
                "color": "neutral",
                "lineStyle": "solid",
            }
        ],
        "encodings": {
            "x": {"field": "layer", "type": "nominal", "label": "I-ORS layer"},
            "y": {
                "field": "rmse_relative_improvement",
                "type": "quantitative",
                "label": "RMSE relative improvement",
                "format": "percent",
            },
            "tooltip": [
                {
                    "field": "baseline_rmse_c",
                    "type": "quantitative",
                    "label": "Baseline RMSE",
                    "unit": "°C",
                },
                {
                    "field": "candidate_rmse_c",
                    "type": "quantitative",
                    "label": "Candidate RMSE",
                    "unit": "°C",
                },
                {"field": "rows", "type": "quantitative", "label": "QC=1 rows"},
                {"field": "direction", "type": "nominal", "label": "Direction"},
            ],
        },
    }

    decision_table = _table(
        table_id="decision_register",
        title="문제·가설별 현재 판정",
        subtitle="검증 단계와 지표가 다른 실험을 하나의 점수로 합치지 않은 exact decision register",
        rows=decision_rows,
        default_sort=("order", "asc"),
        columns=[
            {"field": "problem", "label": "문제", "type": "text"},
            {"field": "experiment", "label": "가설/실험", "type": "text"},
            {"field": "stage", "label": "검증 단계", "type": "text"},
            {"field": "scope", "label": "표본·범위", "type": "text"},
            {"field": "headline", "label": "관측 효과", "type": "text"},
            {"field": "gate_or_blocker", "label": "실패 gate/차단", "type": "text"},
            {"field": "decision", "label": "판정", "type": "text"},
            {"field": "next_action", "label": "다음 행동", "type": "text"},
            {"field": "order", "label": "순서", "type": "number"},
        ],
    )
    source_table = _table(
        table_id="source_readiness",
        title="외부 소스 권리·접근·사용 상태",
        subtitle="공식 허용 여부, 보수적 cutoff, 실제 값 접근 여부와 현재 실험 용도를 분리",
        rows=source_rows,
        default_sort=("order", "asc"),
        columns=[
            {"field": "source", "label": "소스", "type": "text"},
            {"field": "problem", "label": "적용 문제", "type": "text"},
            {"field": "rights", "label": "권리/표기", "type": "text"},
            {"field": "cutoff", "label": "시간 cutoff", "type": "text"},
            {"field": "access", "label": "접근 상태", "type": "text"},
            {"field": "value_accessed", "label": "값 접근", "type": "text"},
            {"field": "current_use", "label": "현재 용도", "type": "text"},
            {"field": "order", "label": "순서", "type": "number"},
        ],
    )
    integrity_table = _table(
        table_id="evidence_integrity",
        title="보고서 입력 증거의 내용주소",
        subtitle="필수 입력은 기대 SHA와 byte-for-byte 일치해야 하며, 선택 P1 결과는 단일 byte snapshot으로 봉인",
        rows=evidence_rows,
        default_sort=("evidence", "asc"),
        columns=[
            {"field": "evidence", "label": "증거", "type": "text"},
            {"field": "path", "label": "프로젝트 상대경로", "type": "text"},
            {"field": "sha256", "label": "SHA-256", "type": "text"},
            {"field": "seal", "label": "봉인 방식", "type": "text"},
        ],
    )

    source_paths = [path.as_posix() for path in RELATIVE_PATHS.values()]
    if point.get("path"):
        source_paths.append(str(point["path"]))
    sources: list[dict[str, Any]] = [
        {
            "id": "aggregate_receipt",
            "label": "외부 데이터 돌파 실험 aggregate-only evidence",
            "path": "scripts/build_external_data_breakthrough_report.py",
            "query": {
                "description": "Sealed aggregate receipts only; no raw observation, OOF row, submission, model, or hidden target values.",
                "tables_used": source_paths,
                "filters": [
                    "raw external observation rows = 0",
                    "competition row-level values = 0",
                    "OOF parquet rows = 0",
                    "submissions and model checkpoints read = 0",
                ],
            },
        },
        {
            "id": "organizer_permission",
            "label": "Official public FAQ external-data permission receipt",
            "path": RELATIVE_PATHS["permission"].as_posix(),
            "href": "https://oceanaidata.org/api/faqs",
            "query": {
                "description": "Organizer FAQ id=9 permission receipt; public external data allowed with source attribution.",
                "tables_used": [RELATIVE_PATHS["permission"].as_posix()],
                "filters": ["status = approved", "FAQ id = 9"],
            },
        },
        {
            "id": "p1_v1_result",
            "label": "P1 I-ORS external profile precheck v1",
            "path": RELATIVE_PATHS["p1_v1"].as_posix(),
            "href": "https://sciwatch.kiost.ac.kr/handle/2020.kiost/46422",
            "query": {
                "description": "Aggregate 2023 external-only point and interval metrics by layer.",
                "tables_used": [RELATIVE_PATHS["p1_v1"].as_posix()],
                "filters": ["QC = 1", "2023 holdout", "competition labels opened = false"],
                "metric_definitions": [
                    "RMSE relative improvement = (depth-linear baseline RMSE - candidate RMSE) / baseline RMSE",
                    "positive values indicate lower candidate RMSE",
                ],
            },
        },
        {
            "id": "p1_v2_result",
            "label": "P1 I-ORS conformal profile follow-up v2",
            "path": RELATIVE_PATHS["p1_v2"].as_posix(),
            "href": "https://sciwatch.kiost.ac.kr/handle/2020.kiost/46422",
            "query": {
                "description": "Aggregate 2022 conformal calibration and 2023 test metrics.",
                "tables_used": [RELATIVE_PATHS["p1_v2"].as_posix()],
                "filters": ["2014-2021 fit", "2022 calibrate", "2023 test"],
            },
        },
        {
            "id": "p2_nasa_result",
            "label": "P2 NASA POWER residual-meta v1",
            "path": RELATIVE_PATHS["p2_nasa"].as_posix(),
            "href": "https://power.larc.nasa.gov/docs/services/api/temporal/hourly/",
            "query": {
                "description": "Aggregate three-block nested OOF metrics for the NASA POWER residual-meta v1 experiment.",
                "tables_used": [RELATIVE_PATHS["p2_nasa"].as_posix()],
                "filters": [
                    "69,850 OOF rows",
                    "held-fold alpha selection prohibited",
                    "hidden inference = false",
                ],
            },
        },
        {
            "id": "p2_era5_result",
            "label": "P2 ERA5 anonymous ARCO transfer receipt",
            "path": RELATIVE_PATHS["p2_era5"].as_posix(),
            "href": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview",
            "query": {
                "description": "Metadata-only transfer-cost receipt; no ERA5 data array was read.",
                "tables_used": [RELATIVE_PATHS["p2_era5"].as_posix()],
                "filters": ["data_array_read = false", "5 GiB cap", "2 hour cap"],
            },
        },
        {
            "id": "p3_kma_status",
            "label": "P3 KMA buoy credential-safe status",
            "path": RELATIVE_PATHS["p3_kma"].as_posix(),
            "href": "https://apihub.kma.go.kr/apiList.do?seqApi=3",
            "query": {
                "description": "Credential-safe preparation status; no KMA value or P3 test context was read.",
                "tables_used": [RELATIVE_PATHS["p3_kma"].as_posix()],
                "filters": ["status = awaiting_credential", "source cutoff <= 2023-12-31 KST"],
            },
        },
    ]
    if point.get("path"):
        sources.append(
            {
                "id": "p1_point_result",
                "label": "P1 I-ORS point-residual nested OOF result",
                "path": str(point["path"]),
                "query": {
                    "description": "Optional aggregate result discovered after completion and content-addressed at build time.",
                    "tables_used": [str(point["path"])],
                    "filters": [
                        "non-virgin follow-up",
                        "frozen model mutation = false",
                        "upload = false",
                    ],
                },
            }
        )

    point_summary = (
        "P1 point-residual OOF가 아직 실행 중이므로 승격 결론은 열려 있습니다."
        if point["status"] == "running"
        else f"P1 point-residual OOF 결과는 `{point['decision']}`이며 {_point_effect(point)}입니다."
    )
    point_scope = (
        "세 chronological outer fold에서 I-ORS eligible 행만 새 모델로 대체하는 사전 계약입니다."
        if point["status"] == "running"
        else "Q2는 eligible 학습 이력이 없어 exact incumbent no-op으로 유지했고, Q3·Q4의 I-ORS eligible 행만 새 모델로 대체했습니다."
    )
    point_next_step = (
        "현재 point-residual nested OOF 한 번을 끝내고 사전 고정 gate 전체를 판정합니다."
        if point["status"] == "running"
        else "동일 outer OOF에서 threshold를 다시 맞추지 않습니다. 후속이 필요하면 incumbent를 전면 교체하지 않는 외부-only 합성 anomaly gate를 먼저 검증하고, 이번 결과는 가설 생성용으로만 사용합니다."
    )
    point_limitation = (
        ""
        if point["status"] == "running"
        else "\n- **P1 point-residual OOF는 외부 q50 표현 자체가 아니라 I-ORS 전용 wholesale replacement 통합을 기각합니다.** Q2는 no-op이었고 Q3·Q4 모두 악화했으므로 같은 outer 결과를 보고 임계값만 다시 고르는 것은 과적합입니다."
    )
    point_question = (
        "P1 q50 residual이 실제 offset/drift recall을 늘리면서 normal FP/day와 Layer 2 F1 guard를 동시에 지키는가?"
        if point["status"] == "running"
        else "P1 외부 q50을 incumbent 전면 교체가 아닌 외부-only 합성 anomaly gate로 학습하면 계절 전이와 정상 FP를 함께 제어할 수 있는가?"
    )
    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {REPORT_TITLE}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## 기술 요약 — 외부 데이터는 허용되지만 아직 동결 모델을 바꿀 증거는 없습니다\n\n"
                "- **가장 강한 신호는 P1 I-ORS q50 point reconstruction입니다.** 2023 external-only holdout에서 depth-linear baseline 대비 RMSE가 50.44% 낮았고 6개 층 중 5개가 비열화였습니다. 다만 이 단계는 P1 F1 검증이 아니며, v1 interval coverage와 v2 layer guard가 실패했습니다.\n"
                f"- **{point_summary}** 이 후속은 v1/v2 관찰 뒤 정의된 non-virgin 가설이므로 통과하더라도 독립 재현으로 해석하지 않습니다.\n"
                "- **P2 NASA POWER residual-meta v1은 증분 효과가 정확히 0이었습니다.** 이 결론은 해당 모델·특징·검증 계약에만 적용되며 ERA5 surface-flux 가설을 기각하지 않습니다. 익명 ERA5 ARCO 경로는 112.989 GiB 추정으로 전송 gate만 실패했습니다.\n"
                "- **P3 KMA는 아직 모델 결과가 없습니다.** `KMA_API_KEY`가 없어 값 접근·훈련·test-context 접근이 모두 0이며, credential 뒤 6시간 smoke부터 시작해야 합니다.\n"
                "- **동결 모델·제출은 변경하지 않았습니다.** 보고서 입력은 고정 SHA의 집계 JSON뿐이며 원시 외부 관측, row-level OOF, hidden target, submission CSV, model checkpoint를 읽지 않습니다."
            ),
        },
        {
            "id": "visual_finding",
            "type": "markdown",
            "sourceId": "p1_v1_result",
            "body": (
                "## P1의 point 신호는 크지만 얕은 Layer 2 반례가 남습니다\n\n"
                "전체 q50 RMSE 개선은 50.44%였으나 층별로는 Layer 7이 +67.87%, Layer 5가 +49.64%인 반면 Layer 2는 −2.29%였습니다. 아래 수평 막대는 점 예측의 이질성을 그대로 보여줍니다. 양수는 개선, 음수는 악화이며 막대 길이를 공식 P1 F1로 해석해서는 안 됩니다. **시사점:** interval을 버리고 point residual만 P1 OOF 특징으로 검증하는 것은 근거가 있지만, 층별 guard 없이는 승격할 수 없습니다."
            ),
        },
        {"id": "layer_chart", "type": "chart", "chartId": chart["id"]},
        {
            "id": "decision_intro",
            "type": "markdown",
            "body": (
                "## 현재 판정은 한 개의 ‘외부 데이터 효과’가 아니라 여섯 개의 서로 다른 관문입니다\n\n"
                "P1 external-only RMSE, P1 nested OOF F1, P2 pooled RMSE, ERA5 전송 가능성, P3 credential 상태는 같은 척도가 아닙니다. 아래 표는 단계·표본·실패 조건을 분리해, 아직 실행되지 않은 모델을 성능 실패로 오인하지 않도록 합니다."
            ),
        },
        {"id": "decision_table_block", "type": "table", "tableId": decision_table["id"]},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "body": (
                "## 범위·데이터·지표 정의\n\n"
                "- **P1 v1/v2 population:** KIOST I-ORS CTD의 QC=1 수온만 사용한 2023 holdout 154,705행입니다. Baseline은 같은 시각 공개 peer 층을 잇는 depth-linear prediction이며, RMSE relative improvement는 `(baseline RMSE − candidate RMSE) / baseline RMSE`입니다.\n"
                f"- **P1 point-residual OOF:** {point_scope} S-ORS, G-ORS, 비대상 I-ORS 예측은 incumbent와 동일하며 공식 hidden F1이 아닙니다.\n"
                "- **P2 NASA metric:** 69,850 OOF 행의 external candidate minus no-external control ΔRMSE(°C)입니다. 음수가 개선입니다. Alpha는 outer-held label을 보지 않는 inner fold에서만 선택했습니다.\n"
                "- **ERA5 transfer metric:** 4,900 planned hours × time-chunked global store의 추정 전송량·시간입니다. 모델 RMSE나 ERA5의 물리적 유용성을 측정하지 않습니다.\n"
                "- **P3 status:** pre-2024 KMA WH_SIG를 30분 cadence로 가져오기 전 credential gate입니다. 예측 RMSE는 아직 존재하지 않습니다."
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "body": (
                "## 모델·실험 설계 — 값 접근과 승격을 단계별로 분리했습니다\n\n"
                "1. **권리·누출 gate:** 주최측 공개 FAQ 허용, 소스별 라이선스, 시간 cutoff를 먼저 고정했습니다. P1은 KIOST v1.1.1 archive의 2014–2023 QC=1만 사용했고, P3는 2023년 말 이후를 금지했습니다.\n"
                "2. **P1 external representation:** 목표 수온을 mask한 뒤 같은 시각 peer 수온·염분·depth와 계절/시각 특징으로 LightGBM q50을 학습했습니다. V1은 2014–2022 fit/2023 holdout, v2는 2014–2021 fit/2022 conformal calibration/2023 test입니다.\n"
                "3. **P1 OOF bridge:** 후속 가설은 q10/q90을 제외하고 signed/absolute q50 residual, peer support, 24/72시간 gap-safe median·slope 8개만 추가했습니다. Q2는 외부 eligible 학습 이력이 없어 exact no-op, Q3·Q4만 단일 XGBoost와 inner-only threshold/early stopping으로 평가했습니다.\n"
                "4. **P2 residual-meta:** frozen P2 prediction residual을 control과 NASA covariate candidate가 각각 보정하도록 3-block nested OOF를 구성했습니다. 세 outer 선택이 모두 alpha=0이어서 최종 예측은 incumbent와 정확히 같았습니다.\n"
                "5. **ERA5/KMA preflight:** 값 다운로드 전에 전송 비용과 credential을 gate로 두었습니다. ERA5 익명 time-chunked store는 비용으로 중단했고, KMA는 key 부재로 network/value access 전에 중단했습니다."
            ),
        },
        {
            "id": "source_intro",
            "type": "markdown",
            "sourceId": "organizer_permission",
            "body": (
                "## 외부 소스는 허용·권리·시간·값 접근을 별도로 추적합니다\n\n"
                "주최측 FAQ id=9는 공개 외부 데이터 사용을 허용하되 출처 표기를 요구합니다. 허용은 곧 유용성을 뜻하지 않으므로, 아래 표는 법적/운영적 준비와 실제 검증 상태를 분리합니다."
            ),
        },
        {"id": "source_table_block", "type": "table", "tableId": source_table["id"]},
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 한계·불확실성·강건성 점검\n\n"
                "- **P1 v1은 point 효과가 커도 provenance confidence가 0.72입니다.** 실행 시 코드 SHA와 prediction artifact가 결과에 완전히 암호학적으로 결합되지 않았고, 동일 metrics를 낸 두 v1 run의 duplicate-run 기록이 정식 봉인되지 않았습니다. 독립 QA의 판정 confidence는 0.93이지만 이는 provenance 결함을 없애지 않습니다.\n"
                "- **P1 v2는 같은 2023을 다시 본 non-independent follow-up입니다.** Coverage 수정은 재현됐고 판정 confidence는 0.98이지만, Layer 1·2 악화로 point layer guard가 실패했습니다."
                f"{point_limitation}\n"
                "- **P2 독립 QA는 수치·key·held-fold selection을 0.99 confidence로 재현했습니다.** 결론은 `NASA_POWER_RESIDUAL_META_V1`의 현재 설계에만 한정합니다. NASA의 모든 물리 신호나 ERA5 surface stress/flux를 기각하지 않습니다.\n"
                "- **ERA5의 NO-GO는 데이터가 아니라 접근 경로입니다.** 익명 ARCO time chunk가 global 721×1440 grid여서 비용 gate를 넘었을 뿐, geo-chunked CDS 자료로 모델 검증한 결과가 아닙니다.\n"
                "- **P3에는 score evidence가 없습니다.** Credential이 생기더라도 station/domain shift, 78시간 독립 anchor, 최소 표본 gate를 통과해야 모델 실험을 시작할 수 있습니다.\n"
                "- 모든 로컬 검증은 official hidden leaderboard 성능을 보증하지 않으며, 이 보고서는 frozen submission 변경 권한을 부여하지 않습니다."
            ),
        },
        {"id": "integrity_table_block", "type": "table", "tableId": integrity_table["id"]},
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## 권고 다음 단계 — 계산보다 먼저 남은 정보 병목을 엽니다\n\n"
                f"- **P1:** {point_next_step} 동결 모델은 변경하지 않습니다.\n"
                "- **P2:** `CDSAPI_KEY`와 `ERA5_CDS_TERMS_ACCEPTED`를 설정한 뒤 geo-chunked/CDS 소지역 자료로 작은 smoke를 실행합니다. NASA 0효과를 근거로 ERA5를 생략하지 않습니다.\n"
                "- **P3:** `KMA_API_KEY`를 process environment에만 설정하고 6시간 smoke→cadence/schema QC→pre-2024 anchor precheck 순서로 진행합니다.\n"
                "- **공통:** frozen model·submission은 승격 gate와 독립 재현 전까지 그대로 유지하고, 새 외부 source는 라이선스·cutoff·SHA·값 접근 상태를 catalog에 먼저 기록합니다."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## 다음 판단을 바꿀 질문\n\n"
                f"- {point_question}\n"
                "- P2 ERA5의 surface stress·net radiation·latent/sensible heat flux가 NASA coarse meteorology에 없던 계절 전이 정보를 제공하는가?\n"
                "- P3 KMA buoy의 station/domain 차이를 보정한 pretraining이 +18/+24시간 RMSE를 줄이는가, 아니면 local persistence skill을 희석하는가?\n"
                "- 외부 소스의 이득이 평균뿐 아니라 fold·layer·station·event-type 최소 성능에서 유지되는가?"
            ),
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": REPORT_TITLE,
            "description": "P1/P2/P3 외부 데이터 실험의 aggregate-only 기술 검증 보고서",
            "generatedAt": generated_at,
            "filters": [],
            "cards": [],
            "charts": [chart],
            "tables": [decision_table, source_table, integrity_table],
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "p1_v1_layer_rmse_improvement": layer_rows,
                "decision_register": decision_rows,
                "source_readiness": source_rows,
                "evidence_integrity": evidence_rows,
            },
            "accessIssues": [],
        },
        "sources": [
            {
                "id": source["id"],
                "label": source["label"],
                "path": source["path"],
                **({"href": source["href"]} if "href" in source else {}),
            }
            for source in sources
        ],
        "package_info": {
            "originUrl": "artifact://external-data-breakthrough-2026-08-21",
            "controls": {"edit": False, "refresh": False},
        },
    }
    _validate_aggregate_artifact(artifact)
    return artifact


def _validate_aggregate_artifact(artifact: Mapping[str, Any]) -> None:
    _require(artifact["surface"] == "report", "surface must be report")
    manifest = artifact["manifest"]
    _require(manifest["title"] == REPORT_TITLE, "manifest title mismatch")
    _require(manifest["blocks"][0]["body"] == f"# {REPORT_TITLE}", "title block mismatch")
    _require(len(manifest["charts"]) == 1, "report must contain exactly one chart")
    _require(manifest["charts"][0]["type"] == "horizontalBar", "chart must be horizontal")
    rows = artifact["snapshot"]["datasets"]["p1_v1_layer_rmse_improvement"]
    _require(len(rows) == 6, "P1 layer chart must contain six aggregate layers")
    _require(
        sum(float(row["rmse_relative_improvement"]) >= 0 for row in rows) == 5,
        "P1 layer direction drifted",
    )
    _require(len(manifest["tables"]) == 3, "decision, source, and integrity tables required")

    required_sections = {
        "technical_summary",
        "visual_finding",
        "scope_definitions",
        "methodology",
        "limitations",
        "next_steps",
        "further_questions",
    }
    block_ids = {block["id"] for block in manifest["blocks"]}
    _require(required_sections <= block_ids, "technical report structure is incomplete")

    serialized = json.dumps(artifact, ensure_ascii=False)
    for forbidden in (
        "C:/Users/",
        "C:\\Users\\",
        "station,time,temp",
        "case_id,station,lead_h",
        "target_hs",
        "incumbent_probability",
        "candidate_probability",
    ):
        _require(forbidden not in serialized, f"forbidden row-level or local content: {forbidden}")
    for source in artifact["sources"]:
        _require(not Path(source["path"]).is_absolute(), "absolute source path leaked")
        _require(".." not in Path(source["path"]).parts, "parent traversal leaked")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generated-at", default=None)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    _require(_path_within(output, root), "output must stay within repository root")
    generated_at = args.generated_at or datetime.now(KST).isoformat(timespec="seconds")
    evidence = collect_evidence(root)
    artifact = build_artifact(evidence, generated_at=generated_at)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": output.relative_to(root).as_posix(),
                "sha256": _sha256(output),
                "p1_point_status": evidence["p1_point"]["status"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
