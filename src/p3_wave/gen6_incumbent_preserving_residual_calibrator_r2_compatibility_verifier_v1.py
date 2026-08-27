"""Read-only compatibility verification for the immutable P3 Gen6r2 attempt.

The frozen r2 verifier had one incorrect learning-curve prefix constant.  This
module derives the real constant from four independently pinned persisted
sources, admits only the exact historical post-publish failure receipt, and
then re-executes the otherwise unchanged frozen verifier.  It additionally
reconciles sealed OOF truth and independently recomputes the five 5,000-draw
bootstrap points and the research-only gate.  No function in this module has a
write, execution, fitting, prediction, source-target parsing, promotion, ledger,
or upload capability.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final

R2_MODULE: Final = "p3_wave.gen6_incumbent_preserving_residual_calibrator_contract_r2"
R2_RELATIVE: Final = "gen6_incumbent_preserving_residual_calibrator_contract_r2.py"
CONFIG_RELATIVE: Final = (
    "configs/experiments/"
    "p3_gen6_incumbent_preserving_residual_calibrator_v1r2_compatibility_verifier_v1.json"
)
CONFIG_SHA256: Final = "6b92a6eb67adfb042958cb518633ead4e2c70ffb1e7de35eceafccd6c6e42d2a"
IDENTITY: Final = "P3_GEN6_INCUMBENT_PRESERVING_RESIDUAL_CALIBRATOR_R2_COMPATIBILITY_VERIFIER_V1"
IMPLEMENTATION_ROLES: Final = {
    "CONFIG": CONFIG_RELATIVE,
    "HELPER": (
        "src/p3_wave/gen6_incumbent_preserving_residual_calibrator_r2_compatibility_verifier_v1.py"
    ),
    "CLI": (
        "scripts/verify_p3_gen6_incumbent_preserving_residual_calibrator_r2_compatibility_v1.py"
    ),
    "TESTS": (
        "tests/"
        "test_p3_gen6_incumbent_preserving_residual_calibrator_"
        "r2_compatibility_verifier_v1.py"
    ),
}
INVENTORY_ALGORITHM: Final = "SHA256_CANONICAL_JSON_SORTED_RELATIVE_PATH_TYPE_BYTES_SHA256_WITH_LF"
OFFICIAL_LEADS: Final = (3, 6, 9, 12, 18, 24)
FOLD_ORDER: Final = ("2024_h2_storm", "winter_transition", "2025_h1")
REPARSE_ATTRIBUTE: Final = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
LOWER_SHA_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_PATCH_LOCK = threading.RLock()


class CompatibilityVerifierError(RuntimeError):
    """The frozen compatibility-verification contract was not satisfied."""


def _load_r2_contract() -> Any:
    path = Path(__file__).resolve(strict=True).with_name(R2_RELATIVE)
    existing = sys.modules.get(R2_MODULE)
    if existing is not None:
        existing_path = Path(existing.__file__).resolve(strict=True)
        if existing_path != path:
            raise CompatibilityVerifierError("noncanonical r2 contract is already imported")
        return existing
    spec = importlib.util.spec_from_file_location(R2_MODULE, path)
    if spec is None or spec.loader is None:
        raise CompatibilityVerifierError("cannot load the frozen r2 contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[R2_MODULE] = module
    spec.loader.exec_module(module)
    return module


r2 = _load_r2_contract()
np = r2.np
pq = r2.pq


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _has_reparse(path: Path) -> bool:
    info = path.lstat()
    attributes = int(getattr(info, "st_file_attributes", 0))
    return path.is_symlink() or bool(attributes & REPARSE_ATTRIBUTE)


def _plain_root(root: Path) -> Path:
    lexical = root.absolute()
    if not _lexists(lexical) or not lexical.is_dir() or _has_reparse(lexical):
        raise CompatibilityVerifierError("canonical workspace must be a plain directory")
    return lexical.resolve(strict=True)


def _relative_parts(relative: str) -> tuple[str, ...]:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise CompatibilityVerifierError("relative path is not canonical POSIX text")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise CompatibilityVerifierError("relative path is not a contained canonical path")
    return pure.parts


def contained_path(
    root: Path,
    relative: str,
    *,
    must_exist: bool = True,
    kind: str | None = None,
) -> Path:
    """Resolve one pinned path while rejecting links, reparse points, and escape."""

    workspace = _plain_root(root)
    candidate = workspace.joinpath(*_relative_parts(relative))
    probe = workspace
    for part in _relative_parts(relative):
        probe /= part
        if _lexists(probe) and _has_reparse(probe):
            raise CompatibilityVerifierError(f"link/reparse path is forbidden: {relative}")
    if must_exist and not _lexists(candidate):
        raise FileNotFoundError(candidate)
    resolved = candidate.resolve(strict=must_exist)
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise CompatibilityVerifierError(f"path escapes workspace: {relative}") from exc
    if must_exist:
        if kind == "file" and not resolved.is_file():
            raise CompatibilityVerifierError(f"regular file required: {relative}")
        if kind == "directory" and not resolved.is_dir():
            raise CompatibilityVerifierError(f"directory required: {relative}")
        if kind is None and not (resolved.is_file() or resolved.is_dir()):
            raise CompatibilityVerifierError(f"special filesystem entry rejected: {relative}")
    return resolved


def _pin(path: Path, root: Path) -> dict[str, Any]:
    workspace = _plain_root(root)
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(workspace).as_posix()
    except ValueError as exc:
        raise CompatibilityVerifierError("pinned file escapes workspace") from exc
    if _has_reparse(path) or not resolved.is_file():
        raise CompatibilityVerifierError(f"pinned path is not a plain file: {relative}")
    return {
        "path": relative,
        "bytes": resolved.stat().st_size,
        "sha256": r2.sha256_file(resolved),
    }


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise CompatibilityVerifierError(f"{label} field set changed")


def load_config(
    root: Path,
    requested_path: Path | None = None,
    *,
    supplied_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    workspace = _plain_root(root)
    canonical = contained_path(workspace, CONFIG_RELATIVE, kind="file")
    requested = requested_path or canonical
    if requested.absolute() != canonical.absolute() or requested.resolve(strict=True) != canonical:
        raise CompatibilityVerifierError("alternate compatibility config path is forbidden")
    raw = canonical.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CONFIG_SHA256:
        raise CompatibilityVerifierError("canonical compatibility config changed")
    config = json.loads(raw)
    if not isinstance(config, dict):
        raise CompatibilityVerifierError("compatibility config must be a JSON object")
    _exact_keys(
        config,
        {
            "schema_version",
            "created_at_kst",
            "status",
            "problem",
            "identity",
            "verifier_only",
            "check_only_default",
            "r2_mutation_allowed",
            "r2_rerun_or_resume_allowed",
            "execution_authorization_or_attempt_lock_allowed",
            "fit_prediction_source_truth_decode_or_experiment_scoring_allowed",
            "compatibility_receipt_write_allowed",
            "official_promotion_allowed",
            "candidate_or_test_prediction_allowed",
            "registry_append_allowed",
            "upload_allowed",
            "implementation_roles",
            "canonical_paths",
            "r2_implementation_pins",
            "r2_control_pins",
            "r2_control_inventory",
            "r2_output_contract",
            "r2_output_pins",
            "r2_output_inventory",
            "compatibility_contract",
            "expected_result",
            "v9_anchor",
            "static_counters",
        },
        label="compatibility config",
    )
    if (
        config.get("schema_version")
        != "p3_gen6_incumbent_preserving_residual_calibrator.r2_compatibility_verifier.v1"
        or config.get("identity") != IDENTITY
        or config.get("problem") != "P3"
        or config.get("implementation_roles") != IMPLEMENTATION_ROLES
    ):
        raise CompatibilityVerifierError("compatibility identity changed")
    if config.get("verifier_only") is not True or config.get("check_only_default") is not True:
        raise CompatibilityVerifierError("read-only verifier flags changed")
    false_flags = (
        "r2_mutation_allowed",
        "r2_rerun_or_resume_allowed",
        "execution_authorization_or_attempt_lock_allowed",
        "fit_prediction_source_truth_decode_or_experiment_scoring_allowed",
        "compatibility_receipt_write_allowed",
        "official_promotion_allowed",
        "candidate_or_test_prediction_allowed",
        "registry_append_allowed",
        "upload_allowed",
    )
    if any(config.get(name) is not False for name in false_flags):
        raise CompatibilityVerifierError("read-only firewall changed")
    if any(value != 0 for value in config.get("static_counters", {}).values()):
        raise CompatibilityVerifierError("static counters are nonzero")
    if supplied_config is not None and dict(supplied_config) != config:
        raise CompatibilityVerifierError("supplied compatibility config differs")
    return config


def implementation_pins(root: Path) -> dict[str, dict[str, Any]]:
    workspace = _plain_root(root)
    return {
        role: _pin(contained_path(workspace, relative, kind="file"), workspace)
        for role, relative in IMPLEMENTATION_ROLES.items()
    }


def verify_pin_map(
    root: Path,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    workspace = _plain_root(root)
    observed: dict[str, dict[str, Any]] = {}
    for role, pin in expected.items():
        if not isinstance(pin, Mapping) or set(pin) != {"path", "bytes", "sha256"}:
            raise CompatibilityVerifierError(f"{label} pin schema changed: {role}")
        if not LOWER_SHA_RE.fullmatch(str(pin["sha256"])):
            raise CompatibilityVerifierError(f"{label} pin digest malformed: {role}")
        current = _pin(contained_path(workspace, str(pin["path"]), kind="file"), workspace)
        if current != dict(pin):
            raise CompatibilityVerifierError(f"{label} pin drift: {role}")
        observed[str(role)] = current
    return observed


def _inventory_entries(path: Path) -> list[dict[str, Any]]:
    if not _lexists(path) or not path.is_dir() or _has_reparse(path):
        raise CompatibilityVerifierError("frozen inventory root is not a plain directory")
    root = path.resolve(strict=True)
    entries: list[dict[str, Any]] = []
    for item in sorted(path.rglob("*"), key=lambda value: value.relative_to(path).as_posix()):
        if _has_reparse(item):
            raise CompatibilityVerifierError(
                f"frozen inventory contains link/reparse: {item.relative_to(path).as_posix()}"
            )
        try:
            item.resolve(strict=True).relative_to(root)
        except ValueError as exc:
            raise CompatibilityVerifierError("frozen inventory escapes its root") from exc
        relative = item.relative_to(path).as_posix()
        if item.is_dir():
            entries.append({"path": relative, "type": "directory"})
        elif item.is_file():
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "bytes": item.stat().st_size,
                    "sha256": r2.sha256_file(item),
                }
            )
        else:
            raise CompatibilityVerifierError("frozen inventory contains a special entry")
    return entries


def frozen_inventory(path: Path) -> dict[str, Any]:
    entries = _inventory_entries(path)
    payload = r2.canonical_json_bytes(entries) + b"\n"
    files = [entry for entry in entries if entry["type"] == "file"]
    directories = [entry for entry in entries if entry["type"] == "directory"]
    return {
        "directories": len(directories),
        "files": len(files),
        "file_bytes": sum(int(entry["bytes"]) for entry in files),
        "algorithm": INVENTORY_ALGORITHM,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _strict_json(path: Path, *, label: str, canonical: bool = False) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise CompatibilityVerifierError(f"{label} must be a JSON object")
    if canonical and raw != r2.canonical_json_bytes(value) + b"\n":
        raise CompatibilityVerifierError(f"{label} is not canonical LF-terminated JSON")
    return value


def _require_timestamp(value: object, *, label: str) -> None:
    if not isinstance(value, str):
        raise CompatibilityVerifierError(f"{label} timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise CompatibilityVerifierError(f"{label} timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CompatibilityVerifierError(f"{label} timestamp lacks timezone")


def _validate_failure_payload(
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    _exact_keys(
        payload,
        {
            "schema_version",
            "experiment_id",
            "created_at_kst",
            "exception_type",
            "message_sha256",
            "capability_revoked",
            "candidate_created",
            "test_prediction_created",
            "registry_appended",
            "uploads",
        },
        label="historical failure receipt",
    )
    historical = contract["historical_failure"]
    _require_timestamp(payload.get("created_at_kst"), label="historical failure receipt")
    if (
        payload.get("schema_version")
        != "p3_gen6_incumbent_preserving_residual_calibrator.run_failure_receipt.r2.v1"
        or payload.get("experiment_id") != r2.EXPERIMENT_ID
        or payload.get("exception_type") != historical["exception_type"]
        or payload.get("message_sha256") != historical["message_sha256"]
        or hashlib.sha256(historical["message"].encode("utf-8")).hexdigest()
        != historical["message_sha256"]
        or payload.get("capability_revoked") is not historical["failure_receipt_capability_revoked"]
        or payload.get("candidate_created") is not False
        or payload.get("test_prediction_created") is not False
        or payload.get("registry_appended") is not False
        or payload.get("uploads") != 0
    ):
        raise CompatibilityVerifierError("historical failure receipt semantics changed")
    return dict(payload)


def verify_historical_failure_receipt(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    pin = config["r2_control_pins"]["RUN_FAILURE_RECEIPT"]
    path = contained_path(root, pin["path"], kind="file")
    payload = _strict_json(path, label="historical failure receipt", canonical=True)
    return _validate_failure_payload(payload, config["compatibility_contract"])


def _prefix_from_ast(path: Path, symbol: str) -> tuple[float, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    values: list[tuple[float, ...]] = []
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and target.id == symbol and value is not None:
            literal = ast.literal_eval(value)
            if not isinstance(literal, tuple):
                raise CompatibilityVerifierError("science prefix symbol is not a tuple")
            values.append(tuple(float(item) for item in literal))
    if len(values) != 1:
        raise CompatibilityVerifierError("science prefix symbol is missing or duplicated")
    return values[0]


def _require_prefix_consensus(
    sources: Mapping[str, Sequence[float]], expected: Sequence[float]
) -> tuple[float, ...]:
    normalized = {name: tuple(float(value) for value in values) for name, values in sources.items()}
    wanted = tuple(float(value) for value in expected)
    if not normalized or any(value != wanted for value in normalized.values()):
        raise CompatibilityVerifierError(f"science prefix sources disagree: {sorted(normalized)}")
    if len(wanted) != 5 or tuple(sorted(wanted)) != wanted or len(set(wanted)) != 5:
        raise CompatibilityVerifierError("science prefix consensus is not five ordered points")
    return wanted


def derive_science_prefixes(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    workspace = _plain_root(root)
    contract = config["compatibility_contract"]
    sources_config = contract["prefix_derivation_sources"]
    ast_pin = sources_config["meaningful_learning_curve_ast"]
    science_pin = sources_config["science_config"]
    verify_pin_map(
        workspace,
        {
            "MEANINGFUL_LEARNING_CURVE": {key: ast_pin[key] for key in ("path", "bytes", "sha256")},
            "SCIENCE_CONFIG": {key: science_pin[key] for key in ("path", "bytes", "sha256")},
        },
        label="prefix derivation",
    )
    source_tuple = _prefix_from_ast(
        contained_path(workspace, ast_pin["path"], kind="file"), ast_pin["symbol"]
    )
    science = _strict_json(
        contained_path(workspace, science_pin["path"], kind="file"),
        label="v1 science config",
    )
    if science_pin["json_pointer"] != "/sealed_surface_contract/prefix_fractions":
        raise CompatibilityVerifierError("science config prefix pointer changed")
    science_tuple = tuple(science["sealed_surface_contract"]["prefix_fractions"])
    output_pins = config["r2_output_pins"]
    commitment_tuples: dict[str, tuple[float, ...]] = {}
    for role in sources_config["fold_commitments"]:
        pin = output_pins[role]
        commitment = _strict_json(
            contained_path(workspace, pin["path"], kind="file"),
            label=f"{role} commitment",
            canonical=True,
        )
        commitment_tuples[role] = tuple(commitment["prefix_fractions"])
    metrics_pin = output_pins[sources_config["metrics_pin"]]
    metrics = _strict_json(
        contained_path(workspace, metrics_pin["path"], kind="file"),
        label="r2 metrics",
        canonical=True,
    )
    metrics_tuple = tuple(float(value) for value in metrics["points"])
    sources: dict[str, Sequence[float]] = {
        "meaningful_learning_curve_ast": source_tuple,
        "science_config": science_tuple,
        "metrics_points": metrics_tuple,
        **commitment_tuples,
    }
    corrected = _require_prefix_consensus(sources, contract["corrected_science_prefix_fractions"])
    frozen_erroneous = tuple(contract["frozen_r2_erroneous_prefix_fractions"])
    if tuple(r2.PREFIX_FRACTIONS) != frozen_erroneous or corrected == frozen_erroneous:
        raise CompatibilityVerifierError("frozen r2 verifier defect identity changed")
    return {
        "corrected_prefix_fractions": list(corrected),
        "frozen_r2_erroneous_prefix_fractions": list(frozen_erroneous),
        "sources": {name: list(values) for name, values in sources.items()},
        "all_sources_exact": True,
    }


def _verify_tree(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    output = contained_path(root, config["canonical_paths"]["r2_output"], kind="directory")
    control = contained_path(root, config["canonical_paths"]["r2_control"], kind="directory")
    output_entries = _inventory_entries(output)
    directories = {"."} | {
        entry["path"] for entry in output_entries if entry["type"] == "directory"
    }
    files = {entry["path"] for entry in output_entries if entry["type"] == "file"}
    expected = config["r2_output_contract"]
    if directories != set(expected["allowed_directories"]):
        raise CompatibilityVerifierError("r2 output directory allowlist changed")
    if files != set(expected["allowed_files"]):
        raise CompatibilityVerifierError("r2 output file allowlist changed")
    output_inventory = frozen_inventory(output)
    control_inventory = frozen_inventory(control)
    if output_inventory != config["r2_output_inventory"]:
        raise CompatibilityVerifierError("r2 whole-output inventory changed")
    if control_inventory != config["r2_control_inventory"]:
        raise CompatibilityVerifierError("r2 whole-control inventory changed")
    return output_inventory, control_inventory


def _verify_v9(root: Path, anchor: Mapping[str, Any]) -> dict[str, Any]:
    path = contained_path(root, str(anchor["path"]), kind="file")
    raw = path.read_bytes()
    if (
        len(raw) != int(anchor["bytes"])
        or hashlib.sha256(raw).hexdigest() != anchor["sha256"]
        or b"\r" in raw
        or not raw.endswith(b"\n")
    ):
        raise CompatibilityVerifierError("v9 byte identity changed")
    records = [json.loads(line) for line in raw.splitlines()]
    sequences = [record.get("seq") for record in records]
    if len(records) != anchor["record_count"] or sequences != anchor["sequences"]:
        raise CompatibilityVerifierError("v9 sequence changed")
    previous: str | None = None
    for record in records:
        claimed = record.get("event_sha256")
        unsigned = {key: value for key, value in record.items() if key != "event_sha256"}
        if r2.deep_sha256(unsigned) != claimed:
            raise CompatibilityVerifierError("v9 event digest changed")
        if previous is not None and record.get("previous_event_sha256") != previous:
            raise CompatibilityVerifierError("v9 chain changed")
        previous = str(claimed)
    if (
        records[-1].get("seq") != anchor["head_sequence"]
        or records[-1].get("event_sha256") != anchor["head_event_sha256"]
        or '"upload_performed":true' in raw.decode("utf-8")
        or anchor["uploads"] != 0
    ):
        raise CompatibilityVerifierError("v9 head or upload state changed")
    return {
        "path": anchor["path"],
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "sequences": sequences,
        "head_sequence": records[-1]["seq"],
        "head_event_sha256": records[-1]["event_sha256"],
        "uploads": 0,
    }


def _environment(root: Path, data_dir: Path | None) -> tuple[Path, Path]:
    workspace = _plain_root(root)
    workspace_value = os.environ.get("P3_WORKSPACE_ROOT")
    data_value = os.environ.get("P3_DATA_DIR")
    if not workspace_value or not data_value:
        raise CompatibilityVerifierError("P3_WORKSPACE_ROOT and P3_DATA_DIR are required")
    workspace_env = Path(workspace_value)
    data_env = Path(data_value)
    if (
        workspace_env.absolute() != workspace.absolute()
        or workspace_env.resolve(strict=True) != workspace
        or _has_reparse(workspace_env)
    ):
        raise CompatibilityVerifierError("P3_WORKSPACE_ROOT is not the canonical root")
    requested_data = data_dir or data_env
    if (
        requested_data.absolute() != data_env.absolute()
        or requested_data.resolve(strict=True) != data_env.resolve(strict=True)
        or not data_env.is_dir()
        or _has_reparse(data_env)
    ):
        raise CompatibilityVerifierError("P3_DATA_DIR differs from the canonical data root")
    for key, value in r2.THREAD_ENVIRONMENT.items():
        if os.environ.get(key) != value:
            raise CompatibilityVerifierError(f"canonical thread environment differs: {key}")
    return workspace, data_env.resolve(strict=True)


def _control_inventory_adapter(
    *,
    root: Path,
    config: Mapping[str, Any],
    r2_config: Mapping[str, Any],
    original: Any,
) -> Any:
    expected_full = [
        "attempt.lock",
        "authorization.json",
        "pre_execution_qa.json",
        "run_failure_receipt.json",
    ]

    def adapter(call_root: Path, call_config: Mapping[str, Any]) -> list[str]:
        if call_root.resolve(strict=True) != root.resolve(strict=True) or dict(call_config) != dict(
            r2_config
        ):
            raise CompatibilityVerifierError("r2 control adapter identity changed")
        names = original(call_root, call_config)
        if names != expected_full:
            raise CompatibilityVerifierError("r2 control inventory is not the exact failed attempt")
        verify_pin_map(root, config["r2_control_pins"], label="r2 control")
        _verify_tree(root, config)
        verify_historical_failure_receipt(root, config)
        return expected_full[:3]

    return adapter


@contextmanager
def _r2_compatibility_scope(
    root: Path,
    config: Mapping[str, Any],
    r2_config: Mapping[str, Any],
    corrected_prefixes: Sequence[float],
) -> Iterator[None]:
    """Temporarily adapt only the two documented historical verifier assumptions."""

    with _PATCH_LOCK:
        original_prefixes = r2.PREFIX_FRACTIONS
        original_control = r2._control_inventory
        expected_erroneous = tuple(
            config["compatibility_contract"]["frozen_r2_erroneous_prefix_fractions"]
        )
        if tuple(original_prefixes) != expected_erroneous:
            raise CompatibilityVerifierError("r2 prefix defect is no longer byte-frozen")
        adapter = _control_inventory_adapter(
            root=root,
            config=config,
            r2_config=r2_config,
            original=original_control,
        )
        r2.PREFIX_FRACTIONS = tuple(corrected_prefixes)
        r2._control_inventory = adapter
        try:
            yield
        finally:
            r2._control_inventory = original_control
            r2.PREFIX_FRACTIONS = original_prefixes
        if (
            r2._control_inventory is not original_control
            or r2.PREFIX_FRACTIONS is not original_prefixes
        ):
            raise CompatibilityVerifierError("r2 compatibility scope did not restore globals")


def _column(table: Any, name: str) -> Any:
    column = table.column(name).combine_chunks()
    if name in {"fold", "station"}:
        return np.asarray(column.to_pylist(), dtype=object)
    return column.to_numpy(zero_copy_only=False)


def _key_rows(columns: Mapping[str, Any]) -> list[tuple[float, str, int, str, int]]:
    return [
        (
            float(columns["prefix_fraction"][index]),
            str(columns["fold"][index]),
            int(columns["anchor_id"][index]),
            str(columns["station"][index]),
            int(columns["lead_h"][index]),
        )
        for index in range(len(columns["fold"]))
    ]


def _key_sha256(rows: Sequence[tuple[float, str, int, str, int]]) -> str:
    digest = hashlib.sha256()
    for prefix, fold, anchor_id, station, lead in rows:
        digest.update(f"{prefix:.2f}|{fold}|{anchor_id}|{station}|{lead}\n".encode("ascii"))
    return digest.hexdigest()


def _ids_sha256(values: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def _verify_oof_reconciliation(
    root: Path, config: Mapping[str, Any], prefixes: Sequence[float]
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = contained_path(root, config["canonical_paths"]["r2_output"], kind="directory")
    output_oof = contained_path(root, config["r2_output_pins"]["OOF"]["path"], kind="file")
    gen1_oof = contained_path(root, config["canonical_paths"]["sealed_gen1_oof"], kind="file")
    common = (
        "prefix_fraction",
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "current_hs",
        "persistence",
        "incumbent_prediction",
        "target_hs",
    )
    output_table = pq.read_table(output_oof)
    gen1_table = pq.read_table(gen1_oof, columns=list(common))
    if output_table.num_rows != 5430 or gen1_table.num_rows != 5430:
        raise CompatibilityVerifierError("OOF row count changed")
    output_columns = {name: _column(output_table, name) for name in output_table.column_names}
    gen1_columns = {name: _column(gen1_table, name) for name in common}
    output_keys = _key_rows(output_columns)
    gen1_keys = _key_rows(gen1_columns)
    if len(set(output_keys)) != 5430 or len(set(gen1_keys)) != 5430:
        raise CompatibilityVerifierError("OOF key duplication changed")
    output_order = np.asarray(sorted(range(5430), key=output_keys.__getitem__), dtype=np.int64)
    gen1_order = np.asarray(sorted(range(5430), key=gen1_keys.__getitem__), dtype=np.int64)
    if [output_keys[i] for i in output_order] != [gen1_keys[i] for i in gen1_order]:
        raise CompatibilityVerifierError("output OOF keys differ from sealed Gen1")
    for name in ("current_hs", "persistence", "incumbent_prediction", "target_hs"):
        left = np.ascontiguousarray(output_columns[name][output_order], dtype="<f8")
        right = np.ascontiguousarray(gen1_columns[name][gen1_order], dtype="<f8")
        if left.tobytes() != right.tobytes():
            raise CompatibilityVerifierError(f"output OOF {name} differs from sealed Gen1")
    candidate = np.asarray(output_columns["gen6_prediction"], dtype=np.float64)
    if not np.isfinite(candidate).all():
        raise CompatibilityVerifierError("output OOF candidate values are non-finite")

    validation_path = contained_path(
        root, config["canonical_paths"]["sealed_validation_keys"], kind="file"
    )
    validation_table = pq.read_table(validation_path, columns=["fold", "anchor_id", "station"])
    validation_fold = _column(validation_table, "fold")
    validation_anchor = _column(validation_table, "anchor_id")
    validation_station = _column(validation_table, "station")
    validation_cases = {
        (str(validation_fold[i]), int(validation_anchor[i]), str(validation_station[i]))
        for i in range(validation_table.num_rows)
    }
    output_cases = {
        (fold, anchor, station) for _prefix, fold, anchor, station, _lead in output_keys
    }
    if len(validation_cases) != 181 or validation_cases != output_cases:
        raise CompatibilityVerifierError("OOF cases differ from sealed validation keys")

    fold_reports: list[dict[str, Any]] = []
    for index, fold in enumerate(FOLD_ORDER):
        commitment = _strict_json(
            output / f"commitments/fold_{index:02d}_{fold}.json",
            label=f"fold {index} commitment",
            canonical=True,
        )
        rows = sorted(row for row in output_keys if row[1] == fold)
        ids = sorted(
            anchor for case_fold, anchor, _station in validation_cases if case_fold == fold
        )
        if (
            _key_sha256(rows) != commitment["key_sha256"]
            or _ids_sha256(ids) != commitment["validation_ids_sha256"]
            or len(rows) != int(commitment["row_count"])
        ):
            raise CompatibilityVerifierError(f"fold {index} commitment key digest changed")
        fold_reports.append(
            {
                "fold": fold,
                "rows": len(rows),
                "key_sha256": commitment["key_sha256"],
                "validation_ids_sha256": commitment["validation_ids_sha256"],
            }
        )
    observed_prefixes = tuple(sorted({float(value) for value in output_columns["prefix_fraction"]}))
    if observed_prefixes != tuple(prefixes):
        raise CompatibilityVerifierError("OOF prefix values differ from consensus")
    return (
        {
            "rows": 5430,
            "cases": 181,
            "unique_keys": 5430,
            "keys_exact_to_sealed_gen1": True,
            "truth_bytes_exact_to_sealed_gen1": True,
            "incumbent_surface_bytes_exact_to_sealed_gen1": True,
            "source_train_target_scalar_decodes": 0,
            "fold_key_digests": fold_reports,
        },
        output_columns,
    )


def _rmse(truth: Any, prediction: Any) -> float:
    truth_array = np.asarray(truth, dtype=float)
    prediction_array = np.asarray(prediction, dtype=float)
    return float(np.sqrt(np.mean(np.square(prediction_array - truth_array))))


def _group_delta(groups: Any, truth: Any, candidate: Any, incumbent: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in sorted(set(groups.tolist())):
        mask = groups == name
        result[str(name)] = _rmse(truth[mask], candidate[mask]) - _rmse(
            truth[mask], incumbent[mask]
        )
    return result


def _evaluate_point(columns: Mapping[str, Any], indices: Any, *, seed: int) -> dict[str, Any]:
    fold = np.asarray(columns["fold"][indices], dtype=object)
    anchor = np.asarray(columns["anchor_id"][indices], dtype=np.int64)
    station = np.asarray(columns["station"][indices], dtype=object)
    lead = np.asarray(columns["lead_h"][indices], dtype=np.int64)
    truth = np.asarray(columns["target_hs"][indices], dtype=float)
    candidate = np.asarray(columns["gen6_prediction"][indices], dtype=float)
    incumbent = np.asarray(columns["incumbent_prediction"][indices], dtype=float)
    if not np.isfinite(np.column_stack([truth, candidate, incumbent])).all():
        raise CompatibilityVerifierError("metric verification values are non-finite")
    blocks_by_case: dict[tuple[str, int], list[int]] = {}
    for position, key in enumerate(zip(fold.tolist(), anchor.tolist(), strict=True)):
        blocks_by_case.setdefault((str(key[0]), int(key[1])), []).append(position)
    blocks = [np.asarray(values, dtype=np.int64) for values in blocks_by_case.values()]
    if len(blocks) != 181 or any(
        tuple(sorted(lead[block].tolist())) != OFFICIAL_LEADS for block in blocks
    ):
        raise CompatibilityVerifierError("bootstrap blocks differ from 181 complete cases")
    rng = np.random.default_rng(seed)
    delta = np.empty(5000, dtype=float)
    for number in range(5000):
        selected = np.concatenate(
            [blocks[index] for index in rng.integers(0, len(blocks), size=len(blocks))]
        )
        delta[number] = _rmse(truth[selected], candidate[selected]) - _rmse(
            truth[selected], incumbent[selected]
        )
    fold_delta = _group_delta(fold, truth, candidate, incumbent)
    station_delta = _group_delta(station, truth, candidate, incumbent)
    lead_delta = _group_delta(lead, truth, candidate, incumbent)
    winter = fold == "winter_transition"
    if not winter.any():
        raise CompatibilityVerifierError("winter metric slice is empty")
    slice_delta = {
        "G-ORS": station_delta["G-ORS"],
        "I-ORS": station_delta["I-ORS"],
        "S-ORS": station_delta["S-ORS"],
        "winter": _rmse(truth[winter], candidate[winter]) - _rmse(truth[winter], incumbent[winter]),
        "lead_12": lead_delta["12"],
        "lead_18": lead_delta["18"],
        "lead_24": lead_delta["24"],
    }
    pooled_candidate = _rmse(truth, candidate)
    pooled_incumbent = _rmse(truth, incumbent)
    ci90 = np.quantile(delta, [0.05, 0.95])
    return {
        "incumbent_rmse_m": float(pooled_incumbent),
        "challenger_rmse_m": float(pooled_candidate),
        "delta_candidate_minus_incumbent_m": float(pooled_candidate - pooled_incumbent),
        "delta_ci90_m": [float(ci90[0]), float(ci90[1])],
        "paired_whole_case_bootstrap": {
            "cases": len(blocks),
            "replicates": 5000,
            "seed": seed,
            "median_delta_m": float(np.median(delta)),
            "probability_candidate_improves_descriptive": float(np.mean(delta < 0.0)),
        },
        "fold_deltas_candidate_minus_incumbent_m": fold_delta,
        "slice_deltas_candidate_minus_incumbent_m": slice_delta,
        "improved_fold_count": int(sum(value < 0.0 for value in fold_delta.values())),
        "worst_critical_slice_regression_m": float(max(slice_delta.values())),
    }


def _evaluate_gate(
    points: Mapping[float, Mapping[str, Any]],
    leakage: Mapping[str, bool],
    reproducibility: Mapping[str, bool],
) -> dict[str, Any]:
    full = points[1.0]
    checks = {
        "late_70_85_100_all_improve": all(
            float(points[value]["delta_candidate_minus_incumbent_m"]) < 0.0
            for value in (0.7, 0.85, 1.0)
        ),
        "full_ci90_excludes_zero": float(full["delta_ci90_m"][1]) < 0.0,
        "another_late_ci90_excludes_zero": any(
            float(points[value]["delta_ci90_m"][1]) < 0.0 for value in (0.7, 0.85)
        ),
        "full_delta_at_most_minus_0p030m": float(full["delta_candidate_minus_incumbent_m"])
        <= -0.030,
        "minimum_two_of_three_folds_improve": int(full["improved_fold_count"]) >= 2,
        "critical_slice_worst_regression_at_most_0p0075m": float(
            full["worst_critical_slice_regression_m"]
        )
        <= 0.0075,
        "all_leakage_checks_pass": bool(leakage)
        and all(value is True for value in leakage.values()),
        "all_reproducibility_checks_pass": bool(reproducibility)
        and all(value is True for value in reproducibility.values()),
    }
    return {
        "passed": bool(all(checks.values())),
        "decision": "CURVE_QUALIFIED" if all(checks.values()) else "RESEARCH_ONLY",
        "checks": checks,
    }


def _verify_metrics(
    root: Path,
    config: Mapping[str, Any],
    prefixes: Sequence[float],
    columns: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = _strict_json(
        contained_path(root, config["r2_output_pins"]["METRICS"]["path"], kind="file"),
        label="r2 metrics",
        canonical=True,
    )
    evidence = _strict_json(
        contained_path(
            root,
            config["r2_output_pins"]["LEARNING_CURVE_EVIDENCE"]["path"],
            kind="file",
        ),
        label="r2 learning-curve evidence",
        canonical=True,
    )
    prefix_values = np.asarray(columns["prefix_fraction"], dtype=float)
    points: dict[float, dict[str, Any]] = {}
    for prefix in prefixes:
        indices = np.flatnonzero(prefix_values == float(prefix))
        if len(indices) != 1086:
            raise CompatibilityVerifierError("metric prefix does not contain 181 cases")
        points[float(prefix)] = _evaluate_point(
            columns,
            indices,
            seed=20260823 + int(round(float(prefix) * 1000)),
        )
    persisted_points = {str(prefix): points[float(prefix)] for prefix in prefixes}
    if metrics.get("points") != persisted_points:
        raise CompatibilityVerifierError("independent 5000-bootstrap points differ")
    leakage = metrics["leakage_checks"]
    reproducibility = metrics["reproducibility_checks"]
    gate = _evaluate_gate(points, leakage, reproducibility)
    if metrics.get("local_gate") != gate:
        raise CompatibilityVerifierError("independent local gate differs")
    full = points[1.0]
    central = {
        "problem": "P3",
        "points": [
            {
                "fraction": float(prefix),
                "incumbent": float(points[float(prefix)]["incumbent_rmse_m"]),
                "challenger": float(points[float(prefix)]["challenger_rmse_m"]),
                "delta_ci90": [float(value) for value in points[float(prefix)]["delta_ci90_m"]],
            }
            for prefix in prefixes
        ],
        "fold_deltas_candidate_minus_incumbent": [
            float(full["fold_deltas_candidate_minus_incumbent_m"][name]) for name in FOLD_ORDER
        ],
        "slice_deltas_candidate_minus_incumbent": {
            name: float(full["slice_deltas_candidate_minus_incumbent_m"][name])
            for name in ("G-ORS", "I-ORS", "S-ORS", "winter", "lead_12", "lead_18", "lead_24")
        },
        "leakage_checks": {str(key): bool(value) for key, value in leakage.items()},
        "reproducibility_checks": {str(key): bool(value) for key, value in reproducibility.items()},
        "comparison_mode": "SEALED_GEN1_OOF_INCUMBENT_PRESERVING_RESEARCH_ONLY",
        "local_numeric_gate": gate,
        "official_promotion": {
            "allowed": False,
            "reason": "SEALED_GEN1_OOF_IS_NOT_AN_EXACT_OFFICIAL_PAIRED_AB",
        },
        "preregistration": {
            "hypothesis_count": 1,
            "science_deep_sha256": r2.EXPECTED_SCIENCE_DEEP_SHA256,
            "alpha_threshold_seed_or_weight_search_count": 0,
        },
    }
    if evidence != central:
        raise CompatibilityVerifierError("independent central evidence differs")
    receipts = metrics["inner_gate_receipts"]
    identity = [row for row in receipts if "IDENTITY" in str(row["decision"])]
    corrected = [row for row in receipts if "APPLY_BOUNDED_CORRECTION" in str(row["decision"])]
    expected = config["expected_result"]
    point_070 = points[0.7]
    if (
        metrics.get("status") != expected["status"]
        or gate.get("decision") != expected["gate_decision"]
        or gate.get("passed") is not expected["local_gate_passed"]
        or len(identity) != expected["identity_cells"]
        or len(corrected) != expected["bounded_correction_cells"]
        or corrected[0].get("outer_fold") != expected["bounded_correction_fold"]
        or float(corrected[0].get("prefix_fraction"))
        != expected["bounded_correction_prefix_fraction"]
        or point_070["delta_candidate_minus_incumbent_m"]
        != expected["prefix_0p70_delta_candidate_minus_incumbent_m"]
        or point_070["delta_ci90_m"] != expected["prefix_0p70_ci90_m"]
    ):
        raise CompatibilityVerifierError("research-only result semantics changed")
    return {
        "point_count": len(points),
        "bootstrap_replicates_per_point": 5000,
        "bootstrap_replicates_total": 25000,
        "points_deep_equal": True,
        "gate_deep_equal": True,
        "central_evidence_deep_equal": True,
        "gate": gate,
        "prefix_0p70": point_070,
        "identity_cells": len(identity),
        "bounded_correction_cells": len(corrected),
        "experiment_score_calls": 0,
    }


def _snapshot(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    compatibility_control = contained_path(
        root,
        config["canonical_paths"]["compatibility_control"],
        must_exist=False,
    )
    compatibility_state: dict[str, Any] = {"exists": compatibility_control.exists()}
    if compatibility_control.exists():
        compatibility_state["inventory"] = frozen_inventory(compatibility_control)
    output_inventory, control_inventory = _verify_tree(root, config)
    return {
        "r2_implementation": verify_pin_map(
            root, config["r2_implementation_pins"], label="r2 implementation"
        ),
        "r2_control": verify_pin_map(root, config["r2_control_pins"], label="r2 control"),
        "r2_output": verify_pin_map(root, config["r2_output_pins"], label="r2 output"),
        "r2_control_inventory": control_inventory,
        "r2_output_inventory": output_inventory,
        "v9": _verify_v9(root, config["v9_anchor"]),
        "compatibility_control": compatibility_state,
    }


def verify_static_compatibility(
    root: Path,
    data_dir: Path | None = None,
    *,
    requested_config: Path | None = None,
) -> dict[str, Any]:
    """Run the full compatibility audit without creating or modifying any file."""

    workspace, source = _environment(root, data_dir)
    config = load_config(workspace, requested_config)
    before = _snapshot(workspace, config)
    failure = verify_historical_failure_receipt(workspace, config)
    prefix_report = derive_science_prefixes(workspace, config)
    corrected = tuple(prefix_report["corrected_prefix_fractions"])
    r2_config, _raw = r2.load_canonical_config(workspace, workspace / r2.CONFIG_RELATIVE)
    with _r2_compatibility_scope(workspace, config, r2_config, corrected):
        frozen_report = r2.verify_published_output(
            workspace,
            source,
            requested_config=workspace / r2.CONFIG_RELATIVE,
        )
    if frozen_report.get("status") != "POST_PUBLISH_VERIFIED_EXACT_ALLOWLIST_AND_LINEAGE":
        raise CompatibilityVerifierError("adapted frozen r2 verifier did not pass")
    reconciliation, columns = _verify_oof_reconciliation(workspace, config, corrected)
    metric_report = _verify_metrics(workspace, config, corrected, columns)
    after = _snapshot(workspace, config)
    if after != before:
        raise CompatibilityVerifierError("frozen r2 or v9 state changed during verification")
    expected = config["expected_result"]
    if (
        frozen_report.get("candidate_created") is not expected["candidate_generated"]
        or frozen_report.get("test_prediction_created") is not expected["test_prediction_generated"]
        or frozen_report.get("registry_appended") is not expected["registry_appended"]
        or frozen_report.get("uploads") != expected["uploads"]
        or frozen_report["commitments"]["fit_count_observed_exact"]
        != expected["fit_count_observed_exact"]
    ):
        raise CompatibilityVerifierError("frozen verifier result prohibitions changed")
    compatibility = before["compatibility_control"]
    return {
        "schema_version": (
            "p3_gen6_incumbent_preserving_residual_calibrator.r2_compatibility_check.v1"
        ),
        "status": "PASS_R2_COMPATIBILITY_VERIFIER_RESEARCH_ONLY_NO_PROMOTION",
        "identity": IDENTITY,
        "config": {"path": CONFIG_RELATIVE, "sha256": CONFIG_SHA256},
        "historical_failure_receipt": {
            "verified_exact": True,
            "exception_type": failure["exception_type"],
            "message_sha256": failure["message_sha256"],
            "capability_revoked": failure["capability_revoked"],
        },
        "prefix_compatibility": prefix_report,
        "frozen_r2_verifier": frozen_report,
        "oof_reconciliation": reconciliation,
        "independent_metric_verification": metric_report,
        "r2_control_inventory": before["r2_control_inventory"],
        "r2_output_inventory": before["r2_output_inventory"],
        "v9": before["v9"],
        "compatibility_control_exists": compatibility["exists"],
        "compatibility_qa_receipt_exists": contained_path(
            workspace,
            config["canonical_paths"]["pre_execution_qa"],
            must_exist=False,
        ).exists(),
        "compatibility_receipt_exists": contained_path(
            workspace,
            config["canonical_paths"]["compatibility_receipt"],
            must_exist=False,
        ).exists(),
        "files_written": 0,
        "independent_qa_receipts_created": 0,
        "compatibility_receipts_created": 0,
        "execution_authorizations_created": 0,
        "attempt_locks_created": 0,
        "model_fit_calls": 0,
        "prediction_calls": 0,
        "source_train_target_scalar_decodes": 0,
        "experiment_score_calls": 0,
        "candidate_files": 0,
        "test_prediction_files": 0,
        "registry_appends": 0,
        "uploads": 0,
    }


__all__ = [
    "CONFIG_RELATIVE",
    "CONFIG_SHA256",
    "CompatibilityVerifierError",
    "IDENTITY",
    "IMPLEMENTATION_ROLES",
    "R2_MODULE",
    "contained_path",
    "derive_science_prefixes",
    "frozen_inventory",
    "implementation_pins",
    "load_config",
    "verify_historical_failure_receipt",
    "verify_pin_map",
    "verify_static_compatibility",
]
