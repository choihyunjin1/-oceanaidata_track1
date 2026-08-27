"""P2 v5 orchestration facade over the byte-pinned v4 resume engine.

V5 changes no scientific surface and does not replace the independently tested
v4 transaction engine.  It adds a new execution identity so the launcher can
bind an exact authorization schema and an absolute observations directory
before any actual execution.  Every job/cell/result/control receipt remains
isolated in the v5 namespace; no v1-v4 JobStore object is reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from p2_restore import authoritative_nested_surrogate_execution as v2
from p2_restore import authoritative_nested_surrogate_execution_v4 as v4
from p2_restore.authoritative_nested_surrogate_conformance import PrefixPlan

MAXIMUM_RESUME_ATTEMPTS = v4.MAXIMUM_RESUME_ATTEMPTS
MAXIMUM_TOTAL_ATTEMPTS = v4.MAXIMUM_TOTAL_ATTEMPTS
TERMINAL_STATUS = "COMPLETE_LOCAL_AUTHORITATIVE_SURROGATE_V5_NO_PROMOTION"
CONTROL_ENGINE_SCHEMA_VERSION = "v4"

TerminalExecutionClosed = v4.TerminalExecutionClosed
ResumeBudgetExhausted = v4.ResumeBudgetExhausted
DeterministicExecutionClosed = v4.DeterministicExecutionClosed
TransientExecutionError = v4.TransientExecutionError


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class ExecutionBindingV5:
    """Immutable v5 binding accepted by the duck-typed v4 control engine."""

    namespace: str
    execution_contract_sha256: str
    parent_recipe_sha256: str
    preexecution_seal_sha256: str
    semantic_preflight_sha256: str
    exact_command_sha256: str
    authorization_sha256: str
    module_sha256: str
    runner_sha256: str
    expected_terminal_status: str = TERMINAL_STATUS
    maximum_resume_attempts: int = MAXIMUM_RESUME_ATTEMPTS

    def as_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "execution_contract_sha256": self.execution_contract_sha256,
            "parent_recipe_sha256": self.parent_recipe_sha256,
            "preexecution_seal_sha256": self.preexecution_seal_sha256,
            "semantic_preflight_sha256": self.semantic_preflight_sha256,
            "exact_command_sha256": self.exact_command_sha256,
            "authorization_sha256": self.authorization_sha256,
            "module_sha256": self.module_sha256,
            "runner_sha256": self.runner_sha256,
            "job_store_contract_sha256": self.preexecution_seal_sha256,
            "expected_terminal_status": self.expected_terminal_status,
            "maximum_resume_attempts": self.maximum_resume_attempts,
            "maximum_total_attempts": 1 + self.maximum_resume_attempts,
            "execution_contract_revision": "v5",
            "control_engine_schema_version": CONTROL_ENGINE_SCHEMA_VERSION,
        }

    def validate(self) -> None:
        _require(self.namespace != "", "v5 namespace is empty")
        _require(Path(self.namespace).name == self.namespace, "v5 namespace is unsafe")
        for name, digest in self.as_dict().items():
            if name.endswith("sha256"):
                _require(
                    isinstance(digest, str)
                    and len(digest) == 64
                    and all(character in "0123456789abcdef" for character in digest),
                    f"invalid v5 binding digest: {name}",
                )
        _require(
            self.maximum_resume_attempts == MAXIMUM_RESUME_ATTEMPTS,
            "v5 resume budget changed",
        )
        _require(
            self.expected_terminal_status == TERMINAL_STATUS,
            "v5 terminal status changed",
        )
        _require("_v5" in self.namespace, "v5 actual namespace is not isolated")


@dataclass(frozen=True)
class SemanticPreflightOutcomeV5:
    semantic_sha256: str
    execution_context: Any


def semantic_preflight_actual_data_v5(
    observations: Any,
    *,
    recipe: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[tuple[PrefixPlan, ...], dict[str, Any]]:
    """Wrap the unchanged v4/v3 zero-fit semantic receipt in a v5 identity."""

    plans, inherited = v4.semantic_preflight_actual_data_v4(
        observations, recipe=recipe, config=config
    )
    inherited_sha = str(inherited["semantic_receipt_sha256"])
    receipt = dict(inherited)
    receipt.pop("semantic_receipt_sha256")
    receipt["schema_version"] = "p2_authoritative_actual_data_semantic_preflight.v5"
    receipt["scientific_preflight_v4_sha256"] = inherited_sha
    receipt["orchestration_revision_v5"] = {
        "science_surface_changed": False,
        "v4_resume_engine_byte_pinned": True,
        "authorization_schema_exactly_bound": True,
        "resolved_data_directory_exactly_bound": True,
        "foreign_v1_v2_v3_v4_job_or_cell_reuse_allowed": False,
        "actual_model_fits": 0,
        "predictions": 0,
        "scores": 0,
    }
    receipt["semantic_receipt_sha256"] = v2.canonical_sha256(receipt)
    return plans, receipt


def execute_authorized_curve_v5(
    *,
    observations: Any,
    plans: Sequence[PrefixPlan],
    parent_recipe: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
    contract_sha256: str,
) -> dict[str, Any]:
    """Execute unchanged v4 science only after the v5 runner authorizes it."""

    result = v4.execute_authorized_curve_v4(
        observations=observations,
        plans=plans,
        parent_recipe=parent_recipe,
        config=config,
        output_dir=output_dir,
        contract_sha256=contract_sha256,
    )
    _require(result.get("status") == v4.TERMINAL_STATUS, "inherited v4 status changed")
    result = dict(result)
    result["status"] = TERMINAL_STATUS
    result["execution_contract_revision"] = "v5"
    result["v4_resume_engine_byte_pinned"] = True
    result["foreign_v1_v2_v3_v4_job_or_cell_reuse"] = 0
    return result


def inspect_actual_namespace_read_only(
    actual_dir: Path, *, binding: ExecutionBindingV5
) -> dict[str, Any]:
    return v4.inspect_actual_namespace_read_only(actual_dir, binding=binding)


def run_resumable_execution_v5(
    *,
    actual_dir: Path,
    binding: ExecutionBindingV5,
    semantic_preflight: Callable[[], SemanticPreflightOutcomeV5],
    execute_curve: Callable[[Any, str], dict[str, Any]],
) -> dict[str, Any]:
    """Run one v5 contract through the frozen v4 atomic resume coordinator."""

    binding.validate()

    def inherited_preflight() -> v4.SemanticPreflightOutcomeV4:
        outcome = semantic_preflight()
        return v4.SemanticPreflightOutcomeV4(
            semantic_sha256=outcome.semantic_sha256,
            execution_context=outcome.execution_context,
        )

    return v4.run_resumable_execution_v4(
        actual_dir=actual_dir,
        binding=binding,
        semantic_preflight=inherited_preflight,
        execute_curve=execute_curve,
    )


__all__ = [
    "CONTROL_ENGINE_SCHEMA_VERSION",
    "DeterministicExecutionClosed",
    "ExecutionBindingV5",
    "MAXIMUM_RESUME_ATTEMPTS",
    "MAXIMUM_TOTAL_ATTEMPTS",
    "ResumeBudgetExhausted",
    "SemanticPreflightOutcomeV5",
    "TERMINAL_STATUS",
    "TerminalExecutionClosed",
    "TransientExecutionError",
    "execute_authorized_curve_v5",
    "inspect_actual_namespace_read_only",
    "run_resumable_execution_v5",
    "semantic_preflight_actual_data_v5",
]
