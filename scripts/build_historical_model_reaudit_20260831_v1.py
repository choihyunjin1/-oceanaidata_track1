"""Build the exhaustive, model-reusable historical experiment re-audit.

This script reads only small, already-aggregated research ledgers and receipts.
It never reads competition test/sample/submission files, raw data, predictions,
or hidden labels.  The output separates four overlapping grains so that counts
are never added incorrectly: 48 historical families, 35 canonical groups,
20 later key cases, and 4 workflow exceptions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "historical_model_reaudit_20260831_v1"
FAMILY_LEDGER = (
    ROOT / "artifacts" / "promotion_retroaudit_20260827_v1" / "family_reclassification_ledger.json"
)
REPLAY_LEDGER = (
    ROOT
    / "reports"
    / "tolerance_recalibration_and_failure_replay_20260830_v2"
    / "failure-replay.json"
)
NEGATIVE_LEDGER = (
    ROOT / "reports" / "negative_evidence_registry_20260830_v1" / "failure-ledger.json"
)

STATUSES = {
    "CLOSED_EXACT": (
        "The exact tested recipe, split, feature set, postprocess, and gate is closed. "
        "The broader model class is not closed."
    ),
    "INVALID_TECHNICAL": (
        "No scientific conclusion is admissible because execution, schema, dependency, "
        "or observability failed."
    ),
    "DISCOVERY_ONLY": (
        "Useful for mechanism discovery or lineage comparison, but not independently "
        "confirmed or submission-ready."
    ),
    "OLD_GATE_REJECTED": (
        "A positive or inconclusive signal was rejected by an old unsupported hard gate; "
        "only a frozen confirmation may reopen it."
    ),
    "CHECKPOINT_PEAK": (
        "A best intermediate checkpoint or selected point exists; it is preserved as a "
        "candidate, not treated as final-epoch evidence."
    ),
    "INFORMATION_POSITIVE": (
        "The result yielded verified directional information or a positive official/local "
        "mechanism signal; this is not automatically deployment readiness."
    ),
    "PROXY_EXPOSED": (
        "Selection/proxy evidence changed sign or materially weakened on a later sealed or "
        "official surface."
    ),
}


CARD_SPECS: dict[str, dict[str, Any]] = {
    "p1-event-router-anchor": {
        "problem": "P1",
        "title": "P1 event router and official anchor lineage",
        "family_ids": ["P1-F00", "P1-F05", "P1-F06", "P1-F10", "P1-F12"],
        "key_cases": [],
        "mechanism": "Tabular/event-day routing, disagreement logic, and frozen row additions.",
        "reuse": [
            "Keep exact public-anchor hashes and row-set identities as regression baselines.",
            "Reuse the factorized G/I/S row-addition decomposition and additive-only safety contract.",
            "Use official probes only as directional mechanism evidence, never as rowwise truth.",
        ],
        "do_not_replay": [
            "Round A exact rescue recipe and union disagreement rule.",
            "Any unfrozen recombination of G/I/S after observing an official score.",
        ],
        "reopen": "A preregistered, hash-frozen factor or a genuinely new router mechanism on a fresh surface.",
    },
    "p1-temporal-neural": {
        "problem": "P1",
        "title": "P1 temporal neural, MS-TCN, and representation learning",
        "family_ids": ["P1-F01", "P1-F07"],
        "key_cases": [
            "environment_balanced_replay",
            "segment_precision_router_core",
            "window_phase_consistency",
            "sobol_trial18_threshold08",
            "group_dro_fixed_objective",
            "event_balanced_supcon",
            "hierarchical_event_precision_addonly",
        ],
        "mechanism": "TCN/MS-TCN sequence backbones, robust objectives, representation learning, and event decoders.",
        "reuse": [
            "Preserve per-epoch best checkpoints and their validation surface instead of final epoch only.",
            "Reuse sealed Sobol candidate manifests, dependent block evaluation, and anchor-preserving decoders.",
            "Reuse the distinction between selection, confirmation, and official surfaces.",
        ],
        "do_not_replay": [
            "The exact 32-point Sobol search space or frozen trial18/threshold 0.8 confirmation.",
            "Fixed Group-DRO and event-balanced SupCon objectives already shown harmful.",
        ],
        "reopen": "A materially new representation or objective with an untouched block surface and frozen checkpoint rule.",
    },
    "p1-imputation-boundary-peer": {
        "problem": "P1",
        "title": "P1 imputation, boundary, topology, and peer reliability",
        "family_ids": [
            "P1-F02",
            "P1-F03",
            "P1-F08",
            "P1-F09",
            "P1-F11",
            "P1-F14",
            "P1-F15",
            "P1-F16",
        ],
        "key_cases": ["block_inpaint", "dynamic_peer_reliability", "gors_depth_invariance"],
        "mechanism": "Postprocessing and event reconstruction from blocks, boundaries, topology, or peers.",
        "reuse": [
            "Reuse paired day/block deltas, exact changed-row sets, and critical-cell diagnostics.",
            "Keep block-inpaint and peer signals as frozen-confirmation hypotheses, not promoted models.",
        ],
        "do_not_replay": [
            "Exact target-masked quantile, topology bridge, seeded boundary v2, and fixed24h peer recipes.",
            "Worst-slice-only vetoes that were not calibrated to official value.",
        ],
        "reopen": "New endpoint/peer mechanism plus fresh block confirmation; old recipes require exact frozen replay only.",
    },
    "p1-external-density-transfer": {
        "problem": "P1",
        "title": "P1 external-profile and density transfer",
        "family_ids": ["P1-F04", "P1-F13"],
        "key_cases": [],
        "mechanism": "External profile transfer, point residuals, and target-covariate density correction.",
        "reuse": [
            "Reuse domain-shift diagnostics and fallback identity checks.",
            "Keep unlabeled profiles as diagnostics only unless competition-compatible targets exist.",
        ],
        "do_not_replay": ["Exact external point-residual and density-ratio recipes."],
        "reopen": "New labeled transport information or a new causal transfer hypothesis.",
    },
    "p2-oas-rank1-lowrank": {
        "problem": "P2",
        "title": "P2 OAS, rank-1, and low-rank residual models",
        "family_ids": ["P2-F00", "P2-F03", "P2-F09", "P2-F10", "P2-F11"],
        "key_cases": ["supervised_rank1", "crossfit_rank1_v2", "nested_pls"],
        "mechanism": "OAS anchor calibration, seasonal/layer rank-1 residuals, and nested low-rank capacity.",
        "reuse": [
            "Reuse the OAS champion lineage, cross-fit rank-1 factorization, and bin-level factor isolation.",
            "Reuse nested selection so capacity is chosen inside each outer split.",
            "Keep bin17 as positive official directional evidence; do not generalize to adjacent bins.",
        ],
        "do_not_replay": [
            "Universal density penalty and already tested full nested-PLS grid.",
            "Pooling bin17 and bin18 merely because they are adjacent.",
        ],
        "reopen": "A new frozen rank/bin factor with positive dependent-block evidence or a new outer surface.",
    },
    "p2-deep-gbm-stack": {
        "problem": "P2",
        "title": "P2 deep, GBM, and stack models",
        "family_ids": ["P2-F01", "P2-F02"],
        "key_cases": [],
        "mechanism": "Deep ensembles, GBM addons, and selected stacks.",
        "reuse": [
            "Reuse LOBO/nested estimates as the admissible score, not fitted-stack confidence intervals.",
            "Keep the tuned zero blend weight as evidence that the addon was unnecessary.",
        ],
        "do_not_replay": ["Exact deep finalist stack and CatBoost layerwise/top-3 HPO recipes."],
        "reopen": "A new representation with nested outer evaluation, not a wider search of the same addon space.",
    },
    "p2-physical-surrogate-transfer": {
        "problem": "P2",
        "title": "P2 physical, external, and surrogate transfer",
        "family_ids": ["P2-F04", "P2-F05", "P2-F06", "P2-F07", "P2-F08"],
        "key_cases": [],
        "mechanism": "TEOS/tide/NASA/ERA5 additives, physical projections, and forward surrogates.",
        "reuse": [
            "Reuse exact reference reconstruction, supported-row accounting, and physical no-op guards.",
            "Retain local-to-official sign reversals as calibration evidence.",
        ],
        "do_not_replay": [
            "Exact surrogate v5, matched-budget fallback A/B, and tested external addons."
        ],
        "reopen": "A genuinely new physical variable with causal timing and supported-row coverage.",
    },
    "p2-copula-probabilistic": {
        "problem": "P2",
        "title": "P2 Gaussian/state-conditioned copula models",
        "family_ids": [],
        "key_cases": [
            "gaussian_copula_v2",
            "state_conditioned_copula",
            "availability_aware_copula_v2",
        ],
        "mechanism": "Conditional residual transport using empirical/Gaussian copulas and state availability.",
        "reuse": [
            "Reuse train-only support audit, profile mapper preflight, and frozen deployment packaging.",
            "Retain the official sign reversal as strong proxy-failure evidence.",
        ],
        "do_not_replay": [
            "Gaussian copula v2 exact frozen official recipe.",
            "Availability-aware v2 exact primary recipe and incomplete v1 mapper.",
        ],
        "reopen": "A materially different conditional target with external calibration; no same-proxy retuning.",
    },
    "p2-profile-analog-prequential": {
        "problem": "P2",
        "title": "P2 profile, analog, RFF, and prequential residual models",
        "family_ids": [
            "P2-F12",
            "P2-F13",
            "P2-F14",
            "P2-F15",
            "P2-F16",
            "P2-F17",
            "P2-F18",
        ],
        "key_cases": [],
        "mechanism": "Annual/profile transport, analog retrieval, RFF state profiles, and prequential residuals.",
        "reuse": [
            "Reuse fail-fast p100 screens and layer/fold instability diagnostics.",
            "Reuse prequential timing and supported-row gates.",
        ],
        "do_not_replay": ["All seven listed exact profile/analog/RFF/prequential recipes."],
        "reopen": "A new representation or target; parameter-only variants of the same transport are closed.",
    },
    "p3-persistence-kma-calibration": {
        "problem": "P3",
        "title": "P3 persistence, shrink, and KMA calibration",
        "family_ids": ["P3-F01", "P3-F03", "P3-F05", "P3-F10", "P3-F11"],
        "key_cases": [],
        "mechanism": "Persistence baselines, lead-specific shrink, reverse official axis, and KMA blending.",
        "reuse": [
            "Reuse lead-specific official-factor decomposition and frozen alpha contracts.",
            "Keep uniform KMA alpha 0.425 as public-best lineage evidence.",
            "Reuse station-ablation direction only at displayed-score resolution.",
        ],
        "do_not_replay": [
            "Positive-shrink A/B and exact local cross-fit KMA strategy.",
            "Nearby-alpha sweeps after observing the official optimum.",
        ],
        "reopen": "A preregistered new lead/station factor or an untouched calibration surface.",
    },
    "p3-catboost-tabular": {
        "problem": "P3",
        "title": "P3 CatBoost and tabular routing",
        "family_ids": ["P3-F02"],
        "key_cases": ["catboost_repaired_confirmation"],
        "mechanism": "Corrected repeated-forward CatBoost, routing, and successive-halving challengers.",
        "reuse": [
            "Reuse synthetic compatibility smoke tests and confirmation-schema preflight.",
            "Keep the exact181 benchmark as a reproducible exposed-surface anchor.",
        ],
        "do_not_replay": [
            "Frozen challenger21 confirmation and incompatible Ordered/non-symmetric grid.",
            "Selection score as confirmation evidence.",
        ],
        "reopen": "A new feature target or untouched episode surface after full parameter compatibility validation.",
    },
    "p3-analog-spectral-gp-lead": {
        "problem": "P3",
        "title": "P3 analog, spectral, sparse-GP, and lead-continuous models",
        "family_ids": ["P3-F06", "P3-F07", "P3-F09"],
        "key_cases": ["lead_continuous", "sparse_gp_abstention"],
        "mechanism": "Episode analogs, spectral kernels, GP abstention, and smooth lead/regime corrections.",
        "reuse": [
            "Reuse globally 78h episode-disjoint splits and episode-block bootstrap.",
            "Keep lead-continuous as discovery evidence with its fresh one-case reversal attached.",
            "Reuse abstention coverage accounting independently from RMSE effect.",
        ],
        "do_not_replay": ["Exact analog chain and matched spectral RFF recipe."],
        "reopen": "Multiple fresh independent episodes with a frozen mechanism; one episode is insufficient.",
    },
    "p3-neural-sequence-revin-ssl": {
        "problem": "P3",
        "title": "P3 neural sequence, RevIN, and masked-SSL models",
        "family_ids": ["P3-F04", "P3-F08"],
        "key_cases": ["selection_matched_masked_ssl"],
        "mechanism": "RevIN patching, NLinear/DLinear/state-space/TimeXer-style models, and masked SSL.",
        "reuse": [
            "Reuse selection-matched confirmation and exact no-op detection.",
            "Keep reference-mismatch subruns as QA lessons only.",
        ],
        "do_not_replay": ["All listed exact valid variants and masked-SSL confirmation."],
        "reopen": "A materially new architecture on a G0-clean, untouched episode surface.",
    },
    "p3-era5-context-transfer": {
        "problem": "P3",
        "title": "P3 ERA5/context-transfer models",
        "family_ids": ["P3-F12"],
        "key_cases": [],
        "mechanism": "Fixed ERA5 source pretraining, source-quality gate, and local continuation.",
        "reuse": [
            "Reuse the 363-file manifest/checksum/time-continuity preflight and environment separation.",
            "Reuse the source-gate result from later valid ERA5 solution tests.",
        ],
        "do_not_replay": [
            "The consumed dependency-failed one-shot lock.",
            "The exact later ERA5 Hs-squared/source-gate solution shown worse than incumbent.",
        ],
        "reopen": "A new preregistered attempt with ML dependency preflight and a materially new transfer target.",
    },
}


FAMILY_CARD: dict[str, str] = {
    family_id: card_id for card_id, spec in CARD_SPECS.items() for family_id in spec["family_ids"]
}
KEY_CARD: dict[str, str] = {
    candidate: card_id for card_id, spec in CARD_SPECS.items() for candidate in spec["key_cases"]
}


BASE_FAMILY_STATUS = {
    "EXACT_RECIPE_CLOSED_UNCHANGED": "CLOSED_EXACT",
    "REFERENCE_OR_CHAMPION_NOT_FAILURE": "DISCOVERY_ONLY",
    "OFFICIAL_EVIDENCE_NOT_FAILURE": "INFORMATION_POSITIVE",
    "REOPEN_FROZEN_CONFIRMATION_ONLY": "OLD_GATE_REJECTED",
    "INCONCLUSIVE_RESEARCH_ONLY": "DISCOVERY_ONLY",
    "INCONCLUSIVE_EXACT_DEPLOYMENT_CLOSED": "CLOSED_EXACT",
    "NO_EFFECT_OR_INCONCLUSIVE": "DISCOVERY_ONLY",
    "EXPLORATORY_CHALLENGER_RESEARCH_ONLY": "DISCOVERY_ONLY",
    "NO_SCIENTIFIC_RESULT": "INVALID_TECHNICAL",
}

FAMILY_OVERRIDES: dict[str, tuple[str, list[str], str]] = {
    "P1-F05": (
        "PROXY_EXPOSED",
        ["CLOSED_EXACT"],
        "Local gain reversed on the official surface; the exact rescue recipe is closed.",
    ),
    "P1-F06": (
        "INFORMATION_POSITIVE",
        [],
        "Event-day balancing produced positive local and Public directional evidence.",
    ),
    "P1-F10": (
        "INFORMATION_POSITIVE",
        [],
        "Official disagreement factorial identified a positive router and harmful union.",
    ),
    "P1-F12": (
        "INFORMATION_POSITIVE",
        [],
        "Later official probes isolated a positive G contribution and zero displayed S marginal.",
    ),
    "P2-F05": (
        "PROXY_EXPOSED",
        ["CLOSED_EXACT"],
        "Forward-surrogate local benefit reversed strongly on the official surface.",
    ),
    "P2-F09": (
        "INFORMATION_POSITIVE",
        [],
        "Official layer-axis probes produced reusable directional evidence.",
    ),
    "P2-F10": (
        "DISCOVERY_ONLY",
        ["PROXY_EXPOSED"],
        "The first positive screen weakened to slight harm on exposed confirmation.",
    ),
    "P3-F03": (
        "PROXY_EXPOSED",
        ["CLOSED_EXACT"],
        "Small local shrink gains reversed on the official surface.",
    ),
    "P3-F05": (
        "INFORMATION_POSITIVE",
        ["CLOSED_EXACT"],
        "The exact global deployment is closed, while the broader KMA calibration axis later improved officially.",
    ),
    "P3-F09": (
        "DISCOVERY_ONLY",
        ["PROXY_EXPOSED"],
        "The exposed positive screen became worse on one fresh independent episode; n=1 remains inconclusive.",
    ),
    "P3-F10": (
        "INFORMATION_POSITIVE",
        [],
        "Official reverse-axis probes established a direction unavailable from local scores.",
    ),
    "P3-F12": (
        "INVALID_TECHNICAL",
        [],
        "The fixed one-shot reached model startup but failed on a missing CatBoost dependency before any fit.",
    ),
}

KEY_OVERRIDES: dict[str, tuple[str, list[str], str, list[str]]] = {
    "block_inpaint": (
        "OLD_GATE_REJECTED",
        [],
        "Positive pooled F1 was rejected by an unsupported worst-slice veto.",
        ["artifacts/p1_block_inpaint_v1/manifest.json"],
    ),
    "dynamic_peer_reliability": (
        "OLD_GATE_REJECTED",
        [],
        "Positive micro F1 with interval crossing zero was over-closed by the old gate.",
        ["artifacts/runs/20260813T205237+0900_strat_gate_fixed24h_59f6d5c6/manifest.json"],
    ),
    "gors_depth_invariance": (
        "DISCOVERY_ONLY",
        [],
        "Mixed depth evidence remains inconclusive and is not a deployment candidate.",
        ["reports/P1_GORS_DEPTH_INVARIANCE_2026-08-13.md"],
    ),
    "environment_balanced_replay": (
        "OLD_GATE_REJECTED",
        [],
        "Tiny positive signal is preserved as low-priority frozen-confirmation evidence.",
        ["reports/p1_value_preflight_robust_screen_20260829_v1/report-source.md"],
    ),
    "segment_precision_router_core": (
        "OLD_GATE_REJECTED",
        [],
        "Core router gain was hidden by a broader sparse-veto package.",
        ["reports/p1_mstcn_segment_precision_router_retroaudit_20260829_v1/evidence.json"],
    ),
    "window_phase_consistency": (
        "OLD_GATE_REJECTED",
        [],
        "A small positive Q2 signal was rejected by the former fixed promotion floor.",
        ["reports/p1_window_phase_consistency_20260829_v1/aggregate.json"],
    ),
    "sobol_trial18_threshold08": (
        "CLOSED_EXACT",
        ["CHECKPOINT_PEAK", "PROXY_EXPOSED"],
        "The selected Q2 peak reversed to -0.011889120 F1 on sealed Q3/Q4 confirmation.",
        [
            "artifacts/p1_mstcn_sobol_hpo_20260829_v1/aggregate.json",
            "reports/p1_mstcn_sobol_trial18_frozen_confirmation_sealed_eval_20260830_v2/independent-qa.json",
        ],
    ),
    "group_dro_fixed_objective": (
        "CLOSED_EXACT",
        [],
        "The frozen robust objective was materially harmful.",
        ["artifacts/p1_mstcn_group_dro_20260829_v2/aggregate.json"],
    ),
    "event_balanced_supcon": (
        "CLOSED_EXACT",
        [],
        "The exact event-balanced SupCon objective was materially harmful.",
        ["artifacts/p1_event_balanced_supcon_f1_head_20260830_v1/aggregate.json"],
    ),
    "hierarchical_event_precision_addonly": (
        "DISCOVERY_ONLY",
        [],
        "The add-only candidate remains inconclusive with an interval spanning zero.",
        ["reports/p1_addonly_hierarchical_event_precision_lcb_20260830_v1/result.json"],
    ),
    "supervised_rank1": (
        "INFORMATION_POSITIVE",
        [],
        "Positive dependent evidence supports the rank-1 mechanism, not automatic deployment.",
        ["artifacts/p2_alpha50_supervised_rank1_functional_residual_20260828_v1/result.json"],
    ),
    "crossfit_rank1_v2": (
        "INFORMATION_POSITIVE",
        [],
        "Positive cross-fit evidence and the later bin17 official probe support the factor axis.",
        [
            "artifacts/p2_alpha50_supervised_rank1_threeway_crossfit_regime_veto_20260828_v2/result.json",
            "reports/official_information_probe_cycle_20260830_v1/p2-official-result.json",
        ],
    ),
    "nested_pls": (
        "INFORMATION_POSITIVE",
        [],
        "The nested winner was positive, while the searched grid remains closed.",
        ["artifacts/p2_nested_pls_capacity_grid_20260829_v1/result.json"],
    ),
    "gaussian_copula_v2": (
        "CLOSED_EXACT",
        ["PROXY_EXPOSED"],
        "A -0.010616 C proxy improvement reversed to +0.012050 C official harm.",
        [
            "artifacts/p2_gaussian_copula_conditional_mean_20260830_v2/result.json",
            "reports/p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v3/official-submission-receipt.json",
        ],
    ),
    "state_conditioned_copula": (
        "INFORMATION_POSITIVE",
        [],
        "Positive local challenger evidence remains research-only after same-family proxy failure.",
        ["artifacts/p2_state_conditioned_copula_20260830_v1/result.json"],
    ),
    "availability_aware_copula_v2": (
        "CLOSED_EXACT",
        [],
        "The exact primary recipe and its interval were harmful.",
        ["reports/p2_availability_aware_continuous_sparse_copula_20260830_v2/result.json"],
    ),
    "lead_continuous": (
        "DISCOVERY_ONLY",
        ["PROXY_EXPOSED"],
        "The exposed screen was positive, but one fresh episode was +0.022617090 m worse; one block is insufficient.",
        [
            "artifacts/structural_challenger_20260827_v1/p3/metrics.json",
            "reports/p3_lead_continuous_fresh_episode_confirmation_20260830_v3/result.json",
        ],
    ),
    "sparse_gp_abstention": (
        "DISCOVERY_ONLY",
        [],
        "The interval crosses zero; coverage behavior is reusable but efficacy is inconclusive.",
        ["reports/p3_selection_matched_sparse_gp_abstention_20260830_v1/result.json"],
    ),
    "catboost_repaired_confirmation": (
        "CLOSED_EXACT",
        ["PROXY_EXPOSED"],
        "Selection improvement reversed to +0.007974131 m harm on repaired confirmation.",
        ["artifacts/p3_catboost_confirmation_contract_repair_20260830_v3/one_shot/result.json"],
    ),
    "selection_matched_masked_ssl": (
        "CLOSED_EXACT",
        [],
        "The exact masked-SSL candidate was materially harmful.",
        ["artifacts/p3_selection_matched_masked_ssl_20260830_v1/result.json"],
    ),
}


CANONICAL_SPECIAL: dict[str, tuple[str, list[str], str]] = {
    "sealed_32_point_sobol_mstcn_space": (
        "CLOSED_EXACT",
        ["CHECKPOINT_PEAK", "PROXY_EXPOSED"],
        "The space remains closed and its selected trial18 peak failed sealed confirmation.",
    ),
    "tested_supervised_rank1_and_heave_residual_recipes": (
        "INFORMATION_POSITIVE",
        ["CLOSED_EXACT"],
        "Exact tested recipes are closed, while the rank-1 factor axis has positive evidence.",
    ),
    "sealed_nested_pls_capacity_grid": (
        "INFORMATION_POSITIVE",
        ["CLOSED_EXACT"],
        "The grid is closed; its nested winner is retained as positive mechanism evidence.",
    ),
    "seasonal_empirical_kendall_copula_alpha50_residual_strict_stability_recipe": (
        "DISCOVERY_ONLY",
        ["CLOSED_EXACT", "PROXY_EXPOSED"],
        "Gaussian v2 is closed after official reversal; different state-conditioned variants remain discovery-only.",
    ),
    "oas_rank1_family_as_a_whole": (
        "INFORMATION_POSITIVE",
        [],
        "The broader OAS/rank-1 family is not a failure and contains the positive bin17 axis.",
    ),
    "local_crossfit_kma_alpha_transport_strategy": (
        "INFORMATION_POSITIVE",
        ["CLOSED_EXACT"],
        "The exact local transport strategy is closed, but the official KMA axis is positive.",
    ),
    "catboost_valid_hpo_v2_frozen_challenger_21_confirmation": (
        "CLOSED_EXACT",
        ["PROXY_EXPOSED"],
        "The selected challenger reversed on valid confirmation.",
    ),
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def canonical_status(row: dict[str, Any]) -> tuple[str, list[str], str]:
    group = row["group"]
    if group in CANONICAL_SPECIAL:
        return CANONICAL_SPECIAL[group]
    old = row["new_state"]
    if old == "EXACT_SCOPE_CLOSED_UNCHANGED":
        return "CLOSED_EXACT", [], "Exact scope remains closed; no broader-family futility claim."
    if old == "INVALID_NO_SCIENTIFIC_CONCLUSION":
        return (
            "INVALID_TECHNICAL",
            [],
            "Technical invalidity yields no scientific model conclusion.",
        )
    if old == "NOT_A_FAILURE":
        return "DISCOVERY_ONLY", [], "Reference/audit group is retained and was never a failure."
    if old.startswith("PARTIAL_REOPEN") or old.startswith("REOPEN"):
        return (
            "OLD_GATE_REJECTED",
            ["CLOSED_EXACT"],
            "Only the named component may reopen; the rest remains exactly closed.",
        )
    if old == "EXACT_LOCAL_STRATEGY_CLOSED_KMA_AXIS_NOT_CLOSED":
        return CANONICAL_SPECIAL["local_crossfit_kma_alpha_transport_strategy"]
    raise ValueError(f"Unmapped canonical state: {old}")


def render_card(card_id: str, spec: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    family_rows = [
        r for r in rows if r["grain"] == "historical_family" and r["model_card_id"] == card_id
    ]
    key_rows = [r for r in rows if r["grain"] == "key_case" and r["model_card_id"] == card_id]
    status_counts = Counter(r["primary_status"] for r in family_rows + key_rows)
    lines = [
        f"# {spec['title']}",
        "",
        f"- card id: `{card_id}`",
        f"- problem: `{spec['problem']}`",
        f"- mechanism: {spec['mechanism']}",
        f"- covered historical families: {len(family_rows)}",
        f"- covered later key cases: {len(key_rows)}",
        "- primary status counts: "
        + ", ".join(f"`{k}` {v}" for k, v in sorted(status_counts.items())),
        "",
        "## 재사용할 것",
        "",
    ]
    lines.extend(f"- {item}" for item in spec["reuse"])
    lines.extend(["", "## 그대로 반복하지 않을 것", ""])
    lines.extend(f"- {item}" for item in spec["do_not_replay"])
    lines.extend(["", "## 재개 조건", "", spec["reopen"], "", "## 근거 레코드", ""])
    lines.append("| grain | id | primary status | tags | evidence |")
    lines.append("|---|---|---|---|---|")
    for row in family_rows + key_rows:
        evidence = str(row.get("evidence_summary", row.get("adjudication", ""))).replace("|", "/")
        tags = ", ".join(row["status_tags"]) or "-"
        lines.append(
            f"| {row['grain']} | `{row['record_id']}` | `{row['primary_status']}` | "
            f"{tags} | {evidence} |"
        )
    lines.extend(
        [
            "",
            "## 해석 경계",
            "",
            "`CLOSED_EXACT`는 이 카드의 모델 계열 전체가 아니라 원장에 적힌 정확한 레시피만 닫는다. "
            "`INFORMATION_POSITIVE`와 `CHECKPOINT_PEAK`도 자동 제출 승인을 뜻하지 않는다.",
            "",
        ]
    )
    return "\n".join(lines)


def build() -> dict[str, Any]:
    family_source = load_json(FAMILY_LEDGER)
    replay = load_json(REPLAY_LEDGER)
    negative = load_json(NEGATIVE_LEDGER)
    family_by_id = {row["family_id"]: row for row in family_source["families"]}
    replay_by_id = {row["family_id"]: row for row in replay["historical_family_replay"]}
    if set(family_by_id) != set(replay_by_id):
        raise RuntimeError("The 48-family source and replay ledgers disagree on IDs")

    records: list[dict[str, Any]] = []
    for family_id in sorted(family_by_id):
        source = family_by_id[family_id]
        replay_row = replay_by_id[family_id]
        primary = BASE_FAMILY_STATUS[replay_row["new_state"]]
        tags: list[str] = []
        adjudication = "Mapped from the exhaustive 2026-08-30 replay under the new taxonomy."
        if family_id in FAMILY_OVERRIDES:
            primary, tags, adjudication = FAMILY_OVERRIDES[family_id]
        if source.get("secondary_evidence_state") == "QA_BLOCKED":
            tags = unique(tags + ["INVALID_TECHNICAL"])
        card_id = FAMILY_CARD[family_id]
        fingerprint_payload = {
            "problem": source["problem"],
            "family_id": family_id,
            "name": source["name"],
            "variants": source.get("variants", []),
            "historical_label": source.get("historical_label"),
            "historical_disposition": source.get("disposition"),
            "sources": source.get("sources", []),
        }
        records.append(
            {
                "grain": "historical_family",
                "record_id": family_id,
                "problem": source["problem"],
                "name": source["name"],
                "model_card_id": card_id,
                "primary_status": primary,
                "status_tags": tags,
                "fingerprint_sha256": canonical_sha(fingerprint_payload),
                "fingerprint": fingerprint_payload,
                "evidence_summary": replay_row.get("evidence_summary"),
                "adjudication": adjudication,
                "reopen_trigger": CARD_SPECS[card_id]["reopen"],
                "do_not_replay": CARD_SPECS[card_id]["do_not_replay"],
                "source_paths": source.get("sources", []),
            }
        )

    key_by_name = {row["candidate"]: row for row in replay["key_case_replay"]}
    if set(key_by_name) != set(KEY_OVERRIDES):
        missing = sorted(set(key_by_name) ^ set(KEY_OVERRIDES))
        raise RuntimeError(f"The 20-key-case mapping is incomplete: {missing}")
    for candidate in sorted(key_by_name):
        source = key_by_name[candidate]
        primary, tags, adjudication, latest_sources = KEY_OVERRIDES[candidate]
        card_id = KEY_CARD[candidate]
        fingerprint_payload = {
            "problem": source["problem"],
            "candidate": candidate,
            "prior_replay_state": source["new_state"],
            "benefit": source.get("benefit"),
            "benefit_ci90": source.get("benefit_ci90"),
            "latest_sources": latest_sources,
        }
        records.append(
            {
                "grain": "key_case",
                "record_id": candidate,
                "problem": source["problem"],
                "name": candidate,
                "model_card_id": card_id,
                "primary_status": primary,
                "status_tags": tags,
                "fingerprint_sha256": canonical_sha(fingerprint_payload),
                "fingerprint": fingerprint_payload,
                "evidence_summary": {
                    "benefit": source.get("benefit"),
                    "benefit_ci90": source.get("benefit_ci90"),
                    "prior_replay_state": source["new_state"],
                },
                "adjudication": adjudication,
                "reopen_trigger": CARD_SPECS[card_id]["reopen"],
                "do_not_replay": CARD_SPECS[card_id]["do_not_replay"],
                "source_paths": latest_sources,
            }
        )

    for row in replay["canonical_group_replay"]:
        primary, tags, adjudication = canonical_status(row)
        payload = {
            "problem": row["problem"],
            "group": row["group"],
            "prior_state": row["old_state"],
            "replayed_state": row["new_state"],
        }
        records.append(
            {
                "grain": "canonical_group",
                "record_id": row["group"],
                "problem": row["problem"],
                "name": row["group"],
                "model_card_id": None,
                "primary_status": primary,
                "status_tags": tags,
                "fingerprint_sha256": canonical_sha(payload),
                "fingerprint": payload,
                "evidence_summary": row["new_state"],
                "adjudication": adjudication,
                "reopen_trigger": "Follow the named partial scope only; otherwise require a materially new recipe.",
                "do_not_replay": ["Do not replay the exact group under a renamed experiment ID."],
                "source_paths": [
                    "reports/negative_evidence_registry_20260830_v1/failure-ledger.json",
                    "reports/tolerance_recalibration_and_failure_replay_20260830_v2/failure-replay.json",
                ],
            }
        )

    workflow_sources = {
        row["id"]: row.get("sources", []) for row in family_source.get("workflow_exceptions", [])
    }
    for row in replay["workflow_exception_replay"]:
        payload = {
            "problem": row["problem"],
            "id": row["id"],
            "old_state": row["old_state"],
            "new_state": row["new_state"],
        }
        records.append(
            {
                "grain": "workflow_exception",
                "record_id": row["id"],
                "problem": row["problem"],
                "name": row["id"],
                "model_card_id": None,
                "primary_status": "INVALID_TECHNICAL",
                "status_tags": [],
                "fingerprint_sha256": canonical_sha(payload),
                "fingerprint": payload,
                "evidence_summary": row["old_state"],
                "adjudication": "No scientific performance conclusion is admissible.",
                "reopen_trigger": "New preregistered attempt after the exact technical defect is preflighted.",
                "do_not_replay": ["Do not consume or overwrite the original failed attempt lock."],
                "source_paths": workflow_sources.get(row["id"], []),
            }
        )

    expected = Counter(
        {
            "historical_family": 48,
            "canonical_group": 35,
            "key_case": 20,
            "workflow_exception": 4,
        }
    )
    actual = Counter(row["grain"] for row in records)
    if actual != expected:
        raise RuntimeError(f"Unexpected audit coverage: {actual} != {expected}")

    status_counts_by_grain: dict[str, dict[str, int]] = {}
    for grain in expected:
        status_counts_by_grain[grain] = dict(
            sorted(Counter(r["primary_status"] for r in records if r["grain"] == grain).items())
        )
    status_counts_by_problem: dict[str, dict[str, int]] = {}
    for problem in ("P1", "P2", "P3"):
        status_counts_by_problem[problem] = dict(
            sorted(
                Counter(
                    r["primary_status"]
                    for r in records
                    if r["grain"] == "historical_family" and r["problem"] == problem
                ).items()
            )
        )

    source_files = [FAMILY_LEDGER, REPLAY_LEDGER, NEGATIVE_LEDGER]
    latest_source_paths = sorted(
        {
            path
            for record in records
            if record["grain"] == "key_case"
            for path in record["source_paths"]
            if (ROOT / path).exists()
        }
    )
    provenance = {
        "source_files": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": file_sha(path)}
            for path in source_files
        ],
        "later_evidence_files": [
            {"path": path, "sha256": file_sha(ROOT / path)} for path in latest_source_paths
        ],
        "negative_ledger_policy": negative.get("policy"),
        "raw_training_or_prediction_rows_read": 0,
        "official_test_sample_submission_hidden_rows_read": 0,
        "model_fits": 0,
        "csv_created": 0,
        "uploads": 0,
    }
    output = {
        "schema_version": "historical_model_reaudit.v1",
        "as_of_kst": "2026-08-31",
        "status": "COMPLETE_EXHAUSTIVE_REAUDIT",
        "status_taxonomy": STATUSES,
        "counting_rule": (
            "The four grains overlap and must never be summed as unique experiments. "
            "The exhaustive family denominator is 48."
        ),
        "coverage": dict(expected),
        "status_counts_by_grain": status_counts_by_grain,
        "historical_family_status_counts_by_problem": status_counts_by_problem,
        "model_card_count": len(CARD_SPECS),
        "records": records,
        "provenance": provenance,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model-cards").mkdir(parents=True, exist_ok=True)
    (OUT / "candidate-ledger.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    taxonomy_lines = [
        "# Historical re-audit status taxonomy",
        "",
        "한 레코드는 `primary_status` 하나와 0개 이상의 `status_tags`를 갖는다. 태그는 서로 다른 "
        "평가면의 사실을 보존한다. 예를 들어 trial18은 exact confirmation상 `CLOSED_EXACT`이면서 "
        "선택면에서는 `CHECKPOINT_PEAK`, 수송 관점에서는 `PROXY_EXPOSED`다.",
        "",
        "| status | operational meaning |",
        "|---|---|",
    ]
    taxonomy_lines.extend(f"| `{key}` | {value} |" for key, value in STATUSES.items())
    taxonomy_lines.extend(
        [
            "",
            "## Counting rule",
            "",
            "48 historical families, 35 canonical groups, 20 key cases, and 4 workflow exceptions are "
            "overlapping grains. The exhaustive historical-family denominator is 48; never report 107 as "
            "107 unique experiments.",
            "",
        ]
    )
    (OUT / "status-taxonomy.md").write_text("\n".join(taxonomy_lines), encoding="utf-8")

    for card_id, spec in CARD_SPECS.items():
        (OUT / "model-cards" / f"{card_id}.md").write_text(
            render_card(card_id, spec, records), encoding="utf-8"
        )
    index_lines = [
        "# Model-family reuse card index",
        "",
        f"총 {len(CARD_SPECS)}개 카드가 48개 historical family와 20개 later key case를 모델 계열별로 연결한다.",
        "",
        "| problem | card | historical families | key cases |",
        "|---|---|---:|---:|",
    ]
    for card_id, spec in CARD_SPECS.items():
        index_lines.append(
            f"| {spec['problem']} | [{spec['title']}](./{card_id}.md) | "
            f"{len(spec['family_ids'])} | {len(spec['key_cases'])} |"
        )
    index_lines.extend(
        ["", "각 카드는 재사용 요소, 반복 금지 범위, 재개 조건을 분리해 기록한다.", ""]
    )
    (OUT / "model-cards" / "README.md").write_text("\n".join(index_lines), encoding="utf-8")

    qa_checks = {
        "family_count_48": actual["historical_family"] == 48,
        "family_split_17_19_12": Counter(
            r["problem"] for r in records if r["grain"] == "historical_family"
        )
        == Counter({"P1": 17, "P2": 19, "P3": 12}),
        "canonical_group_count_35": actual["canonical_group"] == 35,
        "key_case_count_20": actual["key_case"] == 20,
        "workflow_exception_count_4": actual["workflow_exception"] == 4,
        "all_records_have_valid_primary_status": all(
            r["primary_status"] in STATUSES for r in records
        ),
        "all_tags_valid": all(tag in STATUSES for r in records for tag in r["status_tags"]),
        "all_family_and_key_records_have_model_cards": all(
            r["model_card_id"] in CARD_SPECS
            for r in records
            if r["grain"] in {"historical_family", "key_case"}
        ),
        "all_fingerprints_unique_within_grain": all(
            len({r["fingerprint_sha256"] for r in records if r["grain"] == grain}) == actual[grain]
            for grain in actual
        ),
        "trial18_later_evidence_overrides_reopen": any(
            r["record_id"] == "sobol_trial18_threshold08"
            and r["primary_status"] == "CLOSED_EXACT"
            and "PROXY_EXPOSED" in r["status_tags"]
            for r in records
        ),
        "gaussian_copula_official_reversal_recorded": any(
            r["record_id"] == "gaussian_copula_v2"
            and r["primary_status"] == "CLOSED_EXACT"
            and "PROXY_EXPOSED" in r["status_tags"]
            for r in records
        ),
        "lead_continuous_fresh_episode_not_overclosed": any(
            r["record_id"] == "lead_continuous" and r["primary_status"] == "DISCOVERY_ONLY"
            for r in records
        ),
        "era5_dependency_failure_is_invalid_not_harm": any(
            r["record_id"] == "P3-F12" and r["primary_status"] == "INVALID_TECHNICAL"
            for r in records
        ),
        "no_model_fit": provenance["model_fits"] == 0,
        "no_official_input_rows_read": provenance[
            "official_test_sample_submission_hidden_rows_read"
        ]
        == 0,
        "no_csv_created": provenance["csv_created"] == 0,
        "no_upload": provenance["uploads"] == 0,
    }
    if not all(qa_checks.values()):
        raise RuntimeError(f"Independent QA failed: {qa_checks}")
    qa = {
        "schema_version": "historical_model_reaudit.independent_qa.v1",
        "status": "PASS",
        "checks": qa_checks,
        "counts": dict(actual),
        "status_counts_by_grain": status_counts_by_grain,
        "candidate_ledger_sha256": file_sha(OUT / "candidate-ledger.json"),
        "model_card_count": len(CARD_SPECS),
        "policy_boundaries": {
            "broad_model_classes_closed": False,
            "exact_recipe_closure_only": True,
            "overlapping_grains_not_summed": True,
            "official_inputs_read": False,
        },
    }
    (OUT / "independent-qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    family_status_rows = []
    for problem in ("P1", "P2", "P3"):
        counts = status_counts_by_problem[problem]
        for status in STATUSES:
            family_status_rows.append(
                {"problem": problem, "status": status, "family_count": counts.get(status, 0)}
            )
    key_case_rows = []
    for order, row in enumerate(
        sorted(
            (r for r in records if r["grain"] == "key_case"),
            key=lambda item: (item["problem"], item["record_id"]),
        ),
        start=1,
    ):
        key_case_rows.append(
            {
                "order": order,
                "problem": row["problem"],
                "candidate": row["record_id"],
                "primary_status": row["primary_status"],
                "tags": ", ".join(row["status_tags"]) or "-",
                "model_card": row["model_card_id"],
                "adjudication": row["adjudication"],
            }
        )
    artifact = {
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "2026-08-31 모델별 과거 실험 전수 재감사",
            "description": "48개 historical family를 최신 봉인 확인과 공식 정보로 재판정하고 14개 모델 계열 카드로 연결한 감사",
            "generatedAt": "2026-08-31T18:00:00+09:00",
            "sources": [
                {
                    "id": "reaudit_ledger",
                    "label": "Exhaustive local historical re-audit ledger",
                    "path": "reports/historical_model_reaudit_20260831_v1/candidate-ledger.json",
                    "query": {
                        "sql": "SELECT problem, primary_status, COUNT(*) AS family_count FROM historical_family_records GROUP BY problem, primary_status ORDER BY problem, primary_status",
                        "description": "Counts are deterministically generated from the 48-family local aggregate ledger; no raw or official input rows are queried.",
                        "language": "json",
                        "executed_at": "2026-08-31T18:00:00+09:00",
                        "filters": [
                            "grain = historical_family for the chart",
                            "later evidence through 2026-08-30",
                            "no hidden, test, sample, submission, raw, or prediction rows",
                        ],
                        "metric_definitions": [
                            "family_count counts only the 48 historical-family grain.",
                            "Primary status is exclusive; status_tags preserve overlapping facts such as checkpoint peak or proxy exposure.",
                        ],
                        "tables_used": [
                            "artifacts/promotion_retroaudit_20260827_v1/family_reclassification_ledger.json",
                            "reports/tolerance_recalibration_and_failure_replay_20260830_v2/failure-replay.json",
                            "reports/negative_evidence_registry_20260830_v1/failure-ledger.json",
                        ],
                    },
                }
            ],
            "cards": [
                {
                    "id": "family_coverage",
                    "description": "Exhaustive historical-family denominator",
                    "dataset": "coverage",
                    "sourceId": "reaudit_ledger",
                    "metrics": [
                        {
                            "label": "Historical families",
                            "field": "historical_family",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "canonical_coverage",
                    "description": "Overlapping canonical-group cross-check",
                    "dataset": "coverage",
                    "sourceId": "reaudit_ledger",
                    "metrics": [
                        {
                            "label": "Canonical groups",
                            "field": "canonical_group",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "model_card_coverage",
                    "description": "Reusable model-family cards",
                    "dataset": "coverage",
                    "sourceId": "reaudit_ledger",
                    "metrics": [
                        {"label": "Model cards", "field": "model_cards", "format": "number"}
                    ],
                },
            ],
            "charts": [
                {
                    "id": "family_status_chart",
                    "title": "문제별 historical-family 주상태",
                    "subtitle": "정확 레시피 종료가 가장 많지만, 정보가치와 발견용 레코드도 별도로 보존됐다.",
                    "type": "bar",
                    "dataset": "family_status_counts",
                    "sourceId": "reaudit_ledger",
                    "encodings": {
                        "x": {"field": "problem", "type": "nominal", "label": "문제"},
                        "y": {
                            "field": "family_count",
                            "type": "quantitative",
                            "label": "family 수",
                            "format": "number",
                        },
                        "color": {"field": "status", "type": "nominal", "label": "주상태"},
                        "tooltip": [
                            {"field": "problem", "type": "nominal", "label": "문제"},
                            {"field": "status", "type": "nominal", "label": "주상태"},
                            {"field": "family_count", "type": "quantitative", "label": "family 수"},
                        ],
                    },
                    "options": {"orientation": "vertical", "grouping": "stacked"},
                    "layout": "full",
                }
            ],
            "tables": [
                {
                    "id": "key_case_table",
                    "title": "최신 근거로 재판정한 20개 key case",
                    "subtitle": "주상태와 중첩 태그를 함께 읽어야 하며, 정보가치는 제출 준비도와 다르다.",
                    "dataset": "key_cases",
                    "sourceId": "reaudit_ledger",
                    "defaultSort": {"field": "order", "direction": "asc"},
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "order", "label": "순서", "type": "number"},
                        {"field": "problem", "label": "문제", "type": "text"},
                        {"field": "candidate", "label": "후보", "type": "text"},
                        {"field": "primary_status", "label": "주상태", "type": "text"},
                        {"field": "tags", "label": "중첩 태그", "type": "text"},
                        {"field": "model_card", "label": "모델 카드", "type": "text"},
                        {"field": "adjudication", "label": "최신 판정", "type": "text"},
                    ],
                }
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# 2026-08-31 모델별 과거 실험 전수 재감사",
                    "layout": "full",
                },
                {
                    "id": "summary",
                    "type": "markdown",
                    "body": "## 결론\n\n48개 historical family를 모두 다시 판정했다. 25개는 정확 레시피만 닫히고, 11개는 발견용, 6개는 정보가치 양성, 2개는 옛 gate 때문에 탈락, 3개는 프록시 수송 실패, 1개는 기술 무효다. trial18, P2 Gaussian copula, P3 CatBoost처럼 나중 근거가 생긴 후보는 과거 재개 판정을 그대로 두지 않고 최신 근거로 덮어썼다.",
                    "layout": "full",
                },
                {
                    "id": "coverage_strip",
                    "type": "metric-strip",
                    "cardIds": ["family_coverage", "canonical_coverage", "model_card_coverage"],
                    "layout": "full",
                },
                {
                    "id": "status_heading",
                    "type": "markdown",
                    "body": "## 닫힌 것은 모델명이 아니라 정확 조합이다\n\n`CLOSED_EXACT`는 모델 계열 전체의 무가치가 아니라 레시피, split, feature, postprocess, gate 조합의 종료다. 35개 canonical group, 20개 key case, 4개 workflow exception은 48개 family와 겹치므로 합산하지 않는다.",
                    "sourceId": "reaudit_ledger",
                    "layout": "full",
                },
                {
                    "id": "status_chart",
                    "type": "chart",
                    "chartId": "family_status_chart",
                    "layout": "full",
                },
                {
                    "id": "key_heading",
                    "type": "markdown",
                    "body": "## 과거 재개 후보도 최신 결과로 다시 닫거나 낮췄다\n\nP1 trial18은 checkpoint peak를 보존하되 exact confirmation은 닫혔다. P2 Gaussian copula v2와 P3 CatBoost challenger는 선택 또는 프록시 부호가 다음 평가면에서 반전됐다. P3 lead-continuous는 신선한 episode가 1개뿐이라 harm 확정 대신 발견용으로 남겼다.",
                    "sourceId": "reaudit_ledger",
                    "layout": "full",
                },
                {"id": "key_table", "type": "table", "tableId": "key_case_table", "layout": "full"},
                {
                    "id": "reuse",
                    "type": "markdown",
                    "body": "## 재사용 단위는 14개 모델 카드다\n\n각 카드는 재사용 가능한 split, checkpoint, hash, smoke test, factor decomposition과 그대로 반복하지 않을 exact recipe, 그리고 재개 조건을 분리한다. 새 연구는 후보명을 먼저 이 카드와 대조해 exact/semantic 중복을 차단해야 한다.",
                    "layout": "full",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": "## 한계\n\n이 감사는 저장된 집계 결과와 공식 receipt만 재판정했다. 새 학습, raw/prediction 행 읽기, 공식 test/sample/submission/hidden 값 읽기, CSV 생성, 업로드는 0이다. Public 방향성은 Private 성능 보장이 아니며 서로 다른 문제의 F1, C RMSE, m RMSE 크기를 직접 비교하지 않는다.",
                    "sourceId": "reaudit_ledger",
                    "layout": "full",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": "2026-08-31T18:00:00+09:00",
            "status": "ready",
            "datasets": {
                "coverage": [
                    {
                        "historical_family": 48,
                        "canonical_group": 35,
                        "key_case": 20,
                        "workflow_exception": 4,
                        "model_cards": len(CARD_SPECS),
                    }
                ],
                "family_status_counts": family_status_rows,
                "key_cases": key_case_rows,
            },
        },
    }
    (OUT / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="Rebuild and verify deterministic outputs"
    )
    args = parser.parse_args()
    before = None
    ledger_path = OUT / "candidate-ledger.json"
    if args.check and ledger_path.exists():
        before = ledger_path.read_bytes()
    output = build()
    if args.check and before is not None and before != ledger_path.read_bytes():
        raise SystemExit("candidate-ledger.json was not deterministic")
    print(
        json.dumps(
            {
                "status": output["status"],
                "coverage": output["coverage"],
                "model_card_count": output["model_card_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
