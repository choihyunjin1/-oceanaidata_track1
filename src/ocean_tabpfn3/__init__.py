"""Offline-only TabPFN-3 competition integration."""

from .offline import (
    CLASSIFIER_FILENAME,
    REGRESSOR_FILENAME,
    TabPFN3ContractError,
    inspect_preflight,
    make_classifier,
    make_regressor,
    require_ready,
)

__all__ = [
    "CLASSIFIER_FILENAME",
    "REGRESSOR_FILENAME",
    "TabPFN3ContractError",
    "inspect_preflight",
    "make_classifier",
    "make_regressor",
    "require_ready",
]
