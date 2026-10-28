# filepath: backend/app/services/problem_bank.py
"""
PlaceMate — Problem Bank
==========================

Loads and validates ``data/problems/*.json`` against ``ProblemSpec``.
Exposes ``get(problem_id)`` for the harness builder and execution service.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict

from app.schemas.problem import ProblemSpec

logger = logging.getLogger(__name__)

_PROBLEMS_DIR = Path(__file__).resolve().parents[2] / "data" / "problems"


@lru_cache(maxsize=1)
def _load_all() -> Dict[str, ProblemSpec]:
    """Load every JSON file in data/problems/, validate, reject dupes."""
    bank: Dict[str, ProblemSpec] = {}
    if not _PROBLEMS_DIR.is_dir():
        logger.warning("event=problem_bank_missing dir=%s", _PROBLEMS_DIR)
        return bank

    for path in sorted(_PROBLEMS_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
        spec = ProblemSpec.model_validate(raw)
        if spec.id in bank:
            raise ValueError(
                f"Duplicate problem id '{spec.id}' in {path.name} "
                f"and {bank[spec.id].title}"
            )
        bank[spec.id] = spec
        logger.debug("event=problem_loaded problem_id=%s file=%s", spec.id, path.name)

    logger.info("event=problem_bank_ready count=%d", len(bank))
    return bank


def get(problem_id: str) -> ProblemSpec | None:
    """Return the problem spec for *problem_id*, or None."""
    return _load_all().get(problem_id)


def all_ids() -> list[str]:
    """Return sorted list of all loaded problem IDs."""
    return sorted(_load_all().keys())
