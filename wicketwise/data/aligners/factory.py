# Purpose: Factory to get aligner adapters from config
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-10

from __future__ import annotations

from typing import Any
from .adapters import ExactAligner, HybridAligner, LLMAligner, DNAAligner


def get_aligner(name: str):
    name = (name or "").lower()
    if name == "exact":
        return ExactAligner()
    if name == "hybrid":
        return HybridAligner()
    if name == "llm":
        return LLMAligner()
    if name == "dna":
        return DNAAligner()
    raise ValueError(f"Unknown aligner: {name}")
