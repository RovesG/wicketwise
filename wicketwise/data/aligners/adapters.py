# Purpose: Adapter classes implementing unified aligner interface
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-10

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class _BaseAdapter:
    def _validate_inputs(self, nvplay_path: str, decimal_path: str) -> Dict[str, Any]:
        nv = Path(nvplay_path)
        dec = Path(decimal_path)
        if not nv.exists():
            return {"status": "error", "message": f"NVPlay file not found: {nv}"}
        if not dec.exists():
            return {"status": "error", "message": f"Decimal file not found: {dec}"}
        return {"status": "ok"}


class ExactAligner(_BaseAdapter):
    def align(self, nvplay_path: str, decimal_path: str, **kwargs: Any) -> Dict[str, Any]:
        check = self._validate_inputs(nvplay_path, decimal_path)
        if check.get("status") != "ok":
            return check
        # Defer heavy imports until execution
        from match_aligner import MatchAligner

        aligner = MatchAligner(nvplay_path, decimal_path)
        matches = aligner.find_matches(similarity_threshold=float(kwargs.get("threshold", 0.9)))
        return {"status": "success", "num_matches": len(matches)}


class HybridAligner(_BaseAdapter):
    def align(self, nvplay_path: str, decimal_path: str, **kwargs: Any) -> Dict[str, Any]:
        check = self._validate_inputs(nvplay_path, decimal_path)
        if check.get("status") != "ok":
            return check
        from hybrid_match_aligner import hybrid_align_matches

        output_path = kwargs.get("output_path", "workflow_output/aligned_matches.csv")
        res = hybrid_align_matches(
            nvplay_path=nvplay_path,
            decimal_path=decimal_path,
            openai_api_key=kwargs.get("openai_api_key"),
            output_path=output_path,
        )
        # res may be DataFrame or list; normalize
        try:
            num = len(res)
        except Exception:
            num = 0
        return {"status": "success", "num_matches": num, "aligned_path": output_path}


class LLMAligner(_BaseAdapter):
    def align(self, nvplay_path: str, decimal_path: str, **kwargs: Any) -> Dict[str, Any]:
        check = self._validate_inputs(nvplay_path, decimal_path)
        if check.get("status") != "ok":
            return check
        # Import within scope to avoid hard dependency at import time
        from llm_match_aligner import LLMMatchAligner

        openai_api_key = kwargs.get("openai_api_key")
        aligner = LLMMatchAligner(openai_api_key=openai_api_key)
        df = aligner.align(nvplay_path, decimal_path, output_path=kwargs.get("output_path"))
        try:
            num = len(df)
        except Exception:
            num = 0
        return {"status": "success", "num_matches": num, "aligned_path": kwargs.get("output_path")}


class DNAAligner(_BaseAdapter):
    def align(self, nvplay_path: str, decimal_path: str, **kwargs: Any) -> Dict[str, Any]:
        check = self._validate_inputs(nvplay_path, decimal_path)
        if check.get("status") != "ok":
            return check
        from cricket_dna_match_aligner import align_matches_with_cricket_dna

        output_path = kwargs.get("output_path", "workflow_output/dna_aligned_matches.csv")
        df = align_matches_with_cricket_dna(
            decimal_csv_path=decimal_path,
            nvplay_csv_path=nvplay_path,
            output_path=output_path,
            similarity_threshold=float(kwargs.get("threshold", 0.85)),
        )
        try:
            num = len(df)
        except Exception:
            num = 0
        return {"status": "success", "num_matches": num, "aligned_path": output_path}
