# Purpose: Strategy interface for match aligners
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-09

from __future__ import annotations

from typing import Protocol, Dict, Any


class BaseAligner(Protocol):
    def align(self, nvplay_path: str, decimal_path: str, **kwargs: Any) -> Any:
        ...


