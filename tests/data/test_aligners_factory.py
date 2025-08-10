# Purpose: Tests for aligners factory mapping
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-10

import importlib.util
from wicketwise.data.aligners.factory import get_aligner


def test_factory_returns_instances():
    names = ["exact", "hybrid", "dna"]
    if importlib.util.find_spec("openai") is not None:
        names.append("llm")
    for name in names:
        inst = get_aligner(name)
        assert hasattr(inst, "align") and callable(getattr(inst, "align"))


def test_factory_raises_on_unknown():
    import pytest

    with pytest.raises(ValueError):
        get_aligner("unknown")
