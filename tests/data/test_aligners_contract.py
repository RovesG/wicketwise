# Purpose: Contract tests for aligner adapters
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-10

from wicketwise.data.aligners.factory import get_aligner


def test_exact_aligner_missing_files_returns_error(tmp_path):
    a = get_aligner("exact")
    res = a.align(str(tmp_path / "nv.csv"), str(tmp_path / "dec.csv"))
    assert res["status"] == "error"


def test_dna_aligner_missing_files_returns_error(tmp_path):
    a = get_aligner("dna")
    res = a.align(str(tmp_path / "nv.csv"), str(tmp_path / "dec.csv"))
    assert res["status"] == "error"
