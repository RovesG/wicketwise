# Purpose: Test BettingPolicy basic behaviors
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-10

from crickformers.inference.policy import BettingPolicy


def test_fractional_kelly_bounds():
    policy = BettingPolicy(kelly_fraction=0.5, cap=0.02)
    stake = policy.fractional_kelly(prob=0.6, odds=2.2)
    assert 0.0 <= stake <= 0.02


def test_should_bet_with_uncertainty_guard():
    policy = BettingPolicy(uncertainty_threshold=0.1)
    assert policy.should_bet(0.6, 2.0, uncertainty=0.05) is True
    assert policy.should_bet(0.6, 2.0, uncertainty=0.5) is False
