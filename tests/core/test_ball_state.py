# Purpose: Tests for BallStateV1 builder and invariants
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-09

import pandas as pd
from crickformers.gnn.schema_resolver import resolve_schema
from wicketwise.core.ball_state import build_ball_states, BallStateV1


def test_build_ball_states_basic():
    df = pd.DataFrame({
        'match_id': ['M1'] * 6,
        'innings': [1] * 6,
        'over': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        'batter': ['A','A','A','A','A','A'],
        'bowler': ['X','X','X','X','X','X'],
        'venue': ['V'] * 6,
        'runs_scored': [0,1,4,0,0,6],
        'wicket': [0,0,0,0,0,1],
    })
    m = resolve_schema(df, use_llm=False)
    states = build_ball_states(df, m)
    assert len(states) == 6
    assert isinstance(states[0], BallStateV1)
    # Monotonicity and no lookahead (score must be 0 at ball 1)
    assert states[0].score == 0
    assert states[1].score >= states[0].score


