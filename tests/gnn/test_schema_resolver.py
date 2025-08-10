# Purpose: Unit tests for schema resolver
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-09

import pandas as pd
from crickformers.gnn.schema_resolver import resolve_schema


def test_resolve_basic_aliases():
    df = pd.DataFrame({
        'batsman': ['A'],
        'bowler_name': ['B'],
        'Runs': [4],
        'wicket': [0],
        'battingteam': ['T'],
        'Bowling Team': ['U'],
        'Match ID': ['M1'],
        'Innings': [1],
        'Over': [3],
    })
    m = resolve_schema(df, use_llm=False)
    assert m['batter'] == 'batsman'
    assert m['bowler'] == 'bowler_name'
    assert m['runs_scored'] in ('Runs', 'runs_scored')
    assert m['is_wicket'] in ('wicket', 'is_wicket', 'dismissal')
    assert m['team_batting'] in ('battingteam', 'team_batting')
    assert m['team_bowling'] in ('Bowling Team', 'team_bowling')
    assert m['match_id'] in ('Match ID', 'match_id')
    assert m['innings'] in ('Innings', 'innings')
    assert m['over'] in ('Over', 'over')


