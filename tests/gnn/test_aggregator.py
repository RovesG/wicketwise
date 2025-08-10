# Purpose: Unit tests for vectorized KG aggregations
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-09

import pandas as pd
from crickformers.gnn.schema_resolver import resolve_schema
from crickformers.gnn.kg_aggregator import aggregate_core, compute_partnerships


def _toy_df():
    return pd.DataFrame({
        'batter': ['A','A','B','B','A','C'],
        'bowler': ['X','Y','X','Y','X','Y'],
        'venue': ['V','V','V','W','W','W'],
        'runs_scored': [1,4,0,6,2,0],
        'wicket': [0,0,0,1,0,0],
        'team_batting': ['T1','T1','T2','T2','T1','T3'],
        'team_bowling': ['U1','U1','U2','U2','U1','U3'],
        'match_id': ['M1','M1','M1','M1','M1','M2'],
        'innings': [1,1,1,1,2,1],
        'over': [1,2,3,4,5,1]
    })


def test_aggregate_core_basic():
    df = _toy_df()
    m = resolve_schema(df, use_llm=False)
    aggs = aggregate_core(df, m)
    assert 'batter_bowler' in aggs and len(aggs['batter_bowler']) > 0
    assert 'batter_phase' in aggs and 'bowler_phase' in aggs
    assert 'batter_stats' in aggs and 'bowler_stats' in aggs
    assert 'match_stats' in aggs and len(aggs['match_stats']) >= 1


def test_compute_partnerships_outputs():
    df = _toy_df()
    m = resolve_schema(df, use_llm=False)
    parts = compute_partnerships(df, m)
    # partnerships may be approximate; validate schema and non-negative
    assert list(parts.columns) == ['batter_a','batter_b','runs','partnerships']
    assert (parts['runs'] >= 0).all()
    assert (parts['partnerships'] >= 0).all()


