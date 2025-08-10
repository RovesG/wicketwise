# Purpose: Unit tests for scalable graph assembly
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-09

import pandas as pd
from crickformers.gnn.schema_resolver import resolve_schema
from crickformers.gnn.kg_aggregator import aggregate_core
from crickformers.gnn.scalable_graph_builder import build_graph_from_aggregates


def _toy_df():
    return pd.DataFrame({
        'batter': ['A','A','B'],
        'bowler': ['X','Y','X'],
        'venue': ['V','V','W'],
        'runs_scored': [1,4,0],
        'wicket': [0,0,1],
        'team_batting': ['T1','T1','T2'],
        'team_bowling': ['U1','U1','U2'],
        'match_id': ['M1','M1','M1'],
        'innings': [1,1,1],
        'over': [1,2,3]
    })


def test_build_graph_basic():
    df = _toy_df()
    m = resolve_schema(df, use_llm=False)
    aggs = aggregate_core(df, m)
    G = build_graph_from_aggregates(aggs)
    # At least batter and bowler nodes
    assert any(G.nodes[n].get('type') == 'batter' for n in G.nodes)
    assert any(G.nodes[n].get('type') == 'bowler' for n in G.nodes)
    # Edges exist
    assert G.number_of_edges() > 0


