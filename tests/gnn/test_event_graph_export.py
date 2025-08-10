# Purpose: Tests for event-centric KG export and validation
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-09

import pandas as pd
from crickformers.gnn.schema_resolver import resolve_schema
from crickformers.gnn.event_graph_export import build_event_graph_sample, validate_event_graph


def test_event_graph_invariants():
    df = pd.DataFrame({
        'match_id': ['M1','M1'],
        'innings': [1,1],
        'over': [1.1, 1.2],
        'batter': ['A','B'],
        'bowler': ['X','X'],
        'venue': ['V','V'],
    })
    m = resolve_schema(df, use_llm=False)
    G = build_event_graph_sample(df, m, max_events=100)
    ok, msg = validate_event_graph(G)
    assert ok, msg


