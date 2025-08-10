# Purpose: Unit tests for chunked KG pipeline
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-09

import os
import pandas as pd
from crickformers.gnn.kg_pipeline import build_aggregates_from_csv, PipelineSettings


def test_build_aggregates_from_csv(tmp_path):
    csv = tmp_path / "mini.csv"
    df = pd.DataFrame({
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
    df.to_csv(csv, index=False)

    settings = PipelineSettings(chunk_size=2, cache_dir=str(tmp_path / "agg_cache"))
    aggs = build_aggregates_from_csv(str(csv), settings)

    assert 'batter_bowler' in aggs and len(aggs['batter_bowler']) > 0
    assert 'match_stats' in aggs and len(aggs['match_stats']) >= 1

    # Second call should hit cache
    aggs2 = build_aggregates_from_csv(str(csv), settings)
    assert 'batter_bowler' in aggs2


