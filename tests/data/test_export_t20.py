# Purpose: Tests for T20 exporter from BallEvent-like DataFrame
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-10

import pandas as pd
from wicketwise.data.alljson.export_t20 import export_t20_events, TRAIN_COLUMNS


def test_export_t20_filters_and_columns(tmp_path):
    df = pd.DataFrame(
        [
            {"match_type": "Test", "runs_total": 1},
            {"match_type": "T20", "runs_total": 2, "team_batting": "A", "date_utc": "2020-01-01"},
        ]
    )
    out = export_t20_events(df, str(tmp_path))
    out_df = pd.read_parquet(out)
    assert len(out_df) == 1
    assert out_df.iloc[0]["runs_total"] == 2
