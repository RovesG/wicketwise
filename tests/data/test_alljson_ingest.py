# Purpose: Tests for all_json ingestion to BallEventV1
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-10

import json
from wicketwise.data.alljson.ingest import iter_events_from_json


def test_iter_events_basic_mapping(tmp_path):
    data = {
        "info": {
            "match_type": "Test",
            "event": {"name": "Tour"},
            "season": "2001/02",
            "dates": ["2001-12-19"],
            "city": "Bengaluru",
            "venue": "Chinnaswamy",
            "teams": ["England", "India"],
            "registry": {"people": {"A": "idA", "B": "idB"}},
        },
        "innings": [
            {
                "team": "England",
                "overs": [
                    {
                        "over": 0,
                        "deliveries": [
                            {"batter": "A", "bowler": "B", "non_striker": "A", "runs": {"batter": 1, "extras": 0, "total": 1}},
                            {"batter": "A", "bowler": "B", "runs": {"batter": 0, "extras": 1, "total": 1}, "extras": {"wides": 1}},
                        ],
                    }
                ],
            }
        ],
    }
    evts = list(iter_events_from_json(data, "00001.json"))
    assert len(evts) == 2
    e0, e1 = evts
    assert e0.batter_id == "idA" and e0.bowler_id == "idB"
    assert e0.runs_total == 1 and e0.legal_ball is True
    assert e1.runs_extras == 1 and e1.legal_ball is False  # wide
