# Purpose: Overview of the scalable KG pipeline and GNN utilities
# Author: Phi1618 Cricket AI Team, Last Modified: 2025-08-10

## Modules
- `schema_resolver.py`: Robust CSV schema normalization (LLM assist optional/off by default)
- `kg_aggregator.py`: Vectorized aggregations (Pandas groupby) to compute node/edge tables, chunked
- `scalable_graph_builder.py`: Fast NetworkX assembly from aggregates (nodes/edges, light metrics)
- `kg_pipeline.py`: Orchestrates chunked read → resolve → aggregate → cache → assemble
- `event_graph_export.py`: Builds an event-centric subgraph (BallEvent nodes) for analytics/GNN

## Pipeline usage
```python
from crickformers.gnn.kg_pipeline import build_aggregates_from_csv, PipelineSettings
from crickformers.gnn.scalable_graph_builder import build_graph_from_aggregates

settings = PipelineSettings(
    chunk_size=500_000,
    cache_dir="models/aggregates",
    use_llm_schema_hint=False,      # off by default
    compute_heavy_metrics=False,    # degree-only fast metrics by default
    normalize_ids=False,
)

aggs = build_aggregates_from_csv("/absolute/path/to/joined_ball_by_ball_data.csv", settings)
G = build_graph_from_aggregates(aggs)
print(G.number_of_nodes(), G.number_of_edges())
```

## Event graph export
```python
from crickformers.gnn.event_graph_export import export_event_graph

# Build a compact BallEvent-centric subgraph (sample/window recommended)
H = export_event_graph(G, sample_limit=100_000)
```

## Settings & endpoints
- Backend exposes:
  - `GET|POST /api/kg-settings` → `chunk_size`, `cache_dir`, `use_llm_schema_hint`, `compute_heavy_metrics`, `normalize_ids`
  - `POST /api/kg-cache/purge` → clear `models/aggregates`
  - `GET|POST /api/aligner-settings` → `dna|exact|hybrid|llm` selection

## Performance & caching
- Chunked CSV read avoids memory blowups on multi-million rows (tested on 240k+, designed for 7M+)
- Aggregates are cached under `models/aggregates/` and reused when inputs/settings match
- Default metrics are lightweight (degree/strength). Heavy centralities are disabled by default

## Invariants & validation
- Aggregator ensures keyed uniqueness and consistent types
- Event graph invariants (when used):
  - Each `BallEvent` binds to exactly one match/innings/over index
  - Exactly two player-role links (batter, bowler) where available
  - Monotonic time within innings

## Tests
- `tests/gnn/test_schema_resolver.py`
- `tests/gnn/test_aggregator.py`
- `tests/gnn/test_scalable_graph_builder.py`
- `tests/gnn/test_kg_pipeline.py`
- `tests/gnn/test_event_graph_export.py`

## Notes
- LLM schema mapping is optional (off by default). Enable via settings only when needed
- Keep outputs under `artifacts/` for large exports; repo ignores caches and artifacts
