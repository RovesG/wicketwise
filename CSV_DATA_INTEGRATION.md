# Cricket CSV Data Integration Guide

This guide explains how to use the CrickformerDataset with real cricket CSV data files.

## Overview

The CrickformerDataset now supports two data formats:
1. **Directory structure** (original format) - for structured file-based data
2. **CSV format** (new) - for real cricket data from CSV files

## Real Data Structure

Your cricket data consists of two main CSV files:
- `nvplay_data_v3.csv` (381MB) - Ball-by-ball tracking and event data
- `decimal_data_v3.csv` (609MB) - Betting odds and win probability data

### Data Statistics
- **406,432 balls** across **676 matches** in NVPlay data
- **916,090 records** across **3,983 matches** in decimal data
- Covers competitions like Big Bash League, Pakistan Super League, etc.

## Quick Start

```python
from crickformers.crickformer_dataset import CrickformerDataset
from crickformers.csv_data_adapter import CSVDataConfig

# Path to your real data directory
data_path = "/path/to/your/cricket/data"

# Create dataset with CSV adapter
dataset = CrickformerDataset(
    data_root=data_path,
    use_csv_adapter=True,  # Enable CSV mode
    csv_config=CSVDataConfig(),
    history_length=5,
    load_video=True,
    load_embeddings=True,
    load_market_odds=True
)

print(f"Loaded {len(dataset):,} samples from {len(dataset.get_match_ids())} matches")
```

## Data Components

Each sample from the dataset contains the following tensor components:

| Component | Shape | Description |
|-----------|-------|-------------|
| `numeric_ball_features` | `[15]` | Normalized numeric features (over, runs, scores, etc.) |
| `categorical_ball_features` | `[4]` | Encoded categorical features (competition, player types, etc.) |
| `ball_history` | `[5, 5]` | Recent 5-ball history with padding |
| `video_features` | `[99]` | Video signals (mock data for CSV format) |
| `video_mask` | `[1]` | Mask indicating video availability |
| `gnn_embeddings` | `[384]` | Player/venue embeddings (mock data for CSV format) |
| `market_odds` | `[7]` | Market odds and probabilities |
| `market_odds_mask` | `[1]` | Mask indicating odds availability |

## Features Extracted

### Current Ball Features
- **Match Context**: Competition, venue, teams
- **Ball Details**: Over, ball number, innings
- **Players**: Batter and bowler information
- **Outcomes**: Runs scored, extras, wickets
- **Ball Tracking**: Field positions, pitch coordinates
- **Match State**: Team scores, run rates, powerplay status

### Ball History
- Recent 5 balls with runs, extras, wickets
- Player information for each historical ball
- Automatic padding for matches with <5 balls

### Video Signals (Mock)
- Ball tracking confidence
- Player detection confidence
- Motion vectors and optical flow (simulated)

### GNN Embeddings (Mock)
- Batter embeddings (128-dim)
- Bowler embeddings (128-dim)
- Venue embeddings (64-dim)
- Edge embeddings (64-dim)

### Market Odds
- Win probability from decimal data
- Additional betting market placeholders

## Configuration Options

```python
from crickformers.csv_data_adapter import CSVDataConfig

config = CSVDataConfig(
    nvplay_file="nvplay_data_v3.csv",     # NVPlay data filename
    decimal_file="decimal_data_v3.csv",   # Decimal data filename
    max_history_length=5,                 # Ball history length
    default_embedding_dim=128             # Default embedding dimension
)

dataset = CrickformerDataset(
    data_root=data_path,
    use_csv_adapter=True,
    csv_config=config,
    history_length=5,                     # Override history length
    load_video=True,                      # Load video signals
    load_embeddings=True,                 # Load GNN embeddings
    load_market_odds=True                 # Load market odds
)
```

## Training Setup

```python
import torch
from torch.utils.data import DataLoader, random_split

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=0  # Use 0 for compatibility with large datasets
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# Training loop
for batch in train_loader:
    # batch is a dict with all tensor components
    numeric_features = batch['numeric_ball_features']  # Shape: [32, 15]
    categorical_features = batch['categorical_ball_features']  # Shape: [32, 4]
    history = batch['ball_history']  # Shape: [32, 5, 5]
    # ... use in your model
```

## Filtering and Analysis

```python
# Get all match IDs
match_ids = dataset.get_match_ids()
print(f"Available matches: {len(match_ids)}")

# Filter to specific matches
specific_matches = [
    "Sydney Sixers v Melbourne Stars",
    "Sydney Thunder v Brisbane Heat"
]
filtered_dataset = dataset.filter_by_match(specific_matches)
print(f"Filtered dataset: {len(filtered_dataset)} samples")

# Get sample information
sample_info = dataset.get_sample_info(0)
print(f"Sample info: {sample_info}")
```

## Data Preprocessing

The CSV adapter automatically handles:
- **String to float conversion** with fallbacks for "Unknown" values
- **Categorical encoding** for competitions, player types, etc.
- **Normalization** of numeric features using domain-specific statistics
- **Padding** of ball history sequences
- **Missing data handling** with zero vectors for unavailable components

## Performance Considerations

- **Memory Usage**: ~163MB estimated for full dataset in memory
- **Loading Time**: ~10-15 seconds for initial CSV parsing
- **Batch Processing**: Optimized for PyTorch DataLoader compatibility
- **Selective Loading**: Disable video/embeddings for faster loading:

```python
# Faster loading for large-scale training
dataset = CrickformerDataset(
    data_root=data_path,
    use_csv_adapter=True,
    load_video=False,        # Skip video processing
    load_embeddings=False,   # Skip embedding generation
    load_market_odds=True    # Keep odds for targets
)
```

## Example Usage

See `examples/real_data_example.py` for a complete example that demonstrates:
- Loading the full dataset
- Exploring data structure and statistics
- Creating train/validation splits
- Batch processing
- Match filtering
- Performance metrics

## Data Schema

The CSV adapter uses a specialized schema optimized for the real data format:

```python
# Current ball features include:
class CurrentBallFeatures:
    match_id: str
    competition_name: str
    venue: str
    innings: int
    over: float
    ball_in_over: int
    batter_name: str
    bowler_name: str
    runs_scored: int
    is_wicket: bool
    team_score: int
    # ... and more
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the wicketwise directory
2. **File Not Found**: Verify the data path points to the directory containing the CSV files
3. **Memory Issues**: Use selective loading or smaller batch sizes
4. **Slow Loading**: Consider caching preprocessed data for repeated use

### Error Handling

The adapter gracefully handles:
- Missing or "Unknown" values in numeric fields
- Mismatched data between NVPlay and decimal files
- Incomplete ball history for early match balls
- Missing market odds data

## Future Enhancements

- **Real Video Processing**: Integration with actual video feature extraction
- **Real GNN Embeddings**: Pre-computed player/venue embeddings
- **Data Caching**: Preprocessed data storage for faster subsequent loads
- **Advanced Filtering**: Date ranges, competition types, player-specific filters
- **Streaming**: Support for real-time data ingestion during live matches

## Integration with Existing Codebase

The CSV adapter is fully compatible with existing Crickformer components:
- Works with existing model architectures
- Compatible with training pipelines
- Supports the same data augmentation transforms
- Maintains the same tensor output format

This ensures you can use your real data with minimal changes to existing model code. 