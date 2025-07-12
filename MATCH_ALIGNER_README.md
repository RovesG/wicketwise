# 🏏 Cricket Match Aligner

A Python module that identifies overlapping cricket matches across two different data sources using ball-by-ball sequence fingerprints.

## 🎯 Purpose

Cricket data often comes from multiple sources (nvplay.csv, decimal.csv) that contain the same matches but with different match IDs. This module uses ball-by-ball sequence analysis to identify which matches represent the same cricket game, enabling data integration and deduplication.

## 🔧 How It Works

Instead of relying on match IDs, the aligner:

1. **Extracts fingerprints** from the first N balls of each match
2. **Creates sequence tuples** of `(over, ball, batter, bowler, runs)`
3. **Compares fingerprints** using similarity scoring
4. **Identifies matches** above a configurable similarity threshold

## 📊 Key Features

- **Sequence-based matching**: Uses ball-by-ball events, not fuzzy string matching
- **Configurable parameters**: Adjustable fingerprint length and similarity threshold
- **Robust handling**: Gracefully handles missing data and different column structures
- **Multiple interfaces**: Both Python API and command-line interface
- **Comprehensive testing**: 14 test cases covering various scenarios

## 🚀 Quick Start

### Command Line Usage

```bash
python3 match_aligner.py nvplay.csv decimal.csv --output matched_matches.csv --threshold 0.9
```

### Python API Usage

```python
from match_aligner import MatchAligner, align_matches

# Simple function interface
matches = align_matches(
    'nvplay.csv', 
    'decimal.csv', 
    'output.csv',
    fingerprint_length=50,
    similarity_threshold=0.9
)

# Advanced class interface
aligner = MatchAligner('nvplay.csv', 'decimal.csv', fingerprint_length=50)
matches = aligner.find_matches(similarity_threshold=0.9)
aligner.save_matches(matches, 'output.csv')
```

## 📋 Requirements

- **Python 3.7+**
- **pandas** for CSV processing
- **numpy** for numerical operations

## 📁 Input Data Format

### NVPlay CSV Format
Required columns:
- `Match`: Match identifier
- `Innings`: Innings number
- `Over`: Over number
- `Ball`: Ball in over
- `Innings Ball`: Ball number in innings
- `Batter`: Batter name
- `Bowler`: Bowler name
- `Runs`: Runs scored

### Decimal CSV Format
Required columns:
- `date`: Match date
- `competition`: Competition name
- `home`: Home team
- `away`: Away team
- `innings`: Innings number
- `ball`: Ball number
- `batter`: Batter name (optional)
- `bowler`: Bowler name (optional)
- `runs`: Runs scored (optional)

## 📤 Output Format

The output CSV contains:
- `nvplay_match_id`: Match ID from nvplay data
- `decimal_match_id`: Match ID from decimal data
- `similarity_score`: Similarity score (0.0 to 1.0)

Example:
```csv
nvplay_match_id,decimal_match_id,similarity_score
BBL_Match_1,BBL_Australia_vs_England_2024-01-15,1.0
PSL_Match_2,PSL_Pakistan_vs_Sri Lanka_2024-01-16,0.85
```

## ⚙️ Configuration Options

### Fingerprint Length
- **Default**: 50 balls
- **Purpose**: Number of balls to use for match fingerprinting
- **Recommendation**: 30-50 balls for good accuracy without over-fitting

### Similarity Threshold
- **Default**: 0.9 (90%)
- **Purpose**: Minimum similarity score to consider a match
- **Recommendation**: 
  - 0.95+ for high confidence matches
  - 0.8-0.9 for moderate confidence
  - 0.6-0.8 for exploratory analysis

## 🧪 Testing

Run the comprehensive test suite:

```bash
python3 -m pytest tests/test_match_aligner.py -v
```

Test coverage includes:
- ✅ Data loading and fingerprint extraction
- ✅ Similarity calculation algorithms
- ✅ Match finding with various thresholds
- ✅ CSV output generation
- ✅ Edge cases and error handling
- ✅ Command-line interface
- ✅ Empty data and missing columns

## 🎮 Demo

Run the interactive demonstration:

```bash
python3 demo_match_aligner.py
```

This creates sample cricket data and shows:
- Fingerprint extraction process
- Similarity scoring with different thresholds
- Match identification results
- CSV output generation

## 📈 Performance Characteristics

### Time Complexity
- **O(N × M × F)** where:
  - N = number of nvplay matches
  - M = number of decimal matches
  - F = fingerprint length

### Memory Usage
- **Minimal**: Only stores fingerprints, not full datasets
- **Scalable**: Handles large datasets efficiently

### Accuracy
- **High precision**: Sequence-based matching is very reliable
- **Configurable recall**: Adjust threshold based on use case

## 🔍 Algorithm Details

### Fingerprint Creation
1. Sort balls by innings and ball number
2. Extract first N balls from each match
3. Create tuple sequence: `(over, ball, batter, bowler, runs)`

### Similarity Scoring
1. Compare tuples position by position
2. Match criteria: `(batter, bowler, runs)` must be identical
3. Score = matches / min(fingerprint_length1, fingerprint_length2)

### Match Identification
1. Compare all nvplay matches with all decimal matches
2. Calculate similarity score for each pair
3. Return matches above threshold

## 🛠️ Advanced Usage

### Custom Similarity Function
```python
class CustomMatchAligner(MatchAligner):
    def _calculate_similarity(self, fingerprint1, fingerprint2):
        # Custom similarity logic here
        return custom_score
```

### Batch Processing
```python
import glob

for nvplay_file in glob.glob('nvplay_*.csv'):
    for decimal_file in glob.glob('decimal_*.csv'):
        matches = align_matches(nvplay_file, decimal_file)
        # Process matches...
```

## 🐛 Troubleshooting

### Common Issues

1. **No matches found**
   - Check data format and column names
   - Try lower similarity threshold
   - Verify fingerprint length isn't too long

2. **Too many false positives**
   - Increase similarity threshold
   - Increase fingerprint length
   - Check for data quality issues

3. **Memory issues**
   - Reduce fingerprint length
   - Process files in smaller batches

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.INFO)

# Now run aligner with detailed logging
aligner = MatchAligner('nvplay.csv', 'decimal.csv')
```

## 🔮 Future Enhancements

- **Fuzzy matching**: Handle player name variations
- **Temporal alignment**: Account for ball timing differences
- **Confidence scoring**: Multi-factor similarity assessment
- **Parallel processing**: Speed up large dataset processing
- **Interactive GUI**: Visual match exploration interface

## 📜 License

This module follows the engineering principles established in the project:
- **Modular design**: Clean separation of concerns
- **Comprehensive testing**: High test coverage
- **Documentation first**: Clear usage examples
- **Scalability minded**: Designed for production use

## 🤝 Contributing

When extending this module:
1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update documentation
4. Ensure compatibility with existing interfaces

---

**🏏 Built for cricket data integration and analysis** 