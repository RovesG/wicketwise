# 🏏 Crickformer Model Evaluation

A comprehensive evaluation script for trained Crickformer models that generates per-ball predictions on test data and saves detailed results to CSV for analysis.

## 🎯 Purpose

The evaluation script (`eval.py`) provides a complete pipeline for:

- **Loading trained models** from checkpoint files
- **Processing test data** using the CrickformerDataset
- **Generating predictions** for all three prediction heads
- **Logging per-ball results** to CSV for detailed analysis
- **Supporting both CLI and programmatic interfaces**

## 📊 Output Format

The evaluation script generates a CSV file with the following columns:

### Required Columns (as specified)
- `match_id`: Identifier for the cricket match
- `ball_id`: Unique identifier for each ball
- `actual_runs`: Actual runs scored on the ball (0-6)
- `predicted_runs_class`: Predicted outcome class (0_runs, 1_run, 2_runs, 3_runs, 4_runs, 6_runs, wicket)
- `win_prob`: Predicted win probability (0.0-1.0)
- `odds_mispricing`: Predicted odds mispricing probability (0.0-1.0)
- `phase`: Match phase (powerplay, middle_overs, death_overs)
- `batter_id`: Identifier for the batter
- `bowler_id`: Identifier for the bowler

### Additional Columns (for detailed analysis)
- `predicted_runs_0` to `predicted_runs_6`: Individual probabilities for each outcome
- `predicted_wicket`: Probability of wicket
- `actual_win_prob`: Actual win probability from data
- `actual_mispricing`: Actual mispricing from data

## 🚀 Quick Start

### Command Line Usage

```bash
# Basic evaluation
python3 eval.py --checkpoint models/trained_model.pt --data-path /path/to/data

# Advanced usage with custom settings
python3 eval.py \
    --checkpoint models/crickformer_epoch_10.pt \
    --data-path "/Users/user/Cricket Data" \
    --output detailed_predictions.csv \
    --batch-size 64 \
    --test-split 0.3 \
    --config config/eval_config.yaml
```

### Python API Usage

```python
from eval import CrickformerEvaluator

# Initialize evaluator
config = {"batch_size": 32, "test_split": 0.2}
evaluator = CrickformerEvaluator(config)

# Load model and setup data
evaluator.load_model("models/trained_model.pt")
evaluator.setup_dataset("/path/to/data", use_csv=True)

# Run evaluation
total_predictions = evaluator.evaluate_model("predictions.csv")
print(f"Generated {total_predictions:,} predictions")
```

## 📋 Requirements

### Dependencies
- **PyTorch** for model loading and inference
- **pandas** for data processing
- **tqdm** for progress tracking
- **numpy** for numerical operations

### Data Requirements
- **Trained checkpoint** (.pt file) with model weights and configuration
- **Test data** in CSV format (nvplay_data_v3.csv, decimal_data_v3.csv)
- **Compatible model architecture** matching the training configuration

## ⚙️ Configuration Options

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint` | Path to trained model checkpoint | **Required** |
| `--data-path` | Path to cricket data directory | Current data path |
| `--output` | Path to save evaluation CSV | `eval_predictions.csv` |
| `--config` | Path to configuration file | `config/train_config.yaml` |
| `--batch-size` | Evaluation batch size | `32` |
| `--test-split` | Fraction of data for testing | `0.2` |
| `--use-csv` | Use CSV data adapter | `True` |

### Model Configuration

The evaluator automatically loads model configuration from the checkpoint file, including:

- **Sequence encoder** settings (Transformer parameters)
- **Static context encoder** settings (MLP and embedding dimensions)
- **Fusion layer** configuration
- **Prediction heads** parameters

## 🧪 Testing

### Unit Tests

Run the comprehensive test suite:

```bash
# Run all evaluation tests
python3 -m pytest tests/test_eval.py -v

# Test specific functionality
python3 -m pytest tests/test_eval.py::TestCrickformerEvaluator::test_evaluate_model_full_pipeline -v
```

### Test Coverage (14 tests)

- ✅ **Evaluator initialization** and configuration
- ✅ **Model loading** from checkpoint files
- ✅ **Dataset setup** with CSV adapter
- ✅ **Outcome extraction** from batches
- ✅ **Metadata extraction** for CSV output
- ✅ **Full evaluation pipeline** with real data
- ✅ **CSV structure validation** and data types
- ✅ **Error handling** for missing models/data
- ✅ **Probability consistency** checks
- ✅ **Device handling** (CPU/CUDA)

## 📈 Performance Characteristics

### Processing Speed
- **~100-500 samples/second** depending on hardware
- **Batch processing** for efficient GPU utilization
- **Progress tracking** with detailed logging

### Memory Usage
- **Minimal memory footprint** - processes data in batches
- **Automatic cleanup** of intermediate tensors
- **Scalable** to large datasets

### Accuracy
- **Exact model reproduction** from training checkpoints
- **Deterministic results** with consistent data ordering
- **Full precision** probability outputs

## 🔧 Model Architecture Support

### Supported Components
- **CrickformerModel** with all prediction heads
- **CSV data adapter** for real cricket data
- **Multi-task learning** outputs
- **GNN embeddings** and attention mechanisms

### Prediction Heads
1. **Win Probability**: Single probability output (0.0-1.0)
2. **Next Ball Outcome**: 7-class classification (0,1,2,3,4,6,wicket)
3. **Odds Mispricing**: Binary probability for value betting

## 📁 File Structure

```
eval.py                 # Main evaluation script
tests/test_eval.py      # Comprehensive test suite
config/
├── train_config.yaml  # Training configuration (loaded for compatibility)
models/
├── *.pt              # Trained model checkpoints
outputs/
├── eval_predictions.csv  # Evaluation results
```

## 🔍 Output Analysis

### CSV Analysis Examples

```python
import pandas as pd

# Load predictions
df = pd.read_csv('eval_predictions.csv')

# Accuracy analysis
accuracy = (df['actual_runs'] == df['predicted_runs_class'].str.extract('(\d+)').astype(int)).mean()
print(f"Prediction accuracy: {accuracy:.3f}")

# Win probability calibration
win_prob_bins = pd.cut(df['win_prob'], bins=10)
calibration = df.groupby(win_prob_bins)['actual_win_prob'].mean()
print("Win probability calibration:")
print(calibration)

# Phase-based analysis
phase_accuracy = df.groupby('phase').apply(
    lambda x: (x['actual_runs'] == x['predicted_runs_class'].str.extract('(\d+)').astype(int)).mean()
)
print("Accuracy by phase:")
print(phase_accuracy)
```

## 🐛 Troubleshooting

### Common Issues

1. **Checkpoint not found**
   ```
   Error: FileNotFoundError: Checkpoint not found
   Solution: Verify checkpoint path and file existence
   ```

2. **Model architecture mismatch**
   ```
   Error: RuntimeError: Error loading state_dict
   Solution: Ensure checkpoint matches current model architecture
   ```

3. **Data format issues**
   ```
   Error: KeyError: Required column not found
   Solution: Verify CSV files have correct column names
   ```

4. **Memory issues**
   ```
   Error: CUDA out of memory
   Solution: Reduce batch_size or use CPU evaluation
   ```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run evaluation with debug output
evaluator = CrickformerEvaluator(config)
# ... rest of evaluation
```

## 🔮 Future Enhancements

- **Real-time evaluation** for live match inference
- **Confidence intervals** for prediction uncertainty
- **Feature importance** analysis for model interpretability
- **Batch evaluation** for multiple checkpoints
- **Interactive visualization** of predictions
- **Performance benchmarking** against baseline models

## 📊 Example Output

```csv
match_id,ball_id,actual_runs,predicted_runs_class,win_prob,odds_mispricing,phase,batter_id,bowler_id
BBL_Match_1,ball_001,1,1_run,0.652,0.123,powerplay,warner_001,starc_001
BBL_Match_1,ball_002,0,0_runs,0.648,0.089,powerplay,warner_001,starc_001
BBL_Match_1,ball_003,4,4_runs,0.678,0.234,powerplay,smith_002,starc_001
```

## 🎯 Use Cases

### Model Validation
- **Performance assessment** on held-out test data
- **Comparison** between different model versions
- **Ablation studies** for model components

### Production Deployment
- **Batch prediction** for historical analysis
- **Model monitoring** and drift detection
- **A/B testing** for model improvements

### Research Analysis
- **Prediction calibration** studies
- **Feature importance** analysis
- **Error pattern** investigation

---

**🏏 Built for comprehensive cricket model evaluation and analysis** 