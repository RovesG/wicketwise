# WicketWise UI Launcher

## ⚠️ DEVELOPMENT UI ONLY - NOT PRODUCTION HARDENED

This Streamlit-based UI launcher provides a convenient interface for running WicketWise cricket analysis tools during development and testing. **This is NOT intended for production use.**

## Features

### 🔑 API Key Management
- **Secure Input**: Password-masked input fields for API keys
- **Session Storage**: Keys stored in Streamlit session state (not persisted)
- **Environment Variables**: Keys passed to subprocesses via environment variables
- **No Logging**: API keys are never logged or exposed in outputs

### 🚀 Main Functions
- **Training Pipeline**: Run `crickformers/train.py` with selected configuration
- **Model Evaluation**: Execute `eval.py` on test datasets
- **Post-Match Reports**: Generate analysis reports using `post_match_report.py`
- **Live Inference**: Process uploaded match files for real-time predictions

### 📊 Process Management
- **Status Tracking**: Real-time status updates for all running processes
- **Log Capture**: Comprehensive logging of stdout, stderr, and errors
- **Timeout Protection**: 5-minute timeout for all subprocess operations
- **Error Handling**: Graceful handling of process failures and timeouts

### ⚙️ Configuration
- **Dynamic Config Loading**: Automatically detects available configuration files
- **Checkpoint Selection**: Choose from available model checkpoints
- **Sample Data**: Pre-loaded sample datasets for testing

## Installation

### Prerequisites
```bash
# Python 3.9+ required
python --version

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation
```bash
# Check Streamlit installation
streamlit --version

# Verify core dependencies
python -c "import torch, pandas, numpy; print('Dependencies OK')"
```

## Usage

### Starting the UI
```bash
# Launch the Streamlit app
streamlit run ui_launcher.py

# Or specify port
streamlit run ui_launcher.py --server.port 8501
```

### Basic Workflow
1. **Configure API Keys**
   - Enter Betfair API key in sidebar
   - Enter OpenAI API key in sidebar
   - Click "Save API Keys"

2. **Select Configuration**
   - Choose training config from dropdown
   - Select model checkpoint for evaluation
   - Pick sample data file for testing

3. **Run Operations**
   - Click desired function button
   - Monitor status in right panel
   - View detailed logs in expandable sections

### API Key Requirements

#### Betfair API Key
- Required for betting data access
- Used by betting agents and live inference
- Format: Application key from Betfair Developer Program

#### OpenAI API Key
- Required for AI agent functionality
- Used by tactical agents and report generation
- Format: `sk-...` from OpenAI API dashboard

## File Structure

```
wicketwise/
├── ui_launcher.py              # Main Streamlit application
├── config_loader.py            # Configuration management
├── samples/                    # Test data samples
│   ├── test_match.csv         # Sample match data
│   └── evaluation_data.csv    # Sample evaluation data
├── config/                     # Configuration files
│   └── *.yaml                 # Training configurations
└── checkpoints/               # Model checkpoints
    └── *.pt                   # Trained model files
```

## Configuration Files

### Training Configurations
Located in `config/` directory:
- `train_config.yaml` - Standard training configuration
- `test_config.yaml` - Quick testing configuration
- `production_config.yaml` - Full production training

### Model Checkpoints
Located in `checkpoints/` directory:
- `*.pt` files - PyTorch model checkpoints
- Auto-detected by config loader

## Sample Data

### Test Match Data (`samples/test_match.csv`)
- 50 balls of realistic T20 cricket data
- Includes powerplay and middle overs
- Features wickets, boundaries, and partnerships
- Suitable for testing evaluation and inference

### Evaluation Data (`samples/evaluation_data.csv`)
- Multi-match dataset with different scenarios
- Includes both innings data
- Various match situations and outcomes
- Ideal for comprehensive model evaluation

## Process Operations

### Training Pipeline
```bash
# Equivalent command:
python crickformers/train.py --config config/selected_config.yaml
```
- Trains CrickformerModel with selected configuration
- Outputs checkpoints to configured directory
- Logs training progress and metrics

### Model Evaluation
```bash
# Equivalent command:
python eval.py --checkpoint selected_checkpoint.pt --data samples/selected_data.csv
```
- Evaluates trained model on test dataset
- Generates per-ball predictions
- Outputs results to `eval_predictions.csv`

### Post-Match Report
```bash
# Equivalent command:
python crickformers/post_match_report.py --data samples/selected_data.csv
```
- Generates comprehensive match analysis
- Includes statistical summaries and insights
- Outputs formatted report

### Live Inference
```bash
# Equivalent command:
python crickformers/live_inference.py --data uploaded_file.csv --checkpoint selected_checkpoint.pt
```
- Processes uploaded match file
- Generates real-time predictions
- Outputs live inference results

## Safety Features

### 🔒 Security Measures
- **No Persistence**: API keys not saved to disk
- **Environment Isolation**: Keys passed securely to subprocesses
- **Log Sanitization**: No sensitive data in logs
- **Session Scope**: Keys cleared when session ends

### ⚠️ Limitations
- **Development Only**: Not hardened for production use
- **Local Processing**: All operations run locally
- **No Authentication**: No user authentication system
- **No Encryption**: Session state not encrypted
- **No Rate Limiting**: No API rate limiting protection

### 🛡️ Best Practices
1. **Use Test Keys**: Use development/test API keys only
2. **Local Network**: Run on local network only
3. **Regular Updates**: Keep dependencies updated
4. **Monitor Logs**: Review process logs regularly
5. **Clear Sessions**: Clear browser data after use

## Troubleshooting

### Common Issues

#### API Keys Not Working
```
Error: Please configure API keys first
```
**Solution**: Ensure both Betfair and OpenAI keys are entered and saved

#### Process Timeouts
```
Error: Process timed out after 5 minutes
```
**Solution**: Use smaller datasets or increase timeout in code

#### Configuration Not Found
```
Error: Configuration file not found
```
**Solution**: Ensure config files exist in `config/` directory

#### Checkpoint Loading Errors
```
Error: Cannot load checkpoint
```
**Solution**: Verify checkpoint file exists and is compatible

### Debug Mode
Enable debug logging by modifying `ui_launcher.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Log Files
Process logs are captured in session state and displayed in UI. For persistent logging, modify subprocess calls to write to files.

## Development

### Adding New Functions
1. Create new button in main panel
2. Add corresponding subprocess command
3. Update process status tracking
4. Add appropriate error handling

### Extending Configuration
1. Modify `config_loader.py` to support new options
2. Add UI controls in sidebar
3. Update subprocess commands to use new parameters

### Custom Sample Data
1. Add CSV files to `samples/` directory
2. Update `config_loader.py` to detect new files
3. Ensure CSV format matches expected schema

## Support

For issues and questions:
1. Check process logs in UI
2. Review configuration files
3. Verify API key validity
4. Test with sample data first
5. Check system requirements

## License

This development UI is part of the WicketWise project and follows the same licensing terms. 