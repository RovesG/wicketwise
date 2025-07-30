# 🏏 Cricket AI Admin Panel

A Streamlit-based admin panel for managing cricket AI machine learning workflows, including knowledge graph construction, GNN embeddings training, and Crickformer model training.

## 🎯 Features

- **🔧 Backend Job Triggers**: One-click buttons for ML pipeline tasks
- **📊 System Status Dashboard**: Real-time monitoring of model training status
- **📋 Status Log**: Sidebar with action history and completion confirmations
- **🐛 Debug Tools**: Reset functionality and system health monitoring
- **🏏 Cricket AI Integration**: Built for Phi1618 Cricket AI principles

## 🏗️ Architecture

### Admin Panel Components

```
cricket_ai_admin/
├── ui_launcher.py          # Main Streamlit interface
├── admin_tools.py          # Backend stub functions
├── tests/
│   └── test_admin_tools.py # Unit tests for admin tools
├── requirements.txt        # Python dependencies
└── README.md              # This documentation
```

### Backend Jobs

1. **📊 Build Knowledge Graph**
   - Constructs cricket knowledge graph from ball-by-ball data
   - Extracts player, match, and venue relationships
   - Placeholder for NetworkX graph building

2. **🧠 Train GNN Embeddings**
   - Trains graph neural network embeddings
   - Uses knowledge graph as input
   - Placeholder for PyTorch Geometric implementation

3. **🤖 Train Crickformer Model**
   - Trains transformer model for cricket predictions
   - Ball-by-ball sequence modeling
   - Placeholder for attention-based architecture

4. **📈 Run Evaluation**
   - Evaluates trained models on test data
   - Calculates accuracy, F1, precision, recall
   - Placeholder for evaluation pipeline

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the admin panel**:
   ```bash
   streamlit run ui_launcher.py
   ```

3. **Access the interface**:
   - Open browser to `http://localhost:8501`
   - The admin panel will be available with all buttons

## 🎮 Usage

### Main Interface

- **Backend Jobs Section**: Click any button to trigger ML tasks
- **System Status**: Monitor the current state of all components
- **Status Log (Sidebar)**: View recent actions and completion messages

### Available Actions

| Button | Function | Expected Output |
|--------|----------|----------------|
| 📊 Build Knowledge Graph | `admin_tools.build_knowledge_graph()` | "Knowledge graph building complete" |
| 🧠 Train GNN Embeddings | `admin_tools.train_gnn_embeddings()` | "GNN training complete" |
| 🤖 Train Crickformer Model | `admin_tools.train_crickformer_model()` | "Crickformer training complete" |
| 📈 Run Evaluation | `admin_tools.run_evaluation()` | "Evaluation complete" |

### Debug Features

- **Clear Log**: Reset the sidebar status messages
- **Reset System**: Clear all status messages and reset state
- **Live Timestamps**: Current time display in sidebar

### Print Statements

Each function outputs a log message when called:
- `[LOG] Knowledge graph building started...`
- `[LOG] GNN training started...`
- `[LOG] Crickformer training started...`
- `[LOG] Evaluation started...`

## 🧪 Testing

Run the test suite to verify all functions work correctly:

```bash
# Run all tests
pytest tests/test_admin_tools.py -v

# Run specific test categories
pytest tests/test_admin_tools.py::TestAdminTools -v
pytest tests/test_admin_tools.py::TestAdminToolsIntegration -v
```

### Test Coverage

- ✅ Each stub function returns correct completion string
- ✅ Print statements are called with correct log messages
- ✅ System status structure and data types
- ✅ Global admin_tools instance initialization
- ✅ Complete workflow execution without errors
- ✅ Default system health and status values

## 🔧 Engineering Principles

Following **Phi1618 Cricket AI** standards:

### Code Structure
- **Modular Design**: Separate UI, backend, and tests
- **Clean Functions**: Each function ≤ 40 lines
- **Clear Documentation**: Comprehensive docstrings and comments

### Scalability
- **Cloud-Ready**: Environment-agnostic design
- **Agent Integration**: Ready for OpenAI Agent SDK
- **Real-time Capable**: Built for streaming data ingestion

### Quality Assurance
- **Test Coverage**: Unit and integration tests
- **Type Hints**: Full typing support
- **Error Handling**: Graceful failure modes

## 📋 Future Enhancements

### Phase 1: Core Implementation
- [ ] Replace stub functions with actual ML pipelines
- [ ] Add progress bars for long-running tasks
- [ ] Implement persistent state management

### Phase 2: Advanced Features
- [ ] Real-time model training logs
- [ ] Hyperparameter tuning interface
- [ ] Model performance visualization

### Phase 3: Production Ready
- [ ] Authentication and authorization
- [ ] Multi-user support
- [ ] Deployment configuration

## 🛠️ Development

### Project Structure Guidelines

```python
# File header template
# Purpose: [Brief description]
# Author: Phi1618 Cricket AI Team, Last Modified: [Date]
```

### Adding New Admin Functions

1. **Add function to `admin_tools.py`**:
   ```python
   def new_ml_task(self) -> str:
       """Description of the new ML task."""
       print("[LOG] New ML task started...")
       # Implementation here
       return "New ML task complete"
   ```

2. **Add button to `ui_launcher.py`**:
   ```python
   if st.button("🎯 New ML Task"):
       result = admin_tools.new_ml_task()
       st.success(result)
   ```

3. **Add test to `tests/test_admin_tools.py`**:
   ```python
   def test_new_ml_task_returns_correct_message(self):
       result = self.admin_tools.new_ml_task()
       assert result == "New ML task complete"
   ```

### Debugging

- Check Streamlit logs for runtime errors
- Use `st.write()` for debugging UI state
- Monitor console output for backend function calls

## 📝 License

MIT License - Built for Phi1618 Cricket AI research and development.

## 🤝 Contributing

1. Follow the engineering principles above
2. Add comprehensive tests for new features
3. Update documentation for any changes
4. Ensure all functions are properly typed

---

**Built with 🏏 Phi1618 Cricket AI Engineering Principles**  
*Scalable • Modular • Agent-Ready • Cloud-Deployable* 