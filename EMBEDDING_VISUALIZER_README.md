# 🏏 GNN Embedding Visualizer

A comprehensive Streamlit application for interactive visualization and exploration of GNN player embeddings in cricket analytics.

## ✨ Features

### 🔹 **Embedding Loading**
- **File Support**: Load embeddings from Pickle (`.pkl`) or JSON (`.json`) files
- **Sample Data**: Built-in sample embedding generation for testing
- **Error Handling**: Robust loading with comprehensive error messages
- **Metadata Integration**: Automatic metadata extraction from embedding files

### 🔹 **UMAP Dimensionality Reduction**
- **2D Projection**: Reduce 128D embeddings to 2D for visualization
- **Configurable Parameters**: Adjustable `n_neighbors` and `min_dist`
- **Deterministic Results**: Fixed random seeds for reproducible projections
- **Performance Optimized**: Efficient projection for large player datasets

### 🔹 **Interactive Visualization**
- **Plotly Integration**: Rich, interactive scatter plots with zoom/pan
- **Hover Information**: Detailed player stats on hover
- **Click Selection**: Select points to view detailed player information
- **Color Schemes**: Color by team, role, or batting average
- **Professional Styling**: Clean, modern plot aesthetics

### 🔹 **Advanced Filtering**
- **Team Filter**: Multi-select team filtering
- **Role Filter**: Filter by player roles (batter, bowler, all-rounder, etc.)
- **Season Filter**: Filter by playing seasons
- **Combined Filters**: Apply multiple filters simultaneously
- **Real-time Updates**: Instant plot updates on filter changes

### 🔹 **Player Details**
- **Comprehensive Stats**: Team, role, batting average, matches played
- **Video Links**: Optional links to player video highlights
- **Interactive Selection**: Click plot points to view detailed information
- **Export Ready**: Formatted data suitable for reports and presentations

## 🚀 Usage

### **Launch the Application**
```bash
streamlit run embedding_visualizer.py
```

### **Load Embeddings**
1. **Upload File**: Use the sidebar file uploader for `.pkl` or `.json` files
2. **Sample Data**: Select "Use Sample Data" for testing and demonstration

### **Configure Visualization**
1. **UMAP Parameters**: Adjust `n_neighbors` (5-50) and `min_dist` (0.01-1.0)
2. **Color Scheme**: Choose from team, role, or batting average coloring
3. **Filters**: Apply team, role, and season filters as needed

### **Explore Results**
1. **Interactive Plot**: Hover over points for player information
2. **Click Selection**: Click points to view detailed player stats
3. **Clustering Analysis**: Observe natural player groupings in 2D space

## 📊 Expected Input Format

### **Pickle Files (.pkl)**
```python
{
    "embeddings": {
        "player_id_1": numpy.array([...]),  # 128D embedding
        "player_id_2": numpy.array([...]),
        ...
    },
    "model_type": "GraphSAGE",
    "embedding_dim": 128,
    "training_date": "2024-01-15"
}
```

### **JSON Files (.json)**
```json
{
    "player_id_1": [0.1, 0.2, ..., 0.128],
    "player_id_2": [0.3, 0.4, ..., 0.256],
    ...
}
```

## 🎯 Key Components

### **EmbeddingLoader**
- Handles file loading and format conversion
- Supports both pickle and JSON formats
- Generates sample embeddings for testing
- Robust error handling and validation

### **PlayerMetadata**
- Creates realistic player metadata
- Assigns teams based on player names
- Generates batting averages and match statistics
- Provides video links for enhanced exploration

### **UMAPProjector**
- Configurable UMAP dimensionality reduction
- Deterministic projections with fixed random seeds
- Optimized for cricket player embedding analysis
- Cosine distance metric for similarity-based clustering

### **Interactive Plotting**
- Plotly-based scatter plots with rich interactivity
- Multiple color schemes (categorical and continuous)
- Hover tooltips with comprehensive player information
- Professional styling with customizable layouts

## 🧪 Testing

The application includes comprehensive test coverage:

```bash
# Run all tests
python -m pytest tests/test_embedding_visualizer.py -v

# Run specific test categories
python -m pytest tests/test_embedding_visualizer.py::TestEmbeddingLoader -v
python -m pytest tests/test_embedding_visualizer.py::TestUMAPProjector -v
python -m pytest tests/test_embedding_visualizer.py::TestInteractivePlot -v
```

### **Test Categories**
- **Embedding Loading**: File format support, error handling, sample generation
- **Player Metadata**: Metadata creation, team assignments, statistics
- **UMAP Projection**: Dimensionality reduction, parameter handling, determinism
- **Interactive Plotting**: Plot creation, color schemes, interactivity
- **Filtering**: Multi-level filtering, edge cases, performance
- **Streamlit Integration**: End-to-end workflows, error handling

## 📈 Sample Results

### **Clustering Analysis**
The visualizer reveals natural player clusters based on:
- **Team Affiliation**: Players from same teams often cluster together
- **Playing Role**: Batters, bowlers, and all-rounders form distinct groups
- **Performance Level**: High-performing players show similar embedding patterns
- **Playing Style**: Similar techniques and approaches create embedding similarity

### **Interactive Exploration**
- **Hover Details**: Instant access to player statistics
- **Filter Combinations**: Explore specific team/role combinations
- **Outlier Detection**: Identify unique players with distinct embedding patterns
- **Performance Correlation**: Observe relationships between embeddings and batting averages

## 🔧 Dependencies

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
umap-learn>=0.5.3
scikit-learn>=1.3.0
```

## 🎨 UI Features

### **Modern Design**
- Clean, professional interface
- Responsive layout for different screen sizes
- Intuitive controls and navigation
- Consistent color schemes and typography

### **Real-time Feedback**
- Loading indicators for UMAP computation
- Success/error messages for file operations
- Dynamic statistics updates
- Interactive plot responsiveness

### **Export Capabilities**
- Plot export via Plotly's built-in tools
- Data export through Streamlit's download features
- Screenshot capabilities for presentations
- Embeddable visualizations for reports

## 🚀 Advanced Use Cases

### **Research Applications**
- **Player Similarity Analysis**: Identify players with similar playing styles
- **Team Composition Studies**: Analyze team balance and player distributions
- **Performance Clustering**: Group players by performance characteristics
- **Style Evolution**: Track how player embeddings change over time

### **Scouting and Analytics**
- **Talent Identification**: Find players similar to known high performers
- **Opposition Analysis**: Understand opponent player characteristics
- **Transfer Strategy**: Identify players that fit team playing styles
- **Youth Development**: Compare emerging players to established stars

### **Model Validation**
- **Embedding Quality**: Visually assess GNN model performance
- **Cluster Coherence**: Validate that similar players cluster together
- **Outlier Analysis**: Identify unusual players or potential model issues
- **Feature Interpretation**: Understand what the embeddings capture

## 📝 Notes

- **Performance**: Optimized for datasets up to 1000+ players
- **Memory**: Efficient handling of high-dimensional embeddings
- **Scalability**: Configurable parameters for different dataset sizes
- **Extensibility**: Modular design for easy feature additions

The GNN Embedding Visualizer provides a powerful, intuitive interface for exploring cricket player embeddings, enabling deep insights into player similarities, team compositions, and performance patterns through interactive 2D visualization.