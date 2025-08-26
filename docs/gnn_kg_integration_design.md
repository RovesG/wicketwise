# GNN-KG Integration Design Document
# Purpose: Enhance Knowledge Graph Query System with GNN Embeddings
# Author: WicketWise Team, Last Modified: 2025-08-25

## 🎯 **Objective**
Integrate Graph Neural Network (GNN) embeddings with the Knowledge Graph query system to enable:
- Semantic similarity searches using learned embeddings
- Contextual performance predictions
- Advanced player/venue compatibility analysis
- Style-based player comparisons

## 🏗️ **Current Architecture Analysis**

### **Existing Components:**
1. **UnifiedKGQueryEngine**: Handles structured KG queries
2. **KGGNNEmbeddingService**: Generates and manages GNN embeddings
3. **CricketGNNTrainer**: Trains GNN models on cricket data
4. **KGChatAgent**: LLM interface with function calling

### **Current Limitations:**
- No embedding-based similarity in query system
- Limited contextual analysis capabilities
- No GNN-powered predictions in chat interface
- Separate GNN and KG systems without integration

## 🚀 **Proposed Enhancement Architecture**

### **1. Enhanced UnifiedKGQueryEngine**
```python
class GNNEnhancedKGQueryEngine(UnifiedKGQueryEngine):
    """
    Enhanced KG Query Engine with GNN embedding capabilities
    """
    def __init__(self, graph_path: str, gnn_embeddings_path: str):
        super().__init__(graph_path)
        self.gnn_service = KGGNNEmbeddingService(gnn_embeddings_path)
        
    # New GNN-powered methods
    def find_similar_players_gnn(self, player: str, top_k: int = 5)
    def get_contextual_performance_prediction(self, player: str, context: Dict)
    def analyze_venue_compatibility(self, player: str, venue: str)
    def get_playing_style_similarity(self, player1: str, player2: str)
```

### **2. New Function Tools for LLM**
```python
# Enhanced function_tools.py
GNN_FUNCTION_TOOLS = [
    {
        "name": "find_similar_players_gnn",
        "description": "Find players with similar playing styles using GNN embeddings"
    },
    {
        "name": "predict_contextual_performance", 
        "description": "Predict player performance in specific match contexts"
    },
    {
        "name": "analyze_venue_compatibility",
        "description": "Analyze how well a player might perform at a specific venue"
    }
]
```

### **3. Integration Points**
- **KGChatAgent**: Add GNN function mappings
- **Function Tools**: Extend with GNN-powered functions
- **UnifiedKGQueryEngine**: Integrate GNN service
- **Testing Suite**: Comprehensive GNN integration tests

## 📊 **Technical Specifications**

### **Embedding Dimensions:**
- Player embeddings: 128D (batting + bowling + context)
- Venue embeddings: 64D (conditions + characteristics)
- Context embeddings: 32D (match situation)

### **Similarity Metrics:**
- Cosine similarity for style comparison
- Euclidean distance for performance prediction
- Weighted combinations for contextual analysis

### **Performance Requirements:**
- Query response time: < 500ms
- Embedding lookup: < 50ms
- Similarity computation: < 100ms

## 🧪 **Testing Strategy**

### **Unit Tests:**
- GNN service integration
- Embedding similarity calculations
- Function tool execution
- Error handling and fallbacks

### **Integration Tests:**
- End-to-end query workflows
- LLM function calling with GNN
- Performance benchmarking
- Data consistency validation

### **Cricket Domain Tests:**
- Player similarity accuracy
- Venue compatibility predictions
- Style analysis validation
- Contextual performance tests

## 🎨 **Implementation Phases**

### **Phase 1: Core Integration** (Current)
1. Create GNNEnhancedKGQueryEngine
2. Add basic similarity functions
3. Integrate with existing KG system
4. Basic unit tests

### **Phase 2: Advanced Functions**
1. Contextual performance prediction
2. Venue compatibility analysis
3. Style similarity metrics
4. Enhanced function tools

### **Phase 3: LLM Integration**
1. Update KGChatAgent function mappings
2. Add new function tool definitions
3. Test LLM function calling
4. Performance optimization

### **Phase 4: Testing & Validation**
1. Comprehensive test suite
2. Cricket domain validation
3. Performance benchmarking
4. Documentation and examples

## 🏏 **Cricket Use Cases**

### **Player Similarity:**
- "Find batsmen similar to Virat Kohli's aggressive style"
- "Which players have similar bowling actions to Jasprit Bumrah?"

### **Contextual Predictions:**
- "How would Buttler perform in death overs at MCG?"
- "Predict Rashid Khan's effectiveness against left-handed batsmen"

### **Venue Compatibility:**
- "Which venues suit Rohit Sharma's playing style?"
- "How do spinners typically perform at Eden Gardens?"

### **Style Analysis:**
- "Compare the batting approaches of Smith and Williamson"
- "Find bowlers with similar pace and swing characteristics"

## 🔧 **Configuration**

### **Model Paths:**
```python
GNN_CONFIG = {
    "embeddings_path": "models/gnn_embeddings.pt",
    "model_path": "models/cricket_gnn_model.pt", 
    "kg_path": "models/unified_cricket_kg.pkl",
    "similarity_threshold": 0.7,
    "max_results": 10
}
```

### **Performance Tuning:**
- Embedding cache size: 10,000 entities
- Query timeout: 15 seconds
- Batch processing: 100 entities
- Memory optimization: Lazy loading

## 📈 **Success Metrics**

### **Technical Metrics:**
- Query response time improvement: 20%
- Similarity accuracy: >85%
- Function call success rate: >95%
- Memory usage: <500MB additional

### **Cricket Intelligence Metrics:**
- Player similarity relevance: >90%
- Venue prediction accuracy: >80%
- Style analysis coherence: >85%
- User satisfaction: Qualitative feedback

## 🚀 **Future Enhancements**

### **Advanced Features:**
- Real-time embedding updates
- Multi-modal embeddings (video + stats)
- Temporal embedding evolution
- Cross-format compatibility analysis

### **Integration Opportunities:**
- Betting strategy optimization
- Team selection recommendations
- Match outcome predictions
- Fantasy cricket insights

---

**Next Steps:** Begin Phase 1 implementation with GNNEnhancedKGQueryEngine creation.
