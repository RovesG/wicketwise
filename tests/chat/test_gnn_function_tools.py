# Purpose: Tests for GNN-Enhanced Function Tools
# Author: WicketWise Team, Last Modified: 2025-08-25

import pytest
from crickformers.chat.gnn_function_tools import (
    get_gnn_function_tools,
    get_enhanced_function_descriptions,
    get_all_enhanced_function_tools,
    get_all_enhanced_function_descriptions,
    format_gnn_insight,
    GNN_ANALYSIS_TEMPLATES
)


class TestGNNFunctionTools:
    """Test suite for GNN function tools"""
    
    def test_get_gnn_function_tools_structure(self):
        """Test that GNN function tools have correct structure"""
        tools = get_gnn_function_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        # Check each tool has required structure
        for tool in tools:
            assert "type" in tool
            assert tool["type"] == "function"
            assert "function" in tool
            
            func_def = tool["function"]
            assert "name" in func_def
            assert "description" in func_def
            assert "parameters" in func_def
            
            # Check parameters structure
            params = func_def["parameters"]
            assert "type" in params
            assert params["type"] == "object"
            assert "properties" in params
    
    def test_gnn_function_names(self):
        """Test that all expected GNN functions are present"""
        tools = get_gnn_function_tools()
        function_names = [tool["function"]["name"] for tool in tools]
        
        expected_functions = [
            "find_similar_players_gnn",
            "predict_contextual_performance",
            "analyze_venue_compatibility",
            "get_playing_style_similarity",
            "find_best_performers_contextual",
            "analyze_team_composition_gnn"
        ]
        
        for expected in expected_functions:
            assert expected in function_names, f"Missing function: {expected}"
    
    def test_find_similar_players_gnn_parameters(self):
        """Test find_similar_players_gnn function parameters"""
        tools = get_gnn_function_tools()
        func = next(t for t in tools if t["function"]["name"] == "find_similar_players_gnn")
        
        params = func["function"]["parameters"]["properties"]
        
        # Required parameter
        assert "player" in params
        assert params["player"]["type"] == "string"
        
        # Optional parameters with defaults
        assert "top_k" in params
        assert params["top_k"]["type"] == "integer"
        assert params["top_k"]["default"] == 5
        assert params["top_k"]["minimum"] == 1
        assert params["top_k"]["maximum"] == 15
        
        assert "similarity_metric" in params
        assert params["similarity_metric"]["enum"] == ["cosine", "euclidean"]
        
        assert "min_similarity" in params
        assert params["min_similarity"]["type"] == "number"
        assert 0.0 <= params["min_similarity"]["default"] <= 1.0
    
    def test_predict_contextual_performance_parameters(self):
        """Test contextual performance prediction parameters"""
        tools = get_gnn_function_tools()
        func = next(t for t in tools if t["function"]["name"] == "predict_contextual_performance")
        
        params = func["function"]["parameters"]["properties"]
        
        # Required parameters
        assert "player" in params
        assert "context" in params
        
        # Context object structure
        context_props = params["context"]["properties"]
        assert "phase" in context_props
        assert context_props["phase"]["enum"] == ["powerplay", "middle", "death"]
        
        assert "bowling_type" in context_props
        assert context_props["bowling_type"]["enum"] == ["pace", "spin"]
        
        assert "pressure" in context_props
        assert context_props["pressure"]["type"] == "boolean"
    
    def test_analyze_venue_compatibility_parameters(self):
        """Test venue compatibility analysis parameters"""
        tools = get_gnn_function_tools()
        func = next(t for t in tools if t["function"]["name"] == "analyze_venue_compatibility")
        
        params = func["function"]["parameters"]["properties"]
        required = func["function"]["parameters"]["required"]
        
        assert "player" in params
        assert "venue" in params
        assert "player" in required
        assert "venue" in required
    
    def test_get_enhanced_function_descriptions(self):
        """Test enhanced function descriptions"""
        descriptions = get_enhanced_function_descriptions()
        
        assert isinstance(descriptions, dict)
        assert len(descriptions) > 0
        
        # Check all GNN functions have descriptions
        gnn_functions = [
            "find_similar_players_gnn",
            "predict_contextual_performance", 
            "analyze_venue_compatibility",
            "get_playing_style_similarity"
        ]
        
        for func_name in gnn_functions:
            assert func_name in descriptions
            assert isinstance(descriptions[func_name], str)
            assert len(descriptions[func_name]) > 50  # Substantial description
    
    def test_get_all_enhanced_function_tools_combines_correctly(self):
        """Test that all enhanced tools combine original and GNN tools"""
        all_tools = get_all_enhanced_function_tools()
        gnn_tools = get_gnn_function_tools()
        
        # Should have more tools than just GNN tools
        assert len(all_tools) > len(gnn_tools)
        
        # All GNN tool names should be present
        all_names = [tool["function"]["name"] for tool in all_tools]
        gnn_names = [tool["function"]["name"] for tool in gnn_tools]
        
        for gnn_name in gnn_names:
            assert gnn_name in all_names
    
    def test_get_all_enhanced_function_descriptions_combines_correctly(self):
        """Test that all enhanced descriptions combine original and GNN descriptions"""
        all_descriptions = get_all_enhanced_function_descriptions()
        gnn_descriptions = get_enhanced_function_descriptions()
        
        # Should have more descriptions than just GNN
        assert len(all_descriptions) > len(gnn_descriptions)
        
        # All GNN descriptions should be present
        for gnn_name, gnn_desc in gnn_descriptions.items():
            assert gnn_name in all_descriptions
            assert all_descriptions[gnn_name] == gnn_desc
    
    def test_format_gnn_insight_templates(self):
        """Test GNN insight formatting with templates"""
        # Test similarity high template
        insight = format_gnn_insight(
            "similarity_high",
            player1="Virat Kohli",
            player2="Steve Smith", 
            similarity=0.85,
            aspects="aggressive batting, strong technique"
        )
        
        assert "Virat Kohli" in insight
        assert "Steve Smith" in insight
        assert "85.0%" in insight
        assert "aggressive batting" in insight
    
    def test_format_gnn_insight_venue_templates(self):
        """Test venue-related insight templates"""
        # Test excellent venue match
        insight = format_gnn_insight(
            "venue_excellent",
            player="Jos Buttler",
            venue="MCG",
            compatibility=0.92,
            factors="aggressive style, large boundaries"
        )
        
        assert "Jos Buttler" in insight
        assert "MCG" in insight
        assert "92.0%" in insight
        assert "aggressive style" in insight
    
    def test_format_gnn_insight_contextual_templates(self):
        """Test contextual insight templates"""
        # Test strong contextual fit
        insight = format_gnn_insight(
            "contextual_strong",
            player="MS Dhoni",
            context="death overs",
            confidence=0.88,
            metrics="SR: 140+, Avg: 35+"
        )
        
        assert "MS Dhoni" in insight
        assert "death overs" in insight
        assert "88.0%" in insight
        assert "SR: 140+" in insight
    
    def test_format_gnn_insight_missing_template(self):
        """Test insight formatting with missing template"""
        insight = format_gnn_insight(
            "nonexistent_template",
            analysis="Some analysis"
        )
        
        assert "GNN Analysis" in insight
        assert "Some analysis" in insight
    
    def test_format_gnn_insight_missing_variable(self):
        """Test insight formatting with missing template variable"""
        insight = format_gnn_insight(
            "similarity_high",
            player1="Virat Kohli"
            # Missing player2, similarity, aspects
        )
        
        assert "Missing template variable" in insight
    
    def test_gnn_analysis_templates_completeness(self):
        """Test that all analysis templates are properly defined"""
        expected_templates = [
            "similarity_high",
            "similarity_low", 
            "venue_excellent",
            "venue_poor",
            "contextual_strong",
            "contextual_weak"
        ]
        
        for template in expected_templates:
            assert template in GNN_ANALYSIS_TEMPLATES
            assert isinstance(GNN_ANALYSIS_TEMPLATES[template], str)
            assert len(GNN_ANALYSIS_TEMPLATES[template]) > 20
    
    def test_function_parameter_validation(self):
        """Test that function parameters have proper validation"""
        tools = get_gnn_function_tools()
        
        for tool in tools:
            func_def = tool["function"]
            params = func_def["parameters"]["properties"]
            
            # Check numeric parameters have proper bounds
            for param_name, param_def in params.items():
                if param_def.get("type") == "number":
                    if "minimum" in param_def and "maximum" in param_def:
                        assert param_def["minimum"] <= param_def["maximum"]
                
                if param_def.get("type") == "integer":
                    if "minimum" in param_def and "maximum" in param_def:
                        assert param_def["minimum"] <= param_def["maximum"]
                    
                    if "default" in param_def:
                        default = param_def["default"]
                        if "minimum" in param_def:
                            assert default >= param_def["minimum"]
                        if "maximum" in param_def:
                            assert default <= param_def["maximum"]
    
    def test_cricket_domain_specificity(self):
        """Test that functions are cricket-domain specific"""
        tools = get_gnn_function_tools()
        descriptions = get_enhanced_function_descriptions()
        
        cricket_terms = [
            "cricket", "player", "batting", "bowling", "venue", 
            "match", "performance", "style", "wicket", "over"
        ]
        
        # Check function descriptions contain cricket terminology
        for func_name, description in descriptions.items():
            description_lower = description.lower()
            has_cricket_terms = any(term in description_lower for term in cricket_terms)
            assert has_cricket_terms, f"Function {func_name} lacks cricket-specific terminology"
    
    def test_enum_parameter_values(self):
        """Test that enum parameters have valid cricket values"""
        tools = get_gnn_function_tools()
        
        for tool in tools:
            params = tool["function"]["parameters"]["properties"]
            
            for param_name, param_def in params.items():
                if "enum" in param_def:
                    enum_values = param_def["enum"]
                    
                    # Check cricket-specific enums
                    if param_name == "phase":
                        expected_phases = ["powerplay", "middle", "death"]
                        assert set(enum_values) == set(expected_phases)
                    
                    elif param_name == "bowling_type":
                        expected_types = ["pace", "spin"]
                        assert set(enum_values) == set(expected_types)
                    
                    elif param_name == "role":
                        cricket_roles = ["batsman", "bowler", "all-rounder", "wicket-keeper", "any"]
                        for value in enum_values:
                            assert value in cricket_roles
    
    def test_function_descriptions_quality(self):
        """Test quality and completeness of function descriptions"""
        descriptions = get_enhanced_function_descriptions()
        
        quality_indicators = [
            "GNN", "embedding", "analysis", "cricket", "performance"
        ]
        
        for func_name, description in descriptions.items():
            # Should be substantial
            assert len(description) > 100, f"Description for {func_name} too short"
            
            # Should mention GNN/embedding technology
            description_lower = description.lower()
            has_tech_terms = any(term in description_lower for term in ["gnn", "embedding", "neural"])
            assert has_tech_terms, f"Description for {func_name} doesn't mention GNN technology"
            
            # Should be cricket-specific
            has_cricket_terms = any(term in description_lower for term in ["cricket", "player", "match"])
            assert has_cricket_terms, f"Description for {func_name} not cricket-specific"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
