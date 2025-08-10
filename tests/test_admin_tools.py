# Purpose: Unit tests for cricket AI admin tools
# Author: Phi1618 Cricket AI Team, Last Modified: 2024

import pytest
import sys
import os
from unittest.mock import patch
from io import StringIO

# Add parent directory to path to import admin_tools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from admin_tools import AdminTools, admin_tools

class TestAdminTools:
    """Test suite for AdminTools class and its methods."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.admin_tools = AdminTools()
    
    def test_build_knowledge_graph_returns_correct_message(self):
        """Test that build_knowledge_graph returns the expected status message."""
        # Capture print output
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = self.admin_tools.build_knowledge_graph()
        
        assert isinstance(result, str)
        assert "Knowledge graph" in result
        assert "nodes" in result and "edges" in result
        assert "[LOG] Knowledge graph building started..." in mock_stdout.getvalue()
    
    def test_train_gnn_embeddings_returns_correct_message(self):
        """Test that train_gnn_embeddings returns the expected status message."""
        # Capture print output
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = self.admin_tools.train_gnn_embeddings()
        
        assert isinstance(result, str)
        assert "GNN training" in result
        assert "[LOG] GNN training started..." in mock_stdout.getvalue()
    
    def test_train_crickformer_model_returns_correct_message(self):
        """Test that train_crickformer_model returns the expected status message."""
        # Capture print output
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = self.admin_tools.train_crickformer_model()
        
        assert isinstance(result, str)
        assert "Crickformer" in result
        assert "[LOG] Crickformer training started..." in mock_stdout.getvalue()
    
    def test_run_evaluation_returns_correct_message(self):
        """Test that run_evaluation returns the expected status message."""
        # Capture print output
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            result = self.admin_tools.run_evaluation()
        
        assert isinstance(result, str)
        assert "Evaluation complete" in result
        assert "[LOG] Evaluation started..." in mock_stdout.getvalue()
    
    def test_get_system_status_returns_correct_structure(self):
        """Test that get_system_status returns the expected dictionary structure."""
        result = self.admin_tools.get_system_status()
        
        # Check that result is a dictionary
        assert isinstance(result, dict)
        
        # Check that all expected keys are present
        expected_keys = [
            'knowledge_graph_built',
            'gnn_embeddings_trained',
            'crickformer_trained',
            'last_evaluation',
            'system_health'
        ]
        
        for key in expected_keys:
            assert key in result
        
        # Check data types
        assert isinstance(result['knowledge_graph_built'], bool)
        assert isinstance(result['gnn_embeddings_trained'], bool)
        assert isinstance(result['crickformer_trained'], bool)
        assert isinstance(result['system_health'], str)
    
    def test_global_admin_tools_instance(self):
        """Test that the global admin_tools instance is properly initialized."""
        assert admin_tools is not None
        assert isinstance(admin_tools, AdminTools)
        assert hasattr(admin_tools, 'build_knowledge_graph')
        assert hasattr(admin_tools, 'train_gnn_embeddings')
        assert hasattr(admin_tools, 'train_crickformer_model')
        assert hasattr(admin_tools, 'run_evaluation')
        assert hasattr(admin_tools, 'get_system_status')
    
    def test_all_methods_have_docstrings(self):
        """Test that all public methods have proper docstrings."""
        methods = [
            'build_knowledge_graph',
            'train_gnn_embeddings',
            'train_crickformer_model',
            'run_evaluation',
            'get_system_status'
        ]
        
        for method_name in methods:
            method = getattr(self.admin_tools, method_name)
            assert method.__doc__ is not None
            assert len(method.__doc__.strip()) > 0
    
    def test_all_training_functions_complete_without_error(self):
        """Test that all training functions complete without raising exceptions."""
        # Test each function individually
        try:
            result1 = self.admin_tools.build_knowledge_graph()
            assert result1 is not None
            
            result2 = self.admin_tools.train_gnn_embeddings()
            assert result2 is not None
            
            result3 = self.admin_tools.train_crickformer_model()
            assert result3 is not None
            
            result4 = self.admin_tools.run_evaluation()
            assert result4 is not None
            
        except Exception as e:
            pytest.fail(f"Training function raised an exception: {e}")
    
    def test_system_health_is_healthy_by_default(self):
        """Test that system health is 'healthy' by default."""
        status = self.admin_tools.get_system_status()
        assert status['system_health'] == 'healthy'
    
    def test_boolean_flags_are_false_by_default(self):
        """Test that all boolean status flags are False by default."""
        status = self.admin_tools.get_system_status()
        assert status['knowledge_graph_built'] is False
        assert status['gnn_embeddings_trained'] is False
        assert status['crickformer_trained'] is False
    
    def test_last_evaluation_is_none_by_default(self):
        """Test that last_evaluation is None by default."""
        status = self.admin_tools.get_system_status()
        assert status['last_evaluation'] is None

# Integration tests for the complete workflow
class TestAdminToolsIntegration:
    """Integration tests for admin tools workflow."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.admin_tools = AdminTools()
    
    def test_complete_workflow_execution(self):
        """Test that the complete ML workflow can be executed in sequence."""
        # Execute all steps in typical order
        result1 = self.admin_tools.build_knowledge_graph()
        assert "Knowledge graph" in result1
        
        result2 = self.admin_tools.train_gnn_embeddings()
        assert "GNN training" in result2
        
        result3 = self.admin_tools.train_crickformer_model()
        assert "Crickformer" in result3
        
        result4 = self.admin_tools.run_evaluation()
        assert "Evaluation complete" in result4
        
        # Verify system status is still accessible
        status = self.admin_tools.get_system_status()
        assert status['system_health'] == 'healthy'
    
    def test_print_statements_are_called(self):
        """Test that print statements are called for each function."""
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            self.admin_tools.build_knowledge_graph()
            self.admin_tools.train_gnn_embeddings()
            self.admin_tools.train_crickformer_model()
            self.admin_tools.run_evaluation()
            
            output = mock_stdout.getvalue()
            assert "[LOG] Knowledge graph building started..." in output
            assert "[LOG] GNN training started..." in output
            assert "[LOG] Crickformer training started..." in output
            assert "[LOG] Evaluation started..." in output
    
    
if __name__ == '__main__':
    pytest.main([__file__, '-v']) 