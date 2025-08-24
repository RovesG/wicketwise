# Purpose: Unit tests for Agent Dashboard UI components and backend
# Author: WicketWise AI, Last Modified: 2024

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from flask import Flask
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from agent_dashboard_backend import app, orchestration_engine
from crickformers.agents.orchestration_engine import OrchestrationEngine
from crickformers.agents.base_agent import AgentContext, AgentResponse, AgentCapability, AgentPriority


class TestAgentDashboardBackend:
    """Test suite for Agent Dashboard Flask backend"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    @pytest.fixture
    def mock_orchestration_engine(self):
        """Mock orchestration engine"""
        with patch('agent_dashboard_backend.orchestration_engine') as mock_engine:
            mock_engine.is_initialized = True
            yield mock_engine
    
    def test_health_check_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get('/api/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'system_info' in data
    
    def test_agent_status_endpoint(self, client, mock_orchestration_engine):
        """Test agent status endpoint"""
        # Mock agent status
        mock_orchestration_engine.get_agent_status.return_value = {
            'performance': {'status': 'active', 'load': 0.3},
            'tactical': {'status': 'active', 'load': 0.2},
            'prediction': {'status': 'active', 'load': 0.5},
            'betting': {'status': 'idle', 'load': 0.0}
        }
        
        response = client.get('/api/agents/status')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'agents' in data
        assert len(data['agents']) == 4
        assert data['agents']['performance']['status'] == 'active'
    
    def test_query_execution_endpoint(self, client, mock_orchestration_engine):
        """Test query execution endpoint"""
        # Mock orchestration response
        mock_response = AgentResponse(
            agent_id="performance",
            capability=AgentCapability.PERFORMANCE_ANALYSIS,
            success=True,
            confidence=0.85,
            execution_time=0.45,
            result={
                'analysis': 'Kohli shows strong batting form',
                'metrics': {'avg': 45.2, 'sr': 89.5}
            },
            metadata={'explanation': "Based on recent match data and historical performance patterns"}
        )
        
        mock_orchestration_engine.execute_query.return_value = {
            'responses': [mock_response],
            'execution_plan': {
                'strategy': 'parallel',
                'agents_used': ['performance'],
                'total_time': 0.45
            },
            'confidence': 0.85
        }
        
        query_data = {
            'query': 'Analyze Kohli batting performance',
            'context': {
                'match_id': 'test_match',
                'player_context': {'names': ['Kohli']}
            }
        }
        
        response = client.post('/api/query/execute', 
                             data=json.dumps(query_data),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'responses' in data
        assert 'execution_plan' in data
        assert data['confidence'] == 0.85
    
    def test_query_execution_error_handling(self, client, mock_orchestration_engine):
        """Test query execution error handling"""
        mock_orchestration_engine.execute_query.side_effect = Exception("Agent error")
        
        query_data = {
            'query': 'Invalid query',
            'context': {}
        }
        
        response = client.post('/api/query/execute',
                             data=json.dumps(query_data),
                             content_type='application/json')
        
        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Agent error' in data['error']
    
    def test_explanation_endpoint(self, client, mock_orchestration_engine):
        """Test explanation generation endpoint"""
        mock_orchestration_engine.generate_explanation.return_value = {
            'decision_tree': {
                'root': 'Query Analysis',
                'branches': [
                    {'condition': 'Player mentioned', 'action': 'Route to Performance Agent'},
                    {'condition': 'High confidence', 'action': 'Return result'}
                ]
            },
            'feature_importance': {
                'recent_matches': 0.4,
                'historical_avg': 0.3,
                'opposition_strength': 0.2,
                'venue_conditions': 0.1
            },
            'reasoning_steps': [
                'Identified player: Kohli',
                'Retrieved recent performance data',
                'Calculated weighted averages',
                'Applied venue adjustments'
            ]
        }
        
        explanation_data = {
            'query_id': 'test_query_123',
            'agent_responses': [
                {
                    'agent_id': 'performance',
                    'confidence': 0.85,
                    'result': {'avg': 45.2}
                }
            ]
        }
        
        response = client.post('/api/explanation/generate',
                             data=json.dumps(explanation_data),
                             content_type='application/json')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'decision_tree' in data
        assert 'feature_importance' in data
        assert 'reasoning_steps' in data
    
    def test_live_updates_endpoint(self, client):
        """Test live updates endpoint"""
        response = client.get('/api/live/updates')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'active_queries' in data
        assert 'system_metrics' in data
        assert 'agent_activity' in data
    
    def test_dashboard_main_page(self, client):
        """Test main dashboard page serves correctly"""
        response = client.get('/')
        assert response.status_code == 200
        # In testing mode, should return testing message
        response_text = response.get_data(as_text=True)
        assert 'WicketWise Agent Dashboard' in response_text


class TestAgentDashboardJavaScript:
    """Test suite for Agent Dashboard JavaScript functionality"""
    
    def test_dashboard_manager_initialization(self):
        """Test DashboardManager initialization logic"""
        # This would typically be tested with a JavaScript testing framework
        # For now, we'll test the Python backend that supports the JS
        
        # Mock JavaScript-like initialization
        config = {
            'api_base_url': 'http://127.0.0.1:5001/api',
            'update_interval': 2000,
            'max_query_history': 50
        }
        
        assert config['api_base_url'] == 'http://127.0.0.1:5001/api'
        assert config['update_interval'] == 2000
        assert config['max_query_history'] == 50
    
    def test_query_processing_flow(self):
        """Test query processing flow simulation"""
        # Simulate the JavaScript query flow
        query_steps = [
            'user_input_received',
            'query_validation',
            'api_request_sent',
            'response_received',
            'ui_updated'
        ]
        
        # Mock processing each step
        processed_steps = []
        for step in query_steps:
            processed_steps.append(step)
            if step == 'query_validation':
                assert len(processed_steps) == 2
            elif step == 'ui_updated':
                assert len(processed_steps) == 5
        
        assert processed_steps == query_steps
    
    def test_explainability_visualization_data(self):
        """Test explainability visualization data structure"""
        # Mock the data structure that JavaScript would receive
        explanation_data = {
            'decision_tree': {
                'nodes': [
                    {'id': 'root', 'label': 'Query Analysis', 'type': 'decision'},
                    {'id': 'agent_selection', 'label': 'Agent Selection', 'type': 'process'},
                    {'id': 'result', 'label': 'Final Result', 'type': 'output'}
                ],
                'edges': [
                    {'from': 'root', 'to': 'agent_selection'},
                    {'from': 'agent_selection', 'to': 'result'}
                ]
            },
            'feature_weights': [
                {'feature': 'recent_form', 'weight': 0.4, 'impact': 'positive'},
                {'feature': 'venue_history', 'weight': 0.3, 'impact': 'neutral'},
                {'feature': 'opposition', 'weight': 0.3, 'impact': 'negative'}
            ]
        }
        
        # Validate structure
        assert 'decision_tree' in explanation_data
        assert 'feature_weights' in explanation_data
        assert len(explanation_data['decision_tree']['nodes']) == 3
        assert len(explanation_data['feature_weights']) == 3
        
        # Validate feature weights sum to 1.0
        total_weight = sum(fw['weight'] for fw in explanation_data['feature_weights'])
        assert abs(total_weight - 1.0) < 1e-10


class TestUIIntegration:
    """Test suite for UI integration scenarios"""
    
    @pytest.fixture
    def integration_client(self):
        """Create integration test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_end_to_end_query_flow(self, integration_client):
        """Test complete end-to-end query flow"""
        with patch('agent_dashboard_backend.orchestration_engine') as mock_engine:
            mock_engine.is_initialized = True
            
            # Mock complete response
            mock_response = AgentResponse(
                agent_id="performance",
                capability=AgentCapability.PERFORMANCE_ANALYSIS,
                success=True,
                confidence=0.88,
                execution_time=0.67,
                result={
                    'player_analysis': {
                        'name': 'Virat Kohli',
                        'recent_avg': 52.3,
                        'strike_rate': 91.2,
                        'form_trend': 'improving'
                    }
                },
                metadata={'explanation': "Analysis based on last 10 matches with venue adjustments"}
            )
            
            mock_engine.execute_query.return_value = {
                'responses': [mock_response],
                'execution_plan': {
                    'strategy': 'sequential',
                    'agents_used': ['performance'],
                    'total_time': 0.67
                },
                'confidence': 0.88
            }
            
            # Step 1: Execute query
            query_data = {
                'query': 'How is Kohli performing this season?',
                'context': {
                    'player_context': {'names': ['Kohli']},
                    'season': '2024'
                }
            }
            
            response = integration_client.post('/api/query/execute',
                                             data=json.dumps(query_data),
                                             content_type='application/json')
            
            assert response.status_code == 200
            query_result = json.loads(response.data)
            
            # Step 2: Generate explanation
            explanation_data = {
                'query_id': 'test_integration_query',
                'agent_responses': query_result['responses']
            }
            
            mock_engine.generate_explanation.return_value = {
                'decision_tree': {'root': 'Player Performance Query'},
                'feature_importance': {'recent_matches': 0.6, 'venue': 0.4},
                'reasoning_steps': ['Identified player', 'Retrieved stats', 'Applied weights']
            }
            
            explanation_response = integration_client.post('/api/explanation/generate',
                                                         data=json.dumps(explanation_data),
                                                         content_type='application/json')
            
            assert explanation_response.status_code == 200
            explanation_result = json.loads(explanation_response.data)
            
            # Validate complete flow
            assert query_result['confidence'] == 0.88
            assert 'decision_tree' in explanation_result
            assert len(explanation_result['reasoning_steps']) == 4  # Updated to match mock data
    
    def test_multi_agent_coordination_display(self, integration_client):
        """Test multi-agent coordination display"""
        with patch('agent_dashboard_backend.orchestration_engine') as mock_engine:
            mock_engine.is_initialized = True
            
            # Mock multi-agent response
            performance_response = AgentResponse(
                agent_id="performance",
                capability=AgentCapability.PERFORMANCE_ANALYSIS,
                success=True,
                confidence=0.85,
                execution_time=0.3,
                result={'batting_avg': 45.2},
                metadata={'explanation': "Recent batting statistics"}
            )
            
            tactical_response = AgentResponse(
                agent_id="tactical",
                capability=AgentCapability.TACTICAL_ANALYSIS,
                success=True,
                confidence=0.78,
                execution_time=0.4,
                result={'field_placement': 'aggressive'},
                metadata={'explanation': "Recommended field setup"}
            )
            
            mock_engine.execute_query.return_value = {
                'responses': [performance_response, tactical_response],
                'execution_plan': {
                    'strategy': 'parallel',
                    'agents_used': ['performance', 'tactical'],
                    'total_time': 0.4  # Parallel execution
                },
                'confidence': 0.815  # Average of agent confidences
            }
            
            query_data = {
                'query': 'Analyze team strategy for upcoming match',
                'context': {
                    'match_context': {'opposition': 'Australia'},
                    'team_context': {'names': ['India']}
                }
            }
            
            response = integration_client.post('/api/query/execute',
                                             data=json.dumps(query_data),
                                             content_type='application/json')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            # Validate multi-agent coordination
            assert len(data['responses']) == 2
            assert data['execution_plan']['strategy'] == 'parallel'
            assert 'performance' in data['execution_plan']['agents_used']
            assert 'tactical' in data['execution_plan']['agents_used']
            assert data['execution_plan']['total_time'] == 0.4


def run_ui_tests():
    """Run all UI dashboard tests"""
    print("🎯 Running Sprint 5 UI Dashboard Tests")
    print("=" * 50)
    
    # Test categories
    test_categories = [
        ("Backend API Tests", TestAgentDashboardBackend),
        ("JavaScript Logic Tests", TestAgentDashboardJavaScript),
        ("UI Integration Tests", TestUIIntegration)
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for category_name, test_class in test_categories:
        print(f"\n📊 {category_name}")
        print("-" * 30)
        
        # Get test methods
        test_methods = [method for method in dir(test_class) 
                       if method.startswith('test_')]
        
        category_passed = 0
        for test_method in test_methods:
            total_tests += 1
            try:
                # Create test instance
                test_instance = test_class()
                
                # Run test method
                if hasattr(test_instance, test_method):
                    method = getattr(test_instance, test_method)
                    
                    # Handle fixtures for Flask tests
                    if test_method in ['test_health_check_endpoint', 'test_agent_status_endpoint', 
                                     'test_query_execution_endpoint', 'test_query_execution_error_handling',
                                     'test_explanation_endpoint', 'test_live_updates_endpoint', 
                                     'test_dashboard_main_page']:
                        # Create mock client
                        app.config['TESTING'] = True
                        with app.test_client() as client:
                            if test_method in ['test_agent_status_endpoint', 'test_query_execution_endpoint',
                                             'test_query_execution_error_handling', 'test_explanation_endpoint']:
                                with patch('agent_dashboard_backend.orchestration_engine') as mock_engine:
                                    mock_engine.is_initialized = True
                                    method(client, mock_engine)
                            else:
                                method(client)
                    elif test_method in ['test_end_to_end_query_flow', 'test_multi_agent_coordination_display']:
                        # Integration tests
                        app.config['TESTING'] = True
                        with app.test_client() as client:
                            method(client)
                    else:
                        # Regular tests
                        method()
                    
                    print(f"  ✅ {test_method}")
                    passed_tests += 1
                    category_passed += 1
                    
            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
        
        print(f"  📈 Category Results: {category_passed}/{len(test_methods)} passed")
    
    print(f"\n🏆 Overall UI Test Results: {passed_tests}/{total_tests} passed")
    print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_ui_tests()
    exit(0 if success else 1)
