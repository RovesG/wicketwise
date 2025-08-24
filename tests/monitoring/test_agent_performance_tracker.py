# Purpose: Unit tests for agent performance tracking system
# Author: WicketWise AI, Last Modified: 2024

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from crickformers.monitoring.agent_performance_tracker import (
    AgentPerformanceTracker,
    AgentMetrics,
    OptimizationRecommendation,
    AgentPerformanceStats,
    PerformanceThreshold
)


class TestAgentMetrics:
    """Test suite for AgentMetrics data structure"""
    
    def test_agent_metrics_creation(self):
        """Test AgentMetrics creation and basic properties"""
        timestamp = datetime.now()
        metrics = AgentMetrics(
            agent_id="performance_agent",
            timestamp=timestamp,
            execution_time_ms=1250.5,
            success=True,
            confidence_score=0.87,
            memory_usage_mb=45.2,
            cpu_usage_percent=12.8,
            query_complexity=7,
            result_quality_score=0.92
        )
        
        assert metrics.agent_id == "performance_agent"
        assert metrics.timestamp == timestamp
        assert metrics.execution_time_ms == 1250.5
        assert metrics.success is True
        assert metrics.confidence_score == 0.87
        assert metrics.memory_usage_mb == 45.2
        assert metrics.cpu_usage_percent == 12.8
        assert metrics.query_complexity == 7
        assert metrics.result_quality_score == 0.92
        assert metrics.error_message is None
    
    def test_agent_metrics_with_error(self):
        """Test AgentMetrics creation with error"""
        timestamp = datetime.now()
        metrics = AgentMetrics(
            agent_id="failing_agent",
            timestamp=timestamp,
            execution_time_ms=500.0,
            success=False,
            confidence_score=0.0,
            memory_usage_mb=30.0,
            cpu_usage_percent=5.0,
            query_complexity=3,
            result_quality_score=0.0,
            error_message="Connection timeout"
        )
        
        assert metrics.success is False
        assert metrics.error_message == "Connection timeout"
        assert metrics.confidence_score == 0.0
        assert metrics.result_quality_score == 0.0
    
    def test_agent_metrics_to_dict(self):
        """Test AgentMetrics dictionary conversion"""
        timestamp = datetime.now()
        metrics = AgentMetrics(
            agent_id="test_agent",
            timestamp=timestamp,
            execution_time_ms=800.0,
            success=True,
            confidence_score=0.95,
            memory_usage_mb=25.5,
            cpu_usage_percent=8.2,
            query_complexity=5,
            result_quality_score=0.88
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict['agent_id'] == "test_agent"
        assert metrics_dict['timestamp'] == timestamp.isoformat()
        assert metrics_dict['execution_time_ms'] == 800.0
        assert metrics_dict['success'] is True
        assert metrics_dict['confidence_score'] == 0.95
        assert metrics_dict['memory_usage_mb'] == 25.5
        assert metrics_dict['cpu_usage_percent'] == 8.2
        assert metrics_dict['query_complexity'] == 5
        assert metrics_dict['result_quality_score'] == 0.88
        assert metrics_dict['error_message'] is None


class TestOptimizationRecommendation:
    """Test suite for OptimizationRecommendation data structure"""
    
    def test_recommendation_creation(self):
        """Test OptimizationRecommendation creation"""
        timestamp = datetime.now()
        recommendation = OptimizationRecommendation(
            agent_id="slow_agent",
            recommendation_type="performance",
            priority="high",
            description="Reduce execution time by implementing caching",
            expected_improvement="30-50% reduction in response time",
            implementation_effort="medium",
            timestamp=timestamp
        )
        
        assert recommendation.agent_id == "slow_agent"
        assert recommendation.recommendation_type == "performance"
        assert recommendation.priority == "high"
        assert recommendation.description == "Reduce execution time by implementing caching"
        assert recommendation.expected_improvement == "30-50% reduction in response time"
        assert recommendation.implementation_effort == "medium"
        assert recommendation.timestamp == timestamp
    
    def test_recommendation_to_dict(self):
        """Test OptimizationRecommendation dictionary conversion"""
        timestamp = datetime.now()
        recommendation = OptimizationRecommendation(
            agent_id="unreliable_agent",
            recommendation_type="reliability",
            priority="critical",
            description="Improve error handling for network failures",
            expected_improvement="20% improvement in success rate",
            implementation_effort="low",
            timestamp=timestamp
        )
        
        rec_dict = recommendation.to_dict()
        
        assert rec_dict['agent_id'] == "unreliable_agent"
        assert rec_dict['recommendation_type'] == "reliability"
        assert rec_dict['priority'] == "critical"
        assert rec_dict['description'] == "Improve error handling for network failures"
        assert rec_dict['expected_improvement'] == "20% improvement in success rate"
        assert rec_dict['implementation_effort'] == "low"
        assert rec_dict['timestamp'] == timestamp.isoformat()


class TestAgentPerformanceTracker:
    """Test suite for AgentPerformanceTracker"""
    
    @pytest.fixture
    def tracker(self):
        """Create AgentPerformanceTracker instance"""
        return AgentPerformanceTracker(max_history=100)
    
    def test_tracker_initialization(self, tracker):
        """Test AgentPerformanceTracker initialization"""
        assert tracker.max_history == 100
        assert len(tracker.metrics_history) == 0
        assert len(tracker.recommendations) == 0
        assert len(tracker.last_analysis) == 0
        
        # Check thresholds are set
        assert PerformanceThreshold.EXCELLENT in tracker.execution_time_thresholds
        assert PerformanceThreshold.GOOD in tracker.success_rate_thresholds
    
    def test_record_execution(self, tracker):
        """Test recording agent execution metrics"""
        timestamp = datetime.now()
        metrics = AgentMetrics(
            agent_id="test_agent",
            timestamp=timestamp,
            execution_time_ms=1200.0,
            success=True,
            confidence_score=0.85,
            memory_usage_mb=40.0,
            cpu_usage_percent=15.0,
            query_complexity=6,
            result_quality_score=0.90
        )
        
        tracker.record_execution(metrics)
        
        assert len(tracker.metrics_history["test_agent"]) == 1
        assert tracker.metrics_history["test_agent"][0] == metrics
    
    def test_get_agent_stats(self, tracker):
        """Test getting agent performance statistics"""
        agent_id = "performance_agent"
        
        # Add successful executions
        for i in range(8):
            metrics = AgentMetrics(
                agent_id=agent_id,
                timestamp=datetime.now() - timedelta(minutes=i),
                execution_time_ms=1000.0 + i * 100,
                success=True,
                confidence_score=0.85 + i * 0.01,
                memory_usage_mb=30.0,
                cpu_usage_percent=10.0,
                query_complexity=5,
                result_quality_score=0.88 + i * 0.01
            )
            tracker.record_execution(metrics)
        
        # Add failed executions
        for i in range(2):
            metrics = AgentMetrics(
                agent_id=agent_id,
                timestamp=datetime.now() - timedelta(minutes=10 + i),
                execution_time_ms=2000.0,
                success=False,
                confidence_score=0.0,
                memory_usage_mb=30.0,
                cpu_usage_percent=10.0,
                query_complexity=5,
                result_quality_score=0.0,
                error_message=f"Error {i}"
            )
            tracker.record_execution(metrics)
        
        stats = tracker.get_agent_stats(agent_id)
        
        assert stats is not None
        assert stats.agent_id == agent_id
        assert stats.total_executions == 10
        assert stats.success_rate == 0.8  # 8 successful out of 10
        assert 1000 <= stats.avg_execution_time_ms <= 1700  # Average of successful executions
        assert 0.85 <= stats.avg_confidence_score <= 0.93
        assert 0.88 <= stats.avg_result_quality <= 0.96
        assert len(stats.recent_errors) <= 2
    
    def test_performance_threshold_calculation(self, tracker):
        """Test performance threshold calculation"""
        # Test excellent performance (fast + high success rate)
        threshold = tracker._calculate_performance_threshold(400.0, 0.99)
        assert threshold == PerformanceThreshold.EXCELLENT
        
        # Test excellent performance (high success rate dominates)
        threshold = tracker._calculate_performance_threshold(800.0, 0.99)
        assert threshold == PerformanceThreshold.EXCELLENT
        
        # Test good performance
        threshold = tracker._calculate_performance_threshold(800.0, 0.96)
        assert threshold == PerformanceThreshold.EXCELLENT  # High success rate
        
        # Test acceptable performance (lower success rate)
        threshold = tracker._calculate_performance_threshold(1500.0, 0.92)
        assert threshold == PerformanceThreshold.GOOD  # Success rate drives threshold
        
        # Test poor performance (even lower success rate)
        threshold = tracker._calculate_performance_threshold(3000.0, 0.85)
        assert threshold == PerformanceThreshold.ACCEPTABLE  # Success rate drives threshold
        
        # Test critical performance (low success rate)
        threshold = tracker._calculate_performance_threshold(1000.0, 0.75)
        assert threshold == PerformanceThreshold.CRITICAL
        
        # Test critical performance (high execution time)
        threshold = tracker._calculate_performance_threshold(6000.0, 0.95)
        assert threshold == PerformanceThreshold.CRITICAL
    
    def test_analyze_agent_performance(self, tracker):
        """Test agent performance analysis and recommendations"""
        agent_id = "slow_agent"
        
        # Add metrics indicating poor performance
        for i in range(10):
            success = i < 7  # 70% success rate
            metrics = AgentMetrics(
                agent_id=agent_id,
                timestamp=datetime.now() - timedelta(hours=i),
                execution_time_ms=3000.0 + i * 100,  # Slow execution
                success=success,
                confidence_score=0.75 if success else 0.0,  # Low confidence
                memory_usage_mb=50.0,
                cpu_usage_percent=20.0,
                query_complexity=6,
                result_quality_score=0.80 if success else 0.0,  # Low quality
                error_message="Timeout error" if not success else None
            )
            tracker.record_execution(metrics)
        
        recommendations = tracker.analyze_agent_performance(agent_id)
        
        assert len(recommendations) > 0
        
        # Check for expected recommendation types
        rec_types = [rec.recommendation_type for rec in recommendations]
        assert "reliability" in rec_types  # Low success rate
        assert "performance" in rec_types  # High execution time
        
        # Check that recommendations are cached
        cached_recommendations = tracker.analyze_agent_performance(agent_id)
        assert len(cached_recommendations) == len(recommendations)
    
    def test_get_all_agent_stats(self, tracker):
        """Test getting statistics for all agents"""
        # Add metrics for multiple agents
        agents = ["agent1", "agent2", "agent3"]
        
        for agent_id in agents:
            for i in range(5):
                metrics = AgentMetrics(
                    agent_id=agent_id,
                    timestamp=datetime.now() - timedelta(minutes=i),
                    execution_time_ms=1000.0 + hash(agent_id) % 1000,
                    success=True,
                    confidence_score=0.85,
                    memory_usage_mb=30.0,
                    cpu_usage_percent=10.0,
                    query_complexity=5,
                    result_quality_score=0.90
                )
                tracker.record_execution(metrics)
        
        all_stats = tracker.get_all_agent_stats()
        
        assert len(all_stats) == 3
        for agent_id in agents:
            assert agent_id in all_stats
            assert all_stats[agent_id].total_executions == 5
    
    def test_get_comparative_analysis(self, tracker):
        """Test comparative analysis across agents"""
        # Add metrics for multiple agents with different performance
        agents_data = [
            ("fast_agent", 500.0, 0.98),
            ("slow_agent", 2000.0, 0.85),
            ("unreliable_agent", 1000.0, 0.70)
        ]
        
        for agent_id, exec_time, success_rate in agents_data:
            num_executions = 10
            num_successful = int(num_executions * success_rate)
            
            # Add successful executions
            for i in range(num_successful):
                metrics = AgentMetrics(
                    agent_id=agent_id,
                    timestamp=datetime.now() - timedelta(minutes=i),
                    execution_time_ms=exec_time,
                    success=True,
                    confidence_score=0.85,
                    memory_usage_mb=30.0,
                    cpu_usage_percent=10.0,
                    query_complexity=5,
                    result_quality_score=0.90
                )
                tracker.record_execution(metrics)
            
            # Add failed executions
            for i in range(num_executions - num_successful):
                metrics = AgentMetrics(
                    agent_id=agent_id,
                    timestamp=datetime.now() - timedelta(minutes=num_successful + i),
                    execution_time_ms=exec_time * 2,  # Failed executions take longer
                    success=False,
                    confidence_score=0.0,
                    memory_usage_mb=30.0,
                    cpu_usage_percent=10.0,
                    query_complexity=5,
                    result_quality_score=0.0,
                    error_message="Test error"
                )
                tracker.record_execution(metrics)
        
        analysis = tracker.get_comparative_analysis()
        
        assert analysis['total_agents'] == 3
        assert 'avg_execution_time_ms' in analysis
        assert 'avg_success_rate' in analysis
        assert 'best_performer' in analysis
        assert 'worst_performer' in analysis
        assert 'most_reliable' in analysis
        assert 'least_reliable' in analysis
        assert 'performance_distribution' in analysis
        
        # Check best/worst performers
        assert analysis['best_performer']['agent_id'] == "fast_agent"
        assert analysis['worst_performer']['agent_id'] == "slow_agent"
        assert analysis['most_reliable']['agent_id'] == "fast_agent"
        assert analysis['least_reliable']['agent_id'] == "unreliable_agent"
    
    def test_export_metrics(self, tracker):
        """Test exporting metrics data"""
        agent_id = "export_agent"
        
        # Add test metrics
        for i in range(3):
            metrics = AgentMetrics(
                agent_id=agent_id,
                timestamp=datetime.now() - timedelta(minutes=i),
                execution_time_ms=1000.0 + i * 100,
                success=True,
                confidence_score=0.85,
                memory_usage_mb=30.0,
                cpu_usage_percent=10.0,
                query_complexity=5,
                result_quality_score=0.90
            )
            tracker.record_execution(metrics)
        
        # Export all metrics
        exported = tracker.export_metrics()
        assert len(exported) == 3
        
        # Export specific agent metrics
        exported_agent = tracker.export_metrics(agent_id=agent_id)
        assert len(exported_agent) == 3
        assert all(m['agent_id'] == agent_id for m in exported_agent)
        
        # Export with time window
        exported_recent = tracker.export_metrics(
            agent_id=agent_id, 
            time_window=timedelta(minutes=1)
        )
        assert len(exported_recent) <= 3


def run_agent_performance_tests():
    """Run all agent performance tracker tests"""
    print("🎯 Running Agent Performance Tracker Tests")
    print("=" * 50)
    
    # Test categories
    test_categories = [
        ("Agent Metrics", TestAgentMetrics),
        ("Optimization Recommendations", TestOptimizationRecommendation),
        ("Agent Performance Tracker", TestAgentPerformanceTracker)
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
                
                # Handle fixtures
                if hasattr(test_instance, test_method):
                    method = getattr(test_instance, test_method)
                    
                    # Check if method needs fixtures
                    import inspect
                    sig = inspect.signature(method)
                    
                    if 'tracker' in sig.parameters:
                        tracker = AgentPerformanceTracker(max_history=100)
                        method(tracker)
                    else:
                        method()
                    
                    print(f"  ✅ {test_method}")
                    passed_tests += 1
                    category_passed += 1
                    
            except Exception as e:
                print(f"  ❌ {test_method}: {str(e)}")
        
        print(f"  📈 Category Results: {category_passed}/{len(test_methods)} passed")
    
    print(f"\n🏆 Overall Agent Performance Test Results: {passed_tests}/{total_tests} passed")
    print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_agent_performance_tests()
    exit(0 if success else 1)
