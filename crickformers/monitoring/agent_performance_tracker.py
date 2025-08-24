# Purpose: Agent-specific performance tracking and optimization
# Author: WicketWise AI, Last Modified: 2024

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import logging


class PerformanceThreshold(Enum):
    """Performance threshold levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    agent_id: str
    timestamp: datetime
    execution_time_ms: float
    success: bool
    confidence_score: float
    memory_usage_mb: float
    cpu_usage_percent: float
    query_complexity: int  # 1-10 scale
    result_quality_score: float  # 0-1 scale
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'agent_id': self.agent_id,
            'timestamp': self.timestamp.isoformat(),
            'execution_time_ms': self.execution_time_ms,
            'success': self.success,
            'confidence_score': self.confidence_score,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'query_complexity': self.query_complexity,
            'result_quality_score': self.result_quality_score,
            'error_message': self.error_message
        }


@dataclass
class OptimizationRecommendation:
    """Agent optimization recommendation"""
    agent_id: str
    recommendation_type: str
    priority: str  # high, medium, low
    description: str
    expected_improvement: str
    implementation_effort: str  # low, medium, high
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert recommendation to dictionary"""
        return {
            'agent_id': self.agent_id,
            'recommendation_type': self.recommendation_type,
            'priority': self.priority,
            'description': self.description,
            'expected_improvement': self.expected_improvement,
            'implementation_effort': self.implementation_effort,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class AgentPerformanceStats:
    """Aggregated agent performance statistics"""
    agent_id: str
    total_executions: int
    success_rate: float
    avg_execution_time_ms: float
    avg_confidence_score: float
    avg_result_quality: float
    performance_threshold: PerformanceThreshold
    last_execution: Optional[datetime] = None
    recent_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary"""
        return {
            'agent_id': self.agent_id,
            'total_executions': self.total_executions,
            'success_rate': self.success_rate,
            'avg_execution_time_ms': self.avg_execution_time_ms,
            'avg_confidence_score': self.avg_confidence_score,
            'avg_result_quality': self.avg_result_quality,
            'performance_threshold': self.performance_threshold.value,
            'last_execution': self.last_execution.isoformat() if self.last_execution else None,
            'recent_errors': self.recent_errors
        }


class AgentPerformanceTracker:
    """Tracks and analyzes agent performance metrics"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Performance thresholds (configurable)
        self.execution_time_thresholds = {
            PerformanceThreshold.EXCELLENT: 500,    # < 500ms
            PerformanceThreshold.GOOD: 1000,        # < 1s
            PerformanceThreshold.ACCEPTABLE: 2000,  # < 2s
            PerformanceThreshold.POOR: 5000,        # < 5s
            # > 5s is CRITICAL
        }
        
        self.success_rate_thresholds = {
            PerformanceThreshold.EXCELLENT: 0.98,   # > 98%
            PerformanceThreshold.GOOD: 0.95,        # > 95%
            PerformanceThreshold.ACCEPTABLE: 0.90,  # > 90%
            PerformanceThreshold.POOR: 0.80,        # > 80%
            # < 80% is CRITICAL
        }
        
        # Optimization recommendations cache
        self.recommendations: Dict[str, List[OptimizationRecommendation]] = defaultdict(list)
        self.last_analysis: Dict[str, datetime] = {}
    
    def record_execution(self, metrics: AgentMetrics):
        """Record agent execution metrics"""
        with self.lock:
            self.metrics_history[metrics.agent_id].append(metrics)
        
        # Log performance issues
        if not metrics.success:
            self.logger.warning(f"Agent {metrics.agent_id} execution failed: {metrics.error_message}")
        elif metrics.execution_time_ms > 5000:
            self.logger.warning(f"Agent {metrics.agent_id} slow execution: {metrics.execution_time_ms:.1f}ms")
    
    def get_agent_stats(self, agent_id: str, time_window: Optional[timedelta] = None) -> Optional[AgentPerformanceStats]:
        """Get performance statistics for a specific agent"""
        with self.lock:
            if agent_id not in self.metrics_history:
                return None
            
            metrics = list(self.metrics_history[agent_id])
            
            # Filter by time window if specified
            if time_window:
                cutoff_time = datetime.now() - time_window
                metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            if not metrics:
                return None
            
            # Calculate statistics
            total_executions = len(metrics)
            successful_executions = [m for m in metrics if m.success]
            success_rate = len(successful_executions) / total_executions
            
            execution_times = [m.execution_time_ms for m in successful_executions]
            avg_execution_time = statistics.mean(execution_times) if execution_times else 0
            
            confidence_scores = [m.confidence_score for m in successful_executions]
            avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0
            
            quality_scores = [m.result_quality_score for m in successful_executions]
            avg_quality = statistics.mean(quality_scores) if quality_scores else 0
            
            # Determine performance threshold
            performance_threshold = self._calculate_performance_threshold(
                avg_execution_time, success_rate
            )
            
            # Get recent errors
            recent_errors = [
                m.error_message for m in metrics[-10:] 
                if not m.success and m.error_message
            ]
            
            return AgentPerformanceStats(
                agent_id=agent_id,
                total_executions=total_executions,
                success_rate=success_rate,
                avg_execution_time_ms=avg_execution_time,
                avg_confidence_score=avg_confidence,
                avg_result_quality=avg_quality,
                performance_threshold=performance_threshold,
                last_execution=metrics[-1].timestamp if metrics else None,
                recent_errors=recent_errors
            )
    
    def get_all_agent_stats(self, time_window: Optional[timedelta] = None) -> Dict[str, AgentPerformanceStats]:
        """Get performance statistics for all agents"""
        stats = {}
        for agent_id in self.metrics_history.keys():
            agent_stats = self.get_agent_stats(agent_id, time_window)
            if agent_stats:
                stats[agent_id] = agent_stats
        return stats
    
    def _calculate_performance_threshold(self, avg_execution_time: float, success_rate: float) -> PerformanceThreshold:
        """Calculate overall performance threshold based on metrics"""
        # Check success rate first (more critical)
        if success_rate < 0.80:
            return PerformanceThreshold.CRITICAL
        elif success_rate < self.success_rate_thresholds[PerformanceThreshold.POOR]:
            return PerformanceThreshold.POOR
        elif success_rate < self.success_rate_thresholds[PerformanceThreshold.ACCEPTABLE]:
            return PerformanceThreshold.ACCEPTABLE
        elif success_rate < self.success_rate_thresholds[PerformanceThreshold.GOOD]:
            return PerformanceThreshold.GOOD
        
        # Check execution time
        if avg_execution_time > 5000:
            return PerformanceThreshold.CRITICAL
        elif avg_execution_time > self.execution_time_thresholds[PerformanceThreshold.POOR]:
            return PerformanceThreshold.POOR
        elif avg_execution_time > self.execution_time_thresholds[PerformanceThreshold.ACCEPTABLE]:
            return PerformanceThreshold.ACCEPTABLE
        elif avg_execution_time > self.execution_time_thresholds[PerformanceThreshold.GOOD]:
            return PerformanceThreshold.GOOD
        else:
            return PerformanceThreshold.EXCELLENT
    
    def analyze_agent_performance(self, agent_id: str, force_analysis: bool = False) -> List[OptimizationRecommendation]:
        """Analyze agent performance and generate optimization recommendations"""
        # Check if analysis was done recently (unless forced)
        if not force_analysis and agent_id in self.last_analysis:
            time_since_analysis = datetime.now() - self.last_analysis[agent_id]
            if time_since_analysis < timedelta(hours=1):
                return self.recommendations.get(agent_id, [])
        
        stats = self.get_agent_stats(agent_id, time_window=timedelta(days=7))
        if not stats:
            return []
        
        recommendations = []
        
        # Analyze success rate
        if stats.success_rate < 0.95:
            priority = "high" if stats.success_rate < 0.90 else "medium"
            recommendations.append(OptimizationRecommendation(
                agent_id=agent_id,
                recommendation_type="reliability",
                priority=priority,
                description=f"Success rate is {stats.success_rate:.1%}. Investigate error patterns and improve error handling.",
                expected_improvement="10-20% improvement in success rate",
                implementation_effort="medium",
                timestamp=datetime.now()
            ))
        
        # Analyze execution time
        if stats.avg_execution_time_ms > 2000:
            priority = "high" if stats.avg_execution_time_ms > 5000 else "medium"
            recommendations.append(OptimizationRecommendation(
                agent_id=agent_id,
                recommendation_type="performance",
                priority=priority,
                description=f"Average execution time is {stats.avg_execution_time_ms:.0f}ms. Consider caching, optimization, or resource scaling.",
                expected_improvement="30-50% reduction in response time",
                implementation_effort="medium",
                timestamp=datetime.now()
            ))
        
        # Analyze confidence scores
        if stats.avg_confidence_score < 0.80:
            recommendations.append(OptimizationRecommendation(
                agent_id=agent_id,
                recommendation_type="accuracy",
                priority="medium",
                description=f"Average confidence score is {stats.avg_confidence_score:.2f}. Consider model retraining or feature engineering.",
                expected_improvement="15-25% improvement in confidence",
                implementation_effort="high",
                timestamp=datetime.now()
            ))
        
        # Analyze result quality
        if stats.avg_result_quality < 0.85:
            recommendations.append(OptimizationRecommendation(
                agent_id=agent_id,
                recommendation_type="quality",
                priority="medium",
                description=f"Average result quality is {stats.avg_result_quality:.2f}. Review output validation and quality metrics.",
                expected_improvement="10-20% improvement in result quality",
                implementation_effort="medium",
                timestamp=datetime.now()
            ))
        
        # Check for error patterns
        if len(stats.recent_errors) > 3:
            error_types = set(stats.recent_errors)
            if len(error_types) < len(stats.recent_errors):  # Repeated errors
                recommendations.append(OptimizationRecommendation(
                    agent_id=agent_id,
                    recommendation_type="error_handling",
                    priority="high",
                    description="Repeated error patterns detected. Implement specific error handling for common failure modes.",
                    expected_improvement="50-70% reduction in recurring errors",
                    implementation_effort="low",
                    timestamp=datetime.now()
                ))
        
        # Cache recommendations
        self.recommendations[agent_id] = recommendations
        self.last_analysis[agent_id] = datetime.now()
        
        return recommendations
    
    def get_performance_trends(self, agent_id: str, time_window: timedelta = timedelta(days=7)) -> Dict[str, List[float]]:
        """Get performance trends over time"""
        with self.lock:
            if agent_id not in self.metrics_history:
                return {}
            
            metrics = list(self.metrics_history[agent_id])
            cutoff_time = datetime.now() - time_window
            metrics = [m for m in metrics if m.timestamp >= cutoff_time]
            
            if not metrics:
                return {}
            
            # Group metrics by hour for trend analysis
            hourly_metrics = defaultdict(list)
            for metric in metrics:
                hour_key = metric.timestamp.replace(minute=0, second=0, microsecond=0)
                hourly_metrics[hour_key].append(metric)
            
            trends = {
                'timestamps': [],
                'execution_times': [],
                'success_rates': [],
                'confidence_scores': [],
                'quality_scores': []
            }
            
            for hour in sorted(hourly_metrics.keys()):
                hour_metrics = hourly_metrics[hour]
                
                trends['timestamps'].append(hour.isoformat())
                
                # Calculate hourly averages
                execution_times = [m.execution_time_ms for m in hour_metrics if m.success]
                trends['execution_times'].append(
                    statistics.mean(execution_times) if execution_times else 0
                )
                
                success_count = sum(1 for m in hour_metrics if m.success)
                trends['success_rates'].append(success_count / len(hour_metrics))
                
                confidence_scores = [m.confidence_score for m in hour_metrics if m.success]
                trends['confidence_scores'].append(
                    statistics.mean(confidence_scores) if confidence_scores else 0
                )
                
                quality_scores = [m.result_quality_score for m in hour_metrics if m.success]
                trends['quality_scores'].append(
                    statistics.mean(quality_scores) if quality_scores else 0
                )
            
            return trends
    
    def get_comparative_analysis(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get comparative analysis across all agents"""
        all_stats = self.get_all_agent_stats(time_window)
        
        if not all_stats:
            return {}
        
        # Calculate comparative metrics
        execution_times = [stats.avg_execution_time_ms for stats in all_stats.values()]
        success_rates = [stats.success_rate for stats in all_stats.values()]
        confidence_scores = [stats.avg_confidence_score for stats in all_stats.values()]
        
        # Find best and worst performers
        best_performer = min(all_stats.items(), key=lambda x: x[1].avg_execution_time_ms)
        worst_performer = max(all_stats.items(), key=lambda x: x[1].avg_execution_time_ms)
        
        most_reliable = max(all_stats.items(), key=lambda x: x[1].success_rate)
        least_reliable = min(all_stats.items(), key=lambda x: x[1].success_rate)
        
        return {
            'total_agents': len(all_stats),
            'avg_execution_time_ms': statistics.mean(execution_times),
            'avg_success_rate': statistics.mean(success_rates),
            'avg_confidence_score': statistics.mean(confidence_scores),
            'best_performer': {
                'agent_id': best_performer[0],
                'avg_execution_time_ms': best_performer[1].avg_execution_time_ms
            },
            'worst_performer': {
                'agent_id': worst_performer[0],
                'avg_execution_time_ms': worst_performer[1].avg_execution_time_ms
            },
            'most_reliable': {
                'agent_id': most_reliable[0],
                'success_rate': most_reliable[1].success_rate
            },
            'least_reliable': {
                'agent_id': least_reliable[0],
                'success_rate': least_reliable[1].success_rate
            },
            'performance_distribution': {
                threshold.value: sum(1 for stats in all_stats.values() 
                                   if stats.performance_threshold == threshold)
                for threshold in PerformanceThreshold
            }
        }
    
    def export_metrics(self, agent_id: Optional[str] = None, 
                      time_window: Optional[timedelta] = None) -> List[Dict[str, Any]]:
        """Export metrics data for external analysis"""
        with self.lock:
            if agent_id:
                agent_ids = [agent_id] if agent_id in self.metrics_history else []
            else:
                agent_ids = list(self.metrics_history.keys())
            
            exported_metrics = []
            
            for aid in agent_ids:
                metrics = list(self.metrics_history[aid])
                
                if time_window:
                    cutoff_time = datetime.now() - time_window
                    metrics = [m for m in metrics if m.timestamp >= cutoff_time]
                
                exported_metrics.extend([m.to_dict() for m in metrics])
            
            return exported_metrics
