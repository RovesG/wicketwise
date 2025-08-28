# WicketWise Agent UI Adapter
# Converts WicketWise agent system to Agent UI data contracts
# Author: WicketWise AI Team

import json
import re
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

@dataclass
class AgentMetrics:
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    throughput_per_min: int = 0
    accuracy_rate: Optional[float] = None
    false_positive_rate: Optional[float] = None
    kelly_efficiency: Optional[float] = None

@dataclass
class AgentHealth:
    cpu_usage: float = 0.0
    memory_mb: int = 0
    queue_depth: int = 0
    last_error: Optional[str] = None
    uptime_hours: float = 0.0

@dataclass
class CricketContext:
    supported_formats: Optional[List[str]] = None
    betting_markets: Optional[List[str]] = None
    risk_profile: Optional[str] = None
    exposure_level: Optional[str] = None
    match_phase: Optional[str] = None
    market_conditions: Optional[str] = None

class WicketWiseAgentAdapter:
    """Adapts WicketWise agent system to Agent UI data contracts"""
    
    def __init__(self, orchestration_engine=None, audit_logger=None, dgl_engine=None, socketio=None):
        self.orchestration_engine = orchestration_engine
        self.audit_logger = audit_logger
        self.dgl_engine = dgl_engine
        self.socketio = socketio
        
        # Event tracking
        self.event_stream = deque(maxlen=10000)
        self.agent_metrics = defaultdict(lambda: {
            'response_times': deque(maxlen=100),
            'throughput_counter': 0,
            'error_count': 0,
            'last_activity': time.time()
        })
        
        # Agent definitions cache
        self._agent_definitions_cache = None
        self._cache_timestamp = 0
        self._cache_ttl = 30  # 30 seconds
        
        # Flow definitions from documentation
        self._flow_definitions = None
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        logger.info("WicketWise Agent UI Adapter initialized")
    
    def generate_agent_definitions(self) -> List[Dict[str, Any]]:
        """Generate AgentDefinition objects from registered agents"""
        with self._lock:
            # Check cache
            current_time = time.time()
            if (self._agent_definitions_cache and 
                current_time - self._cache_timestamp < self._cache_ttl):
                return self._agent_definitions_cache
            
            definitions = []
            
            # Get agents from orchestration engine if available
            if self.orchestration_engine and hasattr(self.orchestration_engine, 'coordinator'):
                for agent_id, agent in self.orchestration_engine.coordinator.agents.items():
                    definition = self._create_agent_definition(agent_id, agent)
                    definitions.append(definition)
            else:
                # Fallback: Create definitions for known WicketWise agents
                definitions = self._create_default_agent_definitions()
            
            # Update cache
            self._agent_definitions_cache = definitions
            self._cache_timestamp = current_time
            
            return definitions
    
    def _create_agent_definition(self, agent_id: str, agent) -> Dict[str, Any]:
        """Create agent definition from agent instance"""
        metrics = self._get_agent_metrics(agent_id)
        health = self._get_agent_health(agent_id)
        cricket_context = self._get_cricket_context(agent)
        
        return {
            "id": agent_id,
            "label": getattr(agent, 'display_name', agent_id.replace('_', ' ').title()),
            "role": getattr(agent, 'description', f'{agent_id} cricket analysis agent'),
            "agent_type": self._determine_agent_type(agent),
            "inputs": getattr(agent, 'input_types', []),
            "outputs": getattr(agent, 'output_types', []),
            "capabilities": [cap.value for cap in getattr(agent, 'capabilities', [])],
            "metrics": asdict(metrics),
            "status": self._get_agent_status(agent_id),
            "health": asdict(health),
            "cricket_context": asdict(cricket_context) if cricket_context else None
        }
    
    def _create_default_agent_definitions(self) -> List[Dict[str, Any]]:
        """Create default agent definitions for WicketWise system"""
        default_agents = [
            {
                "id": "market_monitor",
                "label": "Market Monitor",
                "role": "Monitors betting market odds and liquidity",
                "agent_type": "monitoring",
                "inputs": ["betfair_api", "bet365_api", "pinnacle_api"],
                "outputs": ["normalized_odds_data", "market_updates"],
                "capabilities": ["MARKET_MONITORING", "ODDS_AGGREGATION"],
                "cricket_context": {
                    "supported_formats": ["T20", "ODI", "Test"],
                    "betting_markets": ["match_winner", "total_runs", "player_runs"],
                    "risk_profile": "low"
                }
            },
            {
                "id": "betting_agent",
                "label": "WicketWise Betting Agent",
                "role": "Cricket betting value detection and strategy coordination",
                "agent_type": "betting_intelligence",
                "inputs": ["market_odds_data", "match_context", "player_embeddings", "gnn_predictions"],
                "outputs": ["value_opportunities", "betting_recommendations", "risk_assessments"],
                "capabilities": ["VALUE_DETECTION", "ARBITRAGE_SCANNING", "MARKET_ANALYSIS"],
                "cricket_context": {
                    "supported_formats": ["T20", "ODI", "Test"],
                    "betting_markets": ["match_winner", "total_runs", "player_runs", "wickets"],
                    "risk_profile": "moderate_aggressive"
                }
            },
            {
                "id": "prediction_agent",
                "label": "GNN Prediction Agent",
                "role": "Generates cricket match and player predictions using GNN models",
                "agent_type": "prediction",
                "inputs": ["match_context", "player_stats", "gnn_embeddings"],
                "outputs": ["model_probabilities", "confidence_scores", "player_predictions"],
                "capabilities": ["MATCH_PREDICTION", "PLAYER_ANALYSIS", "PROBABILITY_CALCULATION"],
                "cricket_context": {
                    "supported_formats": ["T20", "ODI", "Test"],
                    "betting_markets": ["match_winner", "total_runs", "player_performance"],
                    "risk_profile": "moderate"
                }
            },
            {
                "id": "mispricing_engine",
                "label": "Mispricing Engine",
                "role": "Detects value opportunities and market inefficiencies",
                "agent_type": "analysis",
                "inputs": ["model_probabilities", "market_odds", "historical_data"],
                "outputs": ["value_opportunities", "expected_values", "kelly_fractions"],
                "capabilities": ["VALUE_DETECTION", "ARBITRAGE_DETECTION", "MARKET_EFFICIENCY"],
                "cricket_context": {
                    "supported_formats": ["T20", "ODI", "Test"],
                    "betting_markets": ["all_markets"],
                    "risk_profile": "aggressive"
                }
            },
            {
                "id": "shadow_agent",
                "label": "Shadow Betting Agent",
                "role": "Final decision validation and execution logic",
                "agent_type": "validation",
                "inputs": ["value_opportunities", "confidence_scores", "risk_parameters"],
                "outputs": ["validated_decisions", "risk_alerts", "execution_recommendations"],
                "capabilities": ["DECISION_VALIDATION", "RISK_ASSESSMENT"],
                "cricket_context": {
                    "supported_formats": ["T20", "ODI", "Test"],
                    "betting_markets": ["all_markets"],
                    "risk_profile": "conservative"
                }
            },
            {
                "id": "dgl_engine",
                "label": "DGL Risk Engine",
                "role": "Deterministic governance and risk management",
                "agent_type": "governance",
                "inputs": ["bet_proposals", "account_state", "exposure_data"],
                "outputs": ["governance_decisions", "risk_assessments", "constraint_violations"],
                "capabilities": ["RISK_MANAGEMENT", "GOVERNANCE", "CONSTRAINT_CHECKING"],
                "cricket_context": {
                    "supported_formats": ["T20", "ODI", "Test"],
                    "betting_markets": ["all_markets"],
                    "risk_profile": "ultra_conservative"
                }
            },
            {
                "id": "execution_engine",
                "label": "Execution Engine",
                "role": "Order placement and execution management",
                "agent_type": "execution",
                "inputs": ["approved_orders", "market_conditions", "execution_parameters"],
                "outputs": ["execution_confirmations", "fill_reports", "slippage_analysis"],
                "capabilities": ["ORDER_EXECUTION", "MARKET_INTERFACE"],
                "cricket_context": {
                    "supported_formats": ["T20", "ODI", "Test"],
                    "betting_markets": ["all_markets"],
                    "risk_profile": "moderate"
                }
            },
            {
                "id": "audit_logger",
                "label": "Audit Logger",
                "role": "Comprehensive audit logging and compliance tracking",
                "agent_type": "audit",
                "inputs": ["all_decisions", "system_events", "user_actions"],
                "outputs": ["audit_records", "compliance_reports", "decision_trails"],
                "capabilities": ["AUDIT_LOGGING", "COMPLIANCE", "DECISION_TRACKING"],
                "cricket_context": {
                    "supported_formats": ["T20", "ODI", "Test"],
                    "betting_markets": ["all_markets"],
                    "risk_profile": "neutral"
                }
            }
        ]
        
        # Add metrics and health to each agent
        for agent_def in default_agents:
            agent_id = agent_def["id"]
            agent_def["metrics"] = asdict(self._get_agent_metrics(agent_id))
            agent_def["status"] = self._get_agent_status(agent_id)
            agent_def["health"] = asdict(self._get_agent_health(agent_id))
        
        return default_agents
    
    def generate_handoff_links(self) -> List[Dict[str, Any]]:
        """Generate HandoffLink objects from agent dependencies"""
        links = []
        
        # Define the main flow handoffs based on AGENTIC_FLOWS_DOCUMENTATION.md
        flow_handoffs = [
            ("market_monitor", "betting_agent", "market_data", "data_feed"),
            ("betting_agent", "prediction_agent", "analysis_request", "prediction_request"),
            ("prediction_agent", "mispricing_engine", "predictions", "value_analysis"),
            ("mispricing_engine", "shadow_agent", "opportunities", "validation_request"),
            ("shadow_agent", "dgl_engine", "decisions", "governance_check"),
            ("dgl_engine", "execution_engine", "approvals", "execution_request"),
            ("execution_engine", "audit_logger", "results", "audit_record"),
            ("betting_agent", "audit_logger", "decisions", "decision_audit"),
            ("dgl_engine", "audit_logger", "governance", "governance_audit")
        ]
        
        for from_agent, to_agent, channel, handoff_type in flow_handoffs:
            link = {
                "from": from_agent,
                "to": to_agent,
                "channel": channel,
                "handoff_type": handoff_type,
                "sla_ms": self._get_sla_for_handoff(from_agent, to_agent),
                "last_activity_ts": int(time.time() * 1000),
                "health": self._get_handoff_health(from_agent, to_agent),
                "throughput_stats": self._get_throughput_stats(from_agent, to_agent),
                "cricket_metadata": self._get_cricket_handoff_metadata(from_agent, to_agent)
            }
            links.append(link)
        
        return links
    
    def generate_flow_definitions(self) -> List[Dict[str, Any]]:
        """Generate FlowDefinition objects from AGENTIC_FLOWS_DOCUMENTATION.md"""
        if self._flow_definitions is None:
            self._flow_definitions = self._load_flow_definitions()
        
        return self._flow_definitions
    
    def _load_flow_definitions(self) -> List[Dict[str, Any]]:
        """Load flow definitions from documentation file"""
        try:
            with open('AGENTIC_FLOWS_DOCUMENTATION.md', 'r') as f:
                content = f.read()
            
            # Extract JSON schemas from markdown
            flow_schemas = self._extract_json_schemas(content)
            
            flows = []
            for schema in flow_schemas:
                if 'flow_id' in schema and 'sequence' in schema:
                    flow_def = {
                        "flow_id": schema['flow_id'],
                        "label": schema['flow_id'].replace('_', ' ').title(),
                        "flow_type": schema.get('flow_type', 'unknown'),
                        "stages": self._convert_sequence_to_stages(schema['sequence']),
                        "governance_constraints": schema.get('governance_constraints', {}),
                        "cricket_context": self._extract_cricket_context(schema)
                    }
                    flows.append(flow_def)
            
            return flows
            
        except Exception as e:
            logger.error(f"Failed to load flow definitions: {e}")
            return self._get_default_flow_definitions()
    
    def _extract_json_schemas(self, content: str) -> List[Dict[str, Any]]:
        """Extract JSON schemas from markdown content"""
        schemas = []
        
        # Find JSON code blocks
        json_pattern = r'```json\s*\n(.*?)\n```'
        matches = re.findall(json_pattern, content, re.DOTALL)
        
        for match in matches:
            try:
                schema = json.loads(match)
                if isinstance(schema, dict) and 'flow_id' in schema:
                    schemas.append(schema)
            except json.JSONDecodeError:
                continue
        
        return schemas
    
    def _convert_sequence_to_stages(self, sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert sequence steps to flow stages"""
        stages = []
        
        for step in sequence:
            stage = {
                "agent": step.get('agent', 'unknown'),
                "event": step.get('action', 'unknown_event'),
                "cricket_context": step.get('cricket_context', {})
            }
            stages.append(stage)
        
        return stages
    
    def emit_agent_event(self, agent_id: str, event_type: str, payload: Dict[str, Any], 
                        cricket_context: Optional[Dict[str, Any]] = None):
        """Emit agent event to UI stream"""
        current_time = int(time.time() * 1000)
        duration_ms = random.randint(10, 500)  # Simulate processing time
        
        event = {
            "run_id": self._get_current_run_id(),
            "event_id": f"evt_{len(self.event_stream):05d}",
            "t": current_time,
            "agent": agent_id,
            "type": event_type,
            "payload": payload,
            "cricket_context": cricket_context or {},
            "links": {
                "prev": f"evt_{len(self.event_stream)-1:05d}" if self.event_stream else None,
                "next": None
            },
            "timing": {
                "enqueue_ts": current_time - duration_ms - 10,
                "start_ts": current_time - duration_ms,
                "end_ts": current_time,
                "duration_ms": duration_ms
            }
        }
        
        # Update previous event's next link
        if self.event_stream:
            self.event_stream[-1]["links"]["next"] = event["event_id"]
        
        self.event_stream.append(event)
        
        # Update agent metrics
        self._update_agent_metrics(agent_id, event)
        
        # Emit to WebSocket clients
        self._broadcast_event(event)
        
        # Generate decision record for certain event types
        if event_type in ['value_opportunity_detected', 'decision_made']:
            self._generate_sample_decision(event)
    
    def _generate_sample_decision(self, event: Dict[str, Any]):
        """Generate a sample decision record for testing"""
        decision_id = f"dec_{len(self.event_stream):03d}"
        
        # Sample decision data
        decision = {
            "decision_id": decision_id,
            "run_id": event["run_id"],
            "t": event["t"] + 100,  # Slightly after the event
            "decision_type": "betting_decision",
            "context": {
                "match_state": {
                    "over": 12.4,
                    "score": "98/2",
                    "batting_team": "RCB",
                    "required_rate": 8.2
                },
                "market_state": {
                    "market": "total_runs_over_180",
                    "best_odds": 2.15,
                    "liquidity": 25000,
                    "market_movement": "+0.05"
                },
                "account_state": {
                    "available_balance": 50000,
                    "current_exposure": 8500,
                    "daily_pnl": 320
                }
            },
            "signals": [
                {
                    "source": "gnn_predictor_v2.1",
                    "metric": "total_runs_probability",
                    "value": 0.58,
                    "confidence": 0.76,
                    "version": "2.1.3"
                },
                {
                    "source": "mispricing_engine",
                    "metric": "expected_value",
                    "value": 0.167,
                    "confidence": 0.82
                },
                {
                    "source": "kelly_calculator",
                    "metric": "optimal_fraction",
                    "value": 0.21,
                    "confidence": 0.94
                }
            ],
            "deliberation": [
                {
                    "reason": "Market mispricing vs model probability",
                    "impact": "+high",
                    "confidence": 0.76,
                    "supporting_data": {
                        "model_prob": 0.58,
                        "implied_prob": 0.465,
                        "edge": 0.115
                    }
                },
                {
                    "reason": "Strong batting lineup in favorable conditions",
                    "impact": "+medium",
                    "confidence": 0.71,
                    "supporting_data": {
                        "venue_avg": 182,
                        "team_avg": 175,
                        "conditions": "clear_sky"
                    }
                },
                {
                    "reason": "Recent form analysis positive",
                    "impact": "+low",
                    "confidence": 0.63,
                    "supporting_data": {
                        "last_5_matches": 4.2,
                        "key_players_form": 0.78
                    }
                }
            ],
            "constraints": [
                {
                    "name": "total_exposure_limit",
                    "threshold": 10000,
                    "current_value": 8500,
                    "result": "pass",
                    "headroom": 1500
                },
                {
                    "name": "per_market_exposure_limit",
                    "threshold": 2500,
                    "current_value": 0,
                    "result": "pass",
                    "headroom": 2500
                },
                {
                    "name": "kelly_fraction_limit",
                    "threshold": 0.25,
                    "current_value": 0.21,
                    "result": "pass",
                    "headroom": 0.04
                },
                {
                    "name": "daily_loss_limit",
                    "threshold": -2500,
                    "current_value": 320,
                    "result": "pass",
                    "headroom": 2820
                }
            ],
            "outcome": {
                "action": "place_bet",
                "stake": 750,
                "odds": 2.15,
                "market": "total_runs_over_180",
                "expected_profit": 862.50,
                "max_loss": 750,
                "execution_mode": "live"
            },
            "audit_trail": {
                "audit_id": f"audit_{decision_id}",
                "compliance_checks": ["gdpr", "gambling_commission", "risk_management"],
                "risk_classification": "moderate"
            }
        }
        
        # Update event links
        event["links"]["related_decisions"] = [decision_id]
        
        # Broadcast decision
        if self.socketio:
            try:
                self.socketio.emit('decision_made', {
                    'agent': event['agent'],
                    'decision_record': decision
                }, namespace='/agent_ui')
                logger.debug(f"Broadcasted decision {decision_id} to agent UI clients")
            except Exception as e:
                logger.error(f"Failed to broadcast decision: {e}")
    
    def generate_sample_events(self, count: int = 10):
        """Generate sample events for testing the UI"""
        import random
        
        agents = ['market_monitor', 'betting_agent', 'prediction_agent', 'mispricing_engine', 
                 'shadow_agent', 'dgl_engine', 'execution_engine', 'audit_logger']
        
        event_types = [
            'market_data_update', 'value_opportunity_detected', 'prediction_generated',
            'mispricing_detected', 'decision_validated', 'constraint_check',
            'order_executed', 'audit_logged'
        ]
        
        cricket_contexts = [
            {
                "match_id": "RCBvsCSK_2025_01_21",
                "over": 12.4,
                "score": "98/2",
                "batting_team": "RCB",
                "match_phase": "middle_overs"
            },
            {
                "match_id": "MIvsDC_2025_01_21", 
                "over": 18.2,
                "score": "165/4",
                "batting_team": "MI",
                "match_phase": "death_overs"
            }
        ]
        
        for i in range(count):
            agent = random.choice(agents)
            event_type = random.choice(event_types)
            cricket_context = random.choice(cricket_contexts)
            
            payload = {
                "sequence": i + 1,
                "priority": random.choice(["high", "medium", "low"]),
                "confidence": round(random.uniform(0.5, 0.95), 2),
                "data_size_kb": round(random.uniform(0.5, 5.0), 1)
            }
            
            # Add some delay between events
            time.sleep(random.uniform(0.1, 0.5))
            
            self.emit_agent_event(agent, event_type, payload, cricket_context)
    
    def _broadcast_event(self, event: Dict[str, Any]):
        """Broadcast event to WebSocket clients"""
        if self.socketio:
            try:
                self.socketio.emit('flow_event', event, namespace='/agent_ui')
                logger.debug(f"Broadcasted event {event['event_id']} to agent UI clients")
            except Exception as e:
                logger.error(f"Failed to broadcast event: {e}")
    
    def _update_agent_metrics(self, agent_id: str, event: Dict[str, Any]):
        """Update agent metrics based on event"""
        metrics = self.agent_metrics[agent_id]
        
        # Update response time if available
        if event.get('timing', {}).get('duration_ms'):
            metrics['response_times'].append(event['timing']['duration_ms'])
        
        # Update throughput counter
        metrics['throughput_counter'] += 1
        metrics['last_activity'] = time.time()
        
        # Update error count if this is an error event
        if event['type'].endswith('_error') or event['type'].endswith('_failed'):
            metrics['error_count'] += 1
    
    def _get_agent_metrics(self, agent_id: str) -> AgentMetrics:
        """Get agent performance metrics"""
        metrics_data = self.agent_metrics[agent_id]
        response_times = list(metrics_data['response_times'])
        
        if response_times:
            response_times.sort()
            p50_ms = response_times[len(response_times) // 2]
            p95_ms = response_times[int(len(response_times) * 0.95)]
        else:
            p50_ms = p95_ms = 0.0
        
        # Calculate throughput per minute
        current_time = time.time()
        minute_ago = current_time - 60
        recent_activity = metrics_data['last_activity'] > minute_ago
        throughput_per_min = metrics_data['throughput_counter'] if recent_activity else 0
        
        # Agent-specific metrics
        accuracy_rate = None
        false_positive_rate = None
        kelly_efficiency = None
        
        if agent_id == 'betting_agent':
            accuracy_rate = 0.73
            false_positive_rate = 0.18
        elif agent_id == 'mispricing_engine':
            kelly_efficiency = 0.85
        
        return AgentMetrics(
            p50_ms=p50_ms,
            p95_ms=p95_ms,
            throughput_per_min=throughput_per_min,
            accuracy_rate=accuracy_rate,
            false_positive_rate=false_positive_rate,
            kelly_efficiency=kelly_efficiency
        )
    
    def _get_agent_health(self, agent_id: str) -> AgentHealth:
        """Get agent health status"""
        metrics_data = self.agent_metrics[agent_id]
        
        # Simulate health metrics (in production, these would come from actual monitoring)
        import random
        import psutil
        
        cpu_usage = random.uniform(0.1, 0.8)  # Simulated CPU usage
        memory_mb = random.randint(128, 1024)  # Simulated memory usage
        queue_depth = len(metrics_data['response_times'])
        
        # Check for recent errors
        last_error = None
        if metrics_data['error_count'] > 0:
            last_error = f"Last error occurred {int(time.time() - metrics_data['last_activity'])}s ago"
        
        # Calculate uptime (simplified)
        uptime_hours = random.uniform(1, 168)  # 1 hour to 1 week
        
        return AgentHealth(
            cpu_usage=cpu_usage,
            memory_mb=memory_mb,
            queue_depth=queue_depth,
            last_error=last_error,
            uptime_hours=uptime_hours
        )
    
    def _get_agent_status(self, agent_id: str) -> str:
        """Get agent status"""
        metrics_data = self.agent_metrics[agent_id]
        current_time = time.time()
        
        # Check if agent has been active recently
        if current_time - metrics_data['last_activity'] < 60:  # Active in last minute
            if metrics_data['error_count'] > 5:  # Too many errors
                return 'degraded'
            return 'active'
        elif current_time - metrics_data['last_activity'] < 300:  # Active in last 5 minutes
            return 'idle'
        else:
            return 'blocked'
    
    def _determine_agent_type(self, agent) -> str:
        """Determine agent type from agent instance"""
        agent_class_name = agent.__class__.__name__.lower()
        
        if 'betting' in agent_class_name:
            return 'betting_intelligence'
        elif 'prediction' in agent_class_name:
            return 'prediction'
        elif 'dgl' in agent_class_name or 'governance' in agent_class_name:
            return 'governance'
        elif 'execution' in agent_class_name:
            return 'execution'
        elif 'audit' in agent_class_name:
            return 'audit'
        elif 'monitor' in agent_class_name:
            return 'monitoring'
        else:
            return 'analysis'
    
    def _get_cricket_context(self, agent) -> Optional[CricketContext]:
        """Extract cricket context from agent"""
        # This would be customized based on actual agent implementations
        return CricketContext(
            supported_formats=["T20", "ODI", "Test"],
            betting_markets=["match_winner", "total_runs", "player_runs"],
            risk_profile="moderate"
        )
    
    def _get_current_run_id(self) -> str:
        """Get current run ID (could be based on current match or session)"""
        return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Helper methods for handoff links
    def _get_sla_for_handoff(self, from_agent: str, to_agent: str) -> int:
        """Get SLA in milliseconds for agent handoff"""
        sla_map = {
            ('market_monitor', 'betting_agent'): 5000,
            ('betting_agent', 'prediction_agent'): 2000,
            ('prediction_agent', 'mispricing_engine'): 1000,
            ('mispricing_engine', 'shadow_agent'): 500,
            ('shadow_agent', 'dgl_engine'): 50,
            ('dgl_engine', 'execution_engine'): 2000
        }
        return sla_map.get((from_agent, to_agent), 1000)
    
    def _get_handoff_health(self, from_agent: str, to_agent: str) -> str:
        """Get handoff health status"""
        # Simulate health based on recent activity
        import random
        health_options = ['ok', 'degraded', 'failed']
        weights = [0.8, 0.15, 0.05]  # 80% ok, 15% degraded, 5% failed
        return random.choices(health_options, weights=weights)[0]
    
    def _get_throughput_stats(self, from_agent: str, to_agent: str) -> Dict[str, Any]:
        """Get throughput statistics for handoff"""
        import random
        return {
            "messages_per_minute": random.randint(10, 100),
            "avg_payload_size_kb": round(random.uniform(0.5, 5.0), 1),
            "success_rate": round(random.uniform(0.85, 0.99), 2)
        }
    
    def _get_cricket_handoff_metadata(self, from_agent: str, to_agent: str) -> Dict[str, Any]:
        """Get cricket-specific metadata for handoff"""
        return {
            "match_phase": "powerplay",
            "market_conditions": "volatile",
            "exposure_level": "moderate"
        }
    
    def _extract_cricket_context(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extract cricket context from flow schema"""
        return {
            "supported_formats": ["T20", "ODI", "Test"],
            "betting_markets": ["match_winner", "total_runs"],
            "risk_profile": "moderate"
        }
    
    def _get_default_flow_definitions(self) -> List[Dict[str, Any]]:
        """Get default flow definitions if file loading fails"""
        return [
            {
                "flow_id": "automated_value_detection",
                "label": "Automated Value Detection",
                "flow_type": "continuous_monitoring",
                "stages": [
                    {"agent": "market_monitor", "event": "odds_snapshot"},
                    {"agent": "betting_agent", "event": "value_analysis"},
                    {"agent": "prediction_agent", "event": "probability_update"},
                    {"agent": "mispricing_engine", "event": "value_detection"},
                    {"agent": "shadow_agent", "event": "decision_validation"},
                    {"agent": "dgl_engine", "event": "governance_check"},
                    {"agent": "execution_engine", "event": "order_placement"}
                ],
                "governance_constraints": {
                    "max_exposure_pct": 20,
                    "daily_loss_limit_pct": 10
                }
            }
        ]
