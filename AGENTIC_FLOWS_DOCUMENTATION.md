# 🤖 WicketWise Agentic Flows - Complete Documentation

**Version**: 1.0  
**Date**: January 21, 2025  
**Status**: Production Ready  
**Author**: WicketWise AI Team

---

## 📋 **OVERVIEW**

This document provides a comprehensive, machine-readable map of all agentic flows in the WicketWise betting automation system. Each flow is documented with triggers, agents involved, decisions, governance constraints, and feedback loops.

---

## 🔄 **FLOW 1: AUTOMATED VALUE DETECTION & EXECUTION**

### **Description**
The primary automated betting flow that continuously monitors markets, detects value opportunities, validates decisions through multiple agents, applies risk controls, and executes approved bets.

### **Flow Steps**

1. **Trigger/Input**: Market data update (every 30 seconds) or user enables auto-betting
2. **Agents/Modules Involved**:
   - Market Monitor → Betting Agent → Prediction Agent → Mispricing Engine → Shadow Agent → DGL → Execution Engine
3. **Decisions/Outputs**: 
   - Value opportunities identified
   - Bet/no-bet decision with confidence
   - Risk-adjusted position sizing
   - Execution confirmation or rejection
4. **Governance/Constraints**:
   - Bankroll limits (max 20% total exposure)
   - Per-market caps (max 5% per market)
   - Daily loss limits (max 10% daily loss)
   - Kelly fraction limits (max 25% of full Kelly)
5. **Feedback/Logging**:
   - All decisions logged to audit trail
   - Performance metrics updated
   - Model predictions vs outcomes tracked

### **JSON Schema**
```json
{
  "flow_id": "automated_value_detection",
  "flow_type": "continuous_monitoring",
  "trigger": {
    "type": "scheduled_event",
    "interval_seconds": 30,
    "conditions": ["auto_betting_enabled", "market_data_available"]
  },
  "sequence": [
    {
      "step": 1,
      "agent": "market_monitor",
      "action": "fetch_latest_odds",
      "inputs": ["betfair_api", "bet365_api", "pinnacle_api"],
      "outputs": ["normalized_odds_data"],
      "timeout_ms": 5000
    },
    {
      "step": 2,
      "agent": "betting_agent",
      "action": "analyze_opportunities",
      "inputs": ["normalized_odds_data", "match_context"],
      "outputs": ["potential_opportunities"],
      "dependencies": ["market_monitor"],
      "timeout_ms": 2000
    },
    {
      "step": 3,
      "agent": "prediction_agent",
      "action": "generate_probabilities",
      "inputs": ["match_context", "player_stats", "gnn_embeddings"],
      "outputs": ["model_probabilities", "confidence_scores"],
      "dependencies": ["betting_agent"],
      "timeout_ms": 3000
    },
    {
      "step": 4,
      "agent": "mispricing_engine",
      "action": "detect_value",
      "inputs": ["model_probabilities", "market_odds"],
      "outputs": ["value_opportunities", "expected_values", "kelly_fractions"],
      "dependencies": ["prediction_agent"],
      "timeout_ms": 1000
    },
    {
      "step": 5,
      "agent": "shadow_agent",
      "action": "validate_decision",
      "inputs": ["value_opportunities", "confidence_scores"],
      "outputs": ["validated_decisions", "risk_alerts"],
      "dependencies": ["mispricing_engine"],
      "timeout_ms": 500
    },
    {
      "step": 6,
      "agent": "dgl_engine",
      "action": "evaluate_proposal",
      "inputs": ["validated_decisions", "current_exposure", "account_state"],
      "outputs": ["governance_decision", "amended_size", "rule_violations"],
      "dependencies": ["shadow_agent"],
      "timeout_ms": 50
    },
    {
      "step": 7,
      "agent": "execution_engine",
      "action": "place_order",
      "inputs": ["approved_proposals", "market_conditions"],
      "outputs": ["execution_confirmation", "fill_details"],
      "dependencies": ["dgl_engine"],
      "condition": "governance_decision == APPROVE",
      "timeout_ms": 2000
    },
    {
      "step": 8,
      "agent": "audit_logger",
      "action": "record_decision",
      "inputs": ["all_step_outputs", "timestamps", "decision_chain"],
      "outputs": ["audit_record", "performance_metrics"],
      "dependencies": ["execution_engine"],
      "timeout_ms": 100
    }
  ],
  "governance_constraints": {
    "bankroll_limits": {
      "total_exposure_pct": 20,
      "per_market_pct": 5,
      "per_bet_pct": 2
    },
    "pnl_limits": {
      "daily_loss_pct": 10,
      "consecutive_losses": 5,
      "session_loss_pct": 15
    },
    "operational_limits": {
      "max_orders_per_minute": 10,
      "circuit_breaker_loss_pct": 5
    }
  },
  "feedback_loops": {
    "performance_tracking": ["roi", "sharpe_ratio", "win_rate", "kelly_efficiency"],
    "model_validation": ["prediction_accuracy", "calibration_error"],
    "risk_monitoring": ["drawdown", "var", "correlation_exposure"]
  }
}
```

---

## 🔄 **FLOW 2: MANUAL STRATEGY OVERRIDE & SIMULATION**

### **Description**
User-initiated flow for testing custom strategies, adjusting risk parameters, and running paper trading simulations before live deployment.

### **Flow Steps**

1. **Trigger/Input**: User selects custom strategy or modifies parameters
2. **Agents/Modules Involved**:
   - Dashboard → Strategy Configurator → Simulation Engine → DGL → Performance Analyzer
3. **Decisions/Outputs**:
   - Strategy configuration validated
   - Simulation results with performance metrics
   - Risk assessment and recommendations
   - Approval for live trading
4. **Governance/Constraints**:
   - Strategy must pass minimum performance thresholds
   - Risk parameters within acceptable bounds
   - Simulation period requirements met
5. **Feedback/Logging**:
   - Simulation results stored
   - Strategy performance compared to benchmarks
   - User decisions tracked

### **JSON Schema**
```json
{
  "flow_id": "manual_strategy_override",
  "flow_type": "user_initiated",
  "trigger": {
    "type": "user_action",
    "actions": ["strategy_selection", "parameter_modification", "simulation_request"]
  },
  "sequence": [
    {
      "step": 1,
      "agent": "strategy_configurator",
      "action": "validate_parameters",
      "inputs": ["user_strategy_config", "risk_parameters"],
      "outputs": ["validated_config", "parameter_warnings"],
      "timeout_ms": 1000
    },
    {
      "step": 2,
      "agent": "simulation_engine",
      "action": "load_historical_data",
      "inputs": ["date_range", "market_selection", "match_data"],
      "outputs": ["simulation_dataset", "market_conditions"],
      "dependencies": ["strategy_configurator"],
      "timeout_ms": 5000
    },
    {
      "step": 3,
      "agent": "sim_orchestrator",
      "action": "run_backtest",
      "inputs": ["validated_config", "simulation_dataset"],
      "outputs": ["trade_history", "performance_metrics", "risk_metrics"],
      "dependencies": ["simulation_engine"],
      "timeout_ms": 30000
    },
    {
      "step": 4,
      "agent": "dgl_adapter",
      "action": "validate_trades",
      "inputs": ["trade_history", "risk_parameters"],
      "outputs": ["rule_violations", "risk_assessment"],
      "dependencies": ["sim_orchestrator"],
      "timeout_ms": 2000
    },
    {
      "step": 5,
      "agent": "performance_analyzer",
      "action": "generate_report",
      "inputs": ["performance_metrics", "risk_metrics", "benchmark_data"],
      "outputs": ["performance_report", "recommendations"],
      "dependencies": ["dgl_adapter"],
      "timeout_ms": 3000
    },
    {
      "step": 6,
      "agent": "approval_engine",
      "action": "evaluate_for_live",
      "inputs": ["performance_report", "risk_assessment", "user_approval"],
      "outputs": ["live_trading_approval", "conditions"],
      "dependencies": ["performance_analyzer"],
      "timeout_ms": 1000
    }
  ],
  "governance_constraints": {
    "performance_thresholds": {
      "min_sharpe_ratio": 1.0,
      "max_drawdown_pct": 25,
      "min_win_rate_pct": 45
    },
    "simulation_requirements": {
      "min_trades": 50,
      "min_days": 30,
      "required_scenarios": ["bull_market", "bear_market", "volatile_market"]
    }
  },
  "feedback_loops": {
    "strategy_optimization": ["parameter_sensitivity", "performance_attribution"],
    "user_learning": ["decision_outcomes", "strategy_effectiveness"]
  }
}
```

---

## 🔄 **FLOW 3: RISK ALERT & CIRCUIT BREAKER**

### **Description**
Emergency response flow triggered by risk limit breaches, significant losses, or system anomalies. Implements immediate protective actions.

### **Flow Steps**

1. **Trigger/Input**: Loss limit exceeded, circuit breaker triggered, or system anomaly detected
2. **Agents/Modules Involved**:
   - Risk Monitor → Alert System → DGL → Emergency Controller → Notification System
3. **Decisions/Outputs**:
   - Risk alert classification
   - Emergency actions (halt trading, cancel orders)
   - Stakeholder notifications
   - Recovery procedures initiated
4. **Governance/Constraints**:
   - Immediate trading halt
   - All pending orders cancelled
   - Manual approval required to resume
5. **Feedback/Logging**:
   - Incident report generated
   - Root cause analysis initiated
   - Recovery time tracked

### **JSON Schema**
```json
{
  "flow_id": "risk_alert_circuit_breaker",
  "flow_type": "emergency_response",
  "trigger": {
    "type": "threshold_breach",
    "conditions": [
      "daily_loss > daily_loss_limit",
      "drawdown > max_drawdown",
      "consecutive_losses > max_consecutive",
      "system_anomaly_detected"
    ]
  },
  "sequence": [
    {
      "step": 1,
      "agent": "risk_monitor",
      "action": "classify_alert",
      "inputs": ["breach_type", "severity_metrics", "current_state"],
      "outputs": ["alert_classification", "urgency_level"],
      "timeout_ms": 100
    },
    {
      "step": 2,
      "agent": "dgl_engine",
      "action": "activate_circuit_breaker",
      "inputs": ["alert_classification", "current_positions"],
      "outputs": ["trading_halt_status", "cancelled_orders"],
      "dependencies": ["risk_monitor"],
      "timeout_ms": 50
    },
    {
      "step": 3,
      "agent": "emergency_controller",
      "action": "execute_emergency_actions",
      "inputs": ["urgency_level", "current_positions", "market_conditions"],
      "outputs": ["emergency_actions_taken", "position_adjustments"],
      "dependencies": ["dgl_engine"],
      "timeout_ms": 500
    },
    {
      "step": 4,
      "agent": "notification_system",
      "action": "send_alerts",
      "inputs": ["alert_classification", "emergency_actions_taken"],
      "outputs": ["notifications_sent", "escalation_triggered"],
      "dependencies": ["emergency_controller"],
      "timeout_ms": 2000
    },
    {
      "step": 5,
      "agent": "incident_manager",
      "action": "create_incident_report",
      "inputs": ["all_step_outputs", "system_state", "market_context"],
      "outputs": ["incident_report", "recovery_procedures"],
      "dependencies": ["notification_system"],
      "timeout_ms": 5000
    }
  ],
  "governance_constraints": {
    "immediate_actions": {
      "halt_all_trading": true,
      "cancel_pending_orders": true,
      "freeze_new_positions": true
    },
    "recovery_requirements": {
      "manual_approval_required": true,
      "risk_review_mandatory": true,
      "system_health_check": true
    }
  },
  "feedback_loops": {
    "incident_analysis": ["root_cause", "prevention_measures", "system_improvements"],
    "recovery_tracking": ["downtime_duration", "recovery_effectiveness"]
  }
}
```

---

## 🔄 **FLOW 4: LIVE BALL-BY-BALL PREDICTION & BETTING**

### **Description**
Real-time flow that processes live cricket match events, updates predictions, and makes in-play betting decisions based on match dynamics.

### **Flow Steps**

1. **Trigger/Input**: Live ball event from cricket match feed
2. **Agents/Modules Involved**:
   - Live Data Feed → GNN Predictor → Betting Intelligence → Shadow Agent → DGL → Execution
3. **Decisions/Outputs**:
   - Updated match probabilities
   - In-play betting opportunities
   - Position adjustments
   - Execution decisions
4. **Governance/Constraints**:
   - In-play exposure limits
   - Momentum-based position sizing
   - Market suspension handling
5. **Feedback/Logging**:
   - Ball-by-ball predictions logged
   - In-play performance tracked
   - Market timing analysis

### **JSON Schema**
```json
{
  "flow_id": "live_ball_prediction_betting",
  "flow_type": "real_time_event",
  "trigger": {
    "type": "live_data_event",
    "sources": ["cricsheet_live", "match_feed", "ball_by_ball_data"]
  },
  "sequence": [
    {
      "step": 1,
      "agent": "live_data_processor",
      "action": "process_ball_event",
      "inputs": ["ball_data", "match_state", "player_context"],
      "outputs": ["processed_event", "match_update"],
      "timeout_ms": 200
    },
    {
      "step": 2,
      "agent": "gnn_predictor",
      "action": "update_predictions",
      "inputs": ["processed_event", "player_embeddings", "match_context"],
      "outputs": ["updated_probabilities", "confidence_scores"],
      "dependencies": ["live_data_processor"],
      "timeout_ms": 500
    },
    {
      "step": 3,
      "agent": "betting_intelligence_engine",
      "action": "analyze_in_play_opportunities",
      "inputs": ["updated_probabilities", "live_odds", "market_conditions"],
      "outputs": ["in_play_opportunities", "momentum_signals"],
      "dependencies": ["gnn_predictor"],
      "timeout_ms": 300
    },
    {
      "step": 4,
      "agent": "shadow_agent",
      "action": "validate_in_play_decision",
      "inputs": ["in_play_opportunities", "current_positions", "market_volatility"],
      "outputs": ["validated_decisions", "position_adjustments"],
      "dependencies": ["betting_intelligence_engine"],
      "timeout_ms": 100
    },
    {
      "step": 5,
      "agent": "dgl_engine",
      "action": "evaluate_in_play_proposal",
      "inputs": ["validated_decisions", "in_play_exposure", "market_suspension_risk"],
      "outputs": ["in_play_approval", "adjusted_sizing"],
      "dependencies": ["shadow_agent"],
      "timeout_ms": 50
    },
    {
      "step": 6,
      "agent": "execution_engine",
      "action": "execute_in_play_trade",
      "inputs": ["in_play_approval", "market_timing", "liquidity_conditions"],
      "outputs": ["execution_result", "slippage_analysis"],
      "dependencies": ["dgl_engine"],
      "condition": "in_play_approval == APPROVE",
      "timeout_ms": 1000
    }
  ],
  "governance_constraints": {
    "in_play_limits": {
      "max_in_play_exposure_pct": 10,
      "momentum_position_limit": 3,
      "market_suspension_buffer_seconds": 30
    },
    "timing_constraints": {
      "max_execution_delay_ms": 2000,
      "market_data_freshness_ms": 500
    }
  },
  "feedback_loops": {
    "prediction_accuracy": ["ball_by_ball_accuracy", "momentum_prediction_quality"],
    "execution_quality": ["in_play_slippage", "timing_effectiveness"]
  }
}
```

---

## 🔄 **FLOW 5: ARBITRAGE DETECTION & EXECUTION**

### **Description**
Specialized flow for detecting and executing risk-free arbitrage opportunities across multiple bookmakers with time-sensitive execution.

### **Flow Steps**

1. **Trigger/Input**: Odds discrepancy detected across bookmakers
2. **Agents/Modules Involved**:
   - Odds Aggregator → Arbitrage Scanner → Multi-Account Manager → DGL → Parallel Execution
3. **Decisions/Outputs**:
   - Arbitrage opportunities identified
   - Optimal stake allocation calculated
   - Multi-bookmaker execution plan
   - Guaranteed profit confirmation
4. **Governance/Constraints**:
   - Account balance verification
   - Execution time limits
   - Profit margin thresholds
5. **Feedback/Logging**:
   - Arbitrage success rate tracked
   - Execution timing analyzed
   - Profit margins recorded

### **JSON Schema**
```json
{
  "flow_id": "arbitrage_detection_execution",
  "flow_type": "opportunity_driven",
  "trigger": {
    "type": "market_inefficiency",
    "conditions": ["odds_discrepancy > arbitrage_threshold", "sufficient_liquidity"]
  },
  "sequence": [
    {
      "step": 1,
      "agent": "arbitrage_scanner",
      "action": "detect_opportunities",
      "inputs": ["multi_bookmaker_odds", "market_depth", "liquidity_data"],
      "outputs": ["arbitrage_opportunities", "profit_margins", "time_sensitivity"],
      "timeout_ms": 500
    },
    {
      "step": 2,
      "agent": "stake_optimizer",
      "action": "calculate_optimal_stakes",
      "inputs": ["arbitrage_opportunities", "account_balances", "profit_targets"],
      "outputs": ["optimal_stakes", "expected_profit", "capital_requirements"],
      "dependencies": ["arbitrage_scanner"],
      "timeout_ms": 200
    },
    {
      "step": 3,
      "agent": "multi_account_manager",
      "action": "verify_account_status",
      "inputs": ["optimal_stakes", "bookmaker_accounts", "balance_requirements"],
      "outputs": ["account_verification", "available_balances", "execution_feasibility"],
      "dependencies": ["stake_optimizer"],
      "timeout_ms": 1000
    },
    {
      "step": 4,
      "agent": "dgl_engine",
      "action": "validate_arbitrage_proposal",
      "inputs": ["execution_feasibility", "risk_free_nature", "capital_allocation"],
      "outputs": ["arbitrage_approval", "execution_authorization"],
      "dependencies": ["multi_account_manager"],
      "timeout_ms": 50
    },
    {
      "step": 5,
      "agent": "parallel_execution_engine",
      "action": "execute_arbitrage_legs",
      "inputs": ["arbitrage_approval", "optimal_stakes", "bookmaker_apis"],
      "outputs": ["execution_results", "fill_confirmations", "actual_profit"],
      "dependencies": ["dgl_engine"],
      "execution_mode": "parallel",
      "timeout_ms": 3000
    },
    {
      "step": 6,
      "agent": "arbitrage_reconciler",
      "action": "reconcile_positions",
      "inputs": ["execution_results", "expected_outcomes", "partial_fills"],
      "outputs": ["position_status", "profit_realization", "risk_exposure"],
      "dependencies": ["parallel_execution_engine"],
      "timeout_ms": 1000
    }
  ],
  "governance_constraints": {
    "arbitrage_limits": {
      "min_profit_margin_pct": 1.0,
      "max_capital_allocation_pct": 50,
      "execution_time_limit_seconds": 30
    },
    "account_requirements": {
      "min_accounts": 2,
      "balance_verification_required": true,
      "account_status_check": true
    }
  },
  "feedback_loops": {
    "arbitrage_performance": ["success_rate", "average_profit_margin", "execution_speed"],
    "market_efficiency": ["opportunity_frequency", "market_reaction_time"]
  }
}
```

---

## 🔄 **FLOW 6: AGENT ORCHESTRATION & COORDINATION**

### **Description**
Meta-flow that coordinates multiple agents for complex betting queries, manages agent lifecycle, and optimizes execution strategies.

### **Flow Steps**

1. **Trigger/Input**: Complex betting query requiring multiple agent capabilities
2. **Agents/Modules Involved**:
   - Query Analyzer → Orchestration Engine → Agent Coordinator → Result Aggregator
3. **Decisions/Outputs**:
   - Query decomposition plan
   - Agent execution strategy
   - Coordinated results
   - Performance optimization
4. **Governance/Constraints**:
   - Agent resource limits
   - Execution timeouts
   - Quality thresholds
5. **Feedback/Logging**:
   - Agent performance metrics
   - Coordination efficiency
   - Result quality scores

### **JSON Schema**
```json
{
  "flow_id": "agent_orchestration_coordination",
  "flow_type": "meta_coordination",
  "trigger": {
    "type": "complex_query",
    "conditions": ["multi_capability_required", "agent_coordination_needed"]
  },
  "sequence": [
    {
      "step": 1,
      "agent": "query_analyzer",
      "action": "decompose_query",
      "inputs": ["user_query", "context", "available_capabilities"],
      "outputs": ["query_components", "required_capabilities", "execution_strategy"],
      "timeout_ms": 1000
    },
    {
      "step": 2,
      "agent": "orchestration_engine",
      "action": "create_execution_plan",
      "inputs": ["query_components", "agent_availability", "performance_history"],
      "outputs": ["execution_plan", "agent_assignments", "dependency_graph"],
      "dependencies": ["query_analyzer"],
      "timeout_ms": 500
    },
    {
      "step": 3,
      "agent": "agent_coordinator",
      "action": "execute_coordinated_plan",
      "inputs": ["execution_plan", "agent_pool", "resource_constraints"],
      "outputs": ["agent_responses", "execution_metrics", "failed_tasks"],
      "dependencies": ["orchestration_engine"],
      "execution_mode": "coordinated",
      "timeout_ms": 10000
    },
    {
      "step": 4,
      "agent": "result_aggregator",
      "action": "synthesize_results",
      "inputs": ["agent_responses", "confidence_scores", "result_weights"],
      "outputs": ["aggregated_result", "overall_confidence", "result_quality"],
      "dependencies": ["agent_coordinator"],
      "timeout_ms": 1000
    },
    {
      "step": 5,
      "agent": "performance_optimizer",
      "action": "update_agent_performance",
      "inputs": ["execution_metrics", "result_quality", "user_feedback"],
      "outputs": ["performance_updates", "optimization_recommendations"],
      "dependencies": ["result_aggregator"],
      "timeout_ms": 500
    }
  ],
  "governance_constraints": {
    "resource_limits": {
      "max_concurrent_agents": 5,
      "total_execution_timeout_ms": 15000,
      "memory_limit_mb": 1024
    },
    "quality_thresholds": {
      "min_confidence_score": 0.6,
      "max_failed_tasks": 1,
      "min_result_quality": 0.7
    }
  },
  "feedback_loops": {
    "agent_optimization": ["response_time", "accuracy", "resource_usage"],
    "coordination_improvement": ["plan_effectiveness", "dependency_optimization"]
  }
}
```

---

## 📊 **FLOW DEPENDENCIES & RELATIONSHIPS**

### **Dependency Graph**
```json
{
  "flow_dependencies": {
    "automated_value_detection": {
      "depends_on": [],
      "feeds_into": ["risk_alert_circuit_breaker", "live_ball_prediction_betting"],
      "shared_components": ["dgl_engine", "execution_engine", "audit_logger"]
    },
    "manual_strategy_override": {
      "depends_on": ["automated_value_detection"],
      "feeds_into": ["automated_value_detection"],
      "shared_components": ["dgl_adapter", "performance_analyzer"]
    },
    "risk_alert_circuit_breaker": {
      "depends_on": ["automated_value_detection", "live_ball_prediction_betting"],
      "feeds_into": ["manual_strategy_override"],
      "shared_components": ["risk_monitor", "dgl_engine"]
    },
    "live_ball_prediction_betting": {
      "depends_on": ["automated_value_detection"],
      "feeds_into": ["risk_alert_circuit_breaker"],
      "shared_components": ["gnn_predictor", "shadow_agent", "execution_engine"]
    },
    "arbitrage_detection_execution": {
      "depends_on": [],
      "feeds_into": ["risk_alert_circuit_breaker"],
      "shared_components": ["dgl_engine", "execution_engine"]
    },
    "agent_orchestration_coordination": {
      "depends_on": [],
      "feeds_into": ["all_flows"],
      "shared_components": ["all_agents"]
    }
  }
}
```

### **Shared Components Matrix**
```json
{
  "shared_components": {
    "dgl_engine": {
      "used_by": ["automated_value_detection", "risk_alert_circuit_breaker", "live_ball_prediction_betting", "arbitrage_detection_execution"],
      "function": "risk_validation_and_governance",
      "criticality": "high"
    },
    "execution_engine": {
      "used_by": ["automated_value_detection", "live_ball_prediction_betting", "arbitrage_detection_execution"],
      "function": "order_placement_and_execution",
      "criticality": "high"
    },
    "shadow_agent": {
      "used_by": ["automated_value_detection", "live_ball_prediction_betting"],
      "function": "final_decision_validation",
      "criticality": "medium"
    },
    "audit_logger": {
      "used_by": ["all_flows"],
      "function": "decision_tracking_and_compliance",
      "criticality": "high"
    },
    "performance_analyzer": {
      "used_by": ["manual_strategy_override", "agent_orchestration_coordination"],
      "function": "strategy_performance_evaluation",
      "criticality": "medium"
    }
  }
}
```

---

## 🎯 **FLOW PERFORMANCE METRICS**

### **Key Performance Indicators by Flow**
```json
{
  "flow_kpis": {
    "automated_value_detection": {
      "latency_metrics": {
        "end_to_end_latency_ms": {"target": 8000, "alert_threshold": 12000},
        "decision_latency_ms": {"target": 3000, "alert_threshold": 5000}
      },
      "accuracy_metrics": {
        "value_detection_accuracy": {"target": 0.75, "alert_threshold": 0.60},
        "false_positive_rate": {"target": 0.15, "alert_threshold": 0.25}
      },
      "business_metrics": {
        "opportunities_per_hour": {"target": 5, "alert_threshold": 2},
        "execution_success_rate": {"target": 0.95, "alert_threshold": 0.85}
      }
    },
    "risk_alert_circuit_breaker": {
      "response_metrics": {
        "alert_response_time_ms": {"target": 100, "alert_threshold": 500},
        "circuit_breaker_activation_ms": {"target": 50, "alert_threshold": 200}
      },
      "effectiveness_metrics": {
        "false_alert_rate": {"target": 0.05, "alert_threshold": 0.15},
        "risk_prevention_effectiveness": {"target": 0.90, "alert_threshold": 0.75}
      }
    },
    "arbitrage_detection_execution": {
      "speed_metrics": {
        "detection_to_execution_ms": {"target": 5000, "alert_threshold": 10000},
        "parallel_execution_success_rate": {"target": 0.90, "alert_threshold": 0.75}
      },
      "profitability_metrics": {
        "average_profit_margin": {"target": 0.02, "alert_threshold": 0.01},
        "opportunity_capture_rate": {"target": 0.60, "alert_threshold": 0.40}
      }
    }
  }
}
```

---

## 🔧 **IMPLEMENTATION NOTES**

### **Agent Communication Patterns**
- **Synchronous**: DGL validation, Shadow agent decisions
- **Asynchronous**: Market monitoring, Performance analysis
- **Event-driven**: Live ball events, Risk alerts
- **Batch processing**: Historical analysis, Simulation runs

### **Error Handling & Recovery**
- **Graceful degradation**: Fallback to simpler strategies
- **Circuit breakers**: Automatic system protection
- **Retry mechanisms**: Transient failure recovery
- **Manual overrides**: Human intervention capabilities

### **Scalability Considerations**
- **Horizontal scaling**: Agent pool expansion
- **Load balancing**: Request distribution
- **Caching strategies**: Performance optimization
- **Resource management**: Memory and CPU limits

---

**Document Status**: ✅ **COMPLETE**  
**Last Updated**: January 21, 2025  
**Next Review**: February 21, 2025

---

*This documentation provides a complete, machine-readable map of all agentic flows in the WicketWise betting automation system, enabling visualization, optimization, and systematic improvement of the agent coordination architecture.*
