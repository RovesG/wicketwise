# WicketWise Simulator & Market Replay (SIM)

A comprehensive offline and semi-online simulation environment for testing, validating, and optimizing cricket betting strategies with deterministic governance controls.

## 🎯 Overview

The WicketWise SIM system provides:

- **Historical Replay**: Ball-by-ball match events with synchronized market data
- **Strategy Testing**: Edge-based, mean reversion, and momentum strategies
- **Risk Management**: Integrated DGL (Deterministic Governance Layer) enforcement
- **Performance Analytics**: Comprehensive KPIs including Sharpe, Sortino, drawdown
- **Reproducible Results**: Deterministic execution with seed control

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Environment   │    │    Strategy      │    │   Matching      │
│    Adapters     │───▶│   Execution      │───▶│    Engine       │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Match/Market  │    │   DGL Risk       │    │   P&L & Metrics │
│     States      │    │   Enforcement    │    │   Calculation   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Basic Replay Simulation

```python
from sim.config import create_replay_config
from sim.orchestrator import SimOrchestrator

# Create configuration
config = create_replay_config(["match_id_1"], "edge_kelly_v3")

# Run simulation
orchestrator = SimOrchestrator()
orchestrator.initialize(config)
result = orchestrator.run()

print(f"P&L: £{result.kpis.pnl_total:.2f}")
print(f"Sharpe: {result.kpis.sharpe:.3f}")
```

### Strategy Comparison

```python
strategies = ["edge_kelly_v3", "mean_revert_lob", "momentum_follow"]

for strategy in strategies:
    config = create_replay_config(["match_id_1"], strategy)
    config.seed = 42  # Same seed for fair comparison
    
    orchestrator = SimOrchestrator()
    orchestrator.initialize(config)
    result = orchestrator.run()
    
    print(f"{strategy}: £{result.kpis.pnl_total:.2f}")
```

## 📋 Configuration

### Simulation Modes

- **Replay**: Historical match events + market snapshots
- **Monte Carlo**: Synthetic event generation using models
- **Walk Forward**: Rolling window backtesting
- **Paper Trading**: Live-like execution in simulation

### Strategy Parameters

```python
# Edge Kelly Strategy
strategy_params = {
    "edge_threshold": 0.02,     # Minimum edge to trade
    "kelly_fraction": 0.25,     # Fractional Kelly sizing
    "max_stake_pct": 0.05,      # Max 5% of bankroll per bet
    "min_odds": 1.1,            # Minimum acceptable odds
    "max_odds": 10.0            # Maximum acceptable odds
}

# Risk Profile
risk_profile = RiskProfile(
    bankroll=100000.0,          # Total bankroll
    max_exposure_pct=5.0,       # Max 5% total exposure
    per_market_cap_pct=2.0,     # Max 2% per market
    per_bet_cap_pct=0.5         # Max 0.5% per bet
)
```

## 🛡️ Governance & Risk Management

The SIM integrates with the Deterministic Governance Layer (DGL) to enforce:

- **Bankroll Limits**: Total exposure, per-market, per-bet caps
- **P&L Protection**: Daily loss limits, consecutive loss tracking
- **Position Limits**: Concentration, correlation, liquidity constraints
- **Operational Controls**: Rate limiting, circuit breakers

```python
# DGL automatically enforces rules
dgl_response = dgl.evaluate_action(action, account_state)

if dgl_response.decision == DGLDecision.REJECT:
    print(f"Action rejected: {dgl_response.reason}")
elif dgl_response.decision == DGLDecision.AMEND:
    action.size = dgl_response.amended_size
```

## 📊 Performance Metrics

### Core KPIs

- **P&L Metrics**: Total, realized, unrealized
- **Risk-Adjusted Returns**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Drawdown Analysis**: Maximum drawdown, recovery time
- **Trading Metrics**: Hit rate, average edge, fill rate, slippage
- **Operational Metrics**: Exposure utilization, turnover, trade count

### Phase Analysis

Performance breakdown by cricket match phases:
- **Powerplay** (overs 1-6)
- **Middle** (overs 7-15) 
- **Death** (overs 16-20)

## 🔧 Built-in Strategies

### 1. Edge Kelly (`edge_kelly_v3`)
- Bets when `model_prob - implied_prob ≥ threshold`
- Uses fractional Kelly criterion for sizing
- Integrates with DGL for position limits

### 2. Mean Revert LOB (`mean_revert_lob`)
- Fades microstructure dislocations
- Targets price reversions within overround bounds
- Configurable reversion thresholds

### 3. Momentum Follow (`momentum_follow`)
- Follows price trends and cricket events
- Amplifies sizing on boundaries/wickets
- Event-driven position adjustments

## 🧪 Testing

### Run All Tests
```bash
python sim/run_tests.py --coverage
```

### Test Categories
```bash
# Unit tests only
python sim/run_tests.py --unit

# Integration tests only  
python sim/run_tests.py --integration

# Fast tests (excludes slow integration)
python sim/run_tests.py --fast
```

### Test Coverage
- **Matching Engine**: Price-time priority, partial fills, latency
- **Strategy Logic**: All built-in strategies with edge cases
- **DGL Enforcement**: All governance rules and violations
- **Metrics Calculation**: KPIs, trade tracking, time series
- **End-to-End**: Complete simulation flows

## 📁 Examples

### Basic Usage
```bash
python sim/examples/basic_replay.py
```

### Strategy Comparison
```bash
python sim/examples/strategy_comparison.py
```

### Custom Configuration
```python
from sim.config import SimulationConfig, SimulationMode

config = SimulationConfig(
    id="custom_simulation",
    mode=SimulationMode.REPLAY,
    markets=["match_odds", "innings1_total"],
    match_ids=["match_1", "match_2"],
    strategy=StrategyParams("edge_kelly_v3", {
        "edge_threshold": 0.025,
        "kelly_fraction": 0.2
    }),
    seed=12345
)
```

## 🎨 UI Integration

### Streamlit Dashboard
```python
from sim.ui_streamlit import render_simulator_tab

# Add to existing Streamlit app
render_simulator_tab()
```

### Features
- **Configuration Builder**: Visual parameter setting
- **Real-time Monitoring**: Progress, P&L, exposure charts
- **Results Analysis**: KPI dashboard, trade breakdown
- **Export Options**: CSV, HTML reports, artifact download

## 📈 Performance Targets

- **Replay Speed**: ≥50× realtime for historical data
- **Synthetic Generation**: ≥5k balls/minute on laptop
- **Memory Usage**: <100MB for typical simulations
- **Reproducibility**: Identical results with same seed/config

## 🔍 Debugging & Monitoring

### Audit Trail
Every simulation generates:
- **Order Log**: All strategy actions with timestamps
- **Fill Log**: Execution details with slippage/fees
- **DGL Log**: Risk decisions with reason codes
- **State Log**: Match/market state progression

### Metrics Export
```python
# Time series data
time_series = metrics.export_time_series()

# Trade analysis
trade_analysis = metrics.get_trade_analysis()

# Performance snapshots
snapshots = metrics.get_performance_snapshots()
```

## 🛠️ Extending the System

### Custom Strategies
```python
class MyStrategy:
    def on_tick(self, match_state, market_state, account_state):
        actions = []
        
        # Your strategy logic here
        if should_trade():
            action = StrategyAction(
                ts=datetime.now().isoformat(),
                type=OrderType.LIMIT,
                side=OrderSide.BACK,
                market_id="match_odds",
                selection_id="home",
                price=2.0,
                size=100.0,
                client_order_id="my_order_1"
            )
            actions.append(action)
        
        return actions
    
    def get_name(self):
        return "my_custom_strategy"
```

### Custom Adapters
```python
class MyAdapter(EnvironmentAdapter):
    def initialize(self, config):
        # Setup your data source
        return True
    
    def get_events(self):
        # Yield (MatchState, MarketState) tuples
        for event in my_data_source:
            yield self.process_event(event)
```

## 📚 API Reference

### Core Classes
- `SimulationConfig`: Complete simulation configuration
- `SimOrchestrator`: Main simulation controller
- `MatchingEngine`: LOB execution with latency/slippage
- `SimDGLAdapter`: Risk enforcement in simulation mode
- `SimMetrics`: KPI calculation and tracking

### Data Structures
- `MatchEvent`: Ball-level cricket events
- `MarketSnapshot`: Exchange-style order book data
- `StrategyAction`: Trading decisions from strategies
- `FillEvent`: Order execution results
- `AccountState`: P&L and position tracking

## 🔗 Integration Points

- **DGL Service**: Same risk rules as production
- **Model APIs**: Score/win predictions via existing interfaces
- **Data Sources**: Cricsheet, exchange feeds via adapters
- **Logging**: Structured logs with correlation IDs

## 📄 License

Part of the WicketWise Cricket Intelligence Platform.

---

**Ready to simulate your cricket betting strategies with confidence!** 🏏💰
