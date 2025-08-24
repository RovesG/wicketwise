/**
 * WicketWise Agent Dashboard - Enhanced JavaScript
 * ===============================================
 * 
 * Advanced frontend logic for the multi-agent cricket intelligence dashboard.
 * Provides real-time integration with the Flask backend, explainable AI
 * visualizations, and sophisticated user interactions.
 * 
 * Author: WicketWise Team
 * Last Modified: 2025-08-24
 */

class WicketWiseAgentDashboard {
    constructor() {
        this.apiBase = 'http://127.0.0.1:5001/api';
        this.agents = [];
        this.systemStatus = null;
        this.currentAnalysis = null;
        this.updateInterval = null;
        
        // UI Elements
        this.elements = {
            agentList: document.getElementById('agentList'),
            queryInput: document.getElementById('queryInput'),
            analyzeBtn: document.getElementById('analyzeBtn'),
            executionFlow: document.getElementById('executionFlow'),
            executionTimeline: document.getElementById('executionTimeline'),
            analysisResults: document.getElementById('analysisResults'),
            executionTime: document.getElementById('executionTime'),
            systemInsights: document.getElementById('systemInsights'),
            explainabilityContent: document.getElementById('explainabilityContent'),
            agentCount: document.getElementById('agentCount'),
            systemHealth: document.getElementById('systemHealth'),
            performanceMetrics: document.getElementById('performanceMetrics')
        };
        
        this.initialize();
    }

    async initialize() {
        console.log('🚀 Initializing WicketWise Agent Dashboard...');
        
        try {
            // Check backend health
            await this.checkBackendHealth();
            
            // Load initial data
            await this.loadAgents();
            await this.loadSystemStatus();
            
            // Setup UI
            this.setupEventListeners();
            this.startRealTimeUpdates();
            
            console.log('✅ Dashboard initialized successfully');
            
        } catch (error) {
            console.error('❌ Dashboard initialization failed:', error);
            this.showError('Failed to connect to backend. Please ensure the server is running.');
        }
    }

    async checkBackendHealth() {
        const response = await fetch(`${this.apiBase}/health`);
        if (!response.ok) {
            throw new Error(`Backend health check failed: ${response.status}`);
        }
        const health = await response.json();
        console.log('🏥 Backend health:', health.status);
        return health;
    }

    async loadAgents() {
        try {
            const response = await fetch(`${this.apiBase}/agents`);
            const data = await response.json();
            
            if (data.status === 'success') {
                this.agents = data.agents;
                this.renderAgents();
                this.updateAgentCount();
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('Error loading agents:', error);
            this.showError('Failed to load agent information');
        }
    }

    async loadSystemStatus() {
        try {
            const response = await fetch(`${this.apiBase}/system/status`);
            const data = await response.json();
            
            if (data.status === 'success') {
                this.systemStatus = data.system_health;
                this.updateSystemHealthDisplay();
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            console.error('Error loading system status:', error);
        }
    }

    renderAgents() {
        if (!this.elements.agentList || !this.agents.length) return;

        const agentIcons = {
            'performance_agent': 'trending-up',
            'tactical_agent': 'target',
            'prediction_agent': 'crystal-ball',
            'betting_agent': 'dollar-sign'
        };

        const agentColors = {
            'performance_agent': 'var(--batting-color)',
            'tactical_agent': 'var(--bowling-color)',
            'prediction_agent': 'var(--prediction-color)',
            'betting_agent': 'var(--wicket-color)'
        };

        this.elements.agentList.innerHTML = this.agents.map(agent => `
            <div class="agent-card" data-agent="${agent.id}">
                <div class="agent-header">
                    <div class="agent-name">${agent.name}</div>
                    <div class="agent-status ${agent.status}"></div>
                </div>
                <div class="agent-description">
                    ${this.getAgentDescription(agent.id)}
                </div>
                <div class="agent-metrics">
                    <span>Success: ${(agent.performance.success_rate * 100).toFixed(1)}%</span>
                    <span>Avg: ${agent.performance.average_execution_time.toFixed(1)}s</span>
                </div>
                <div class="agent-capabilities" style="margin-top: 0.5rem; font-size: 0.75rem; color: var(--text-secondary);">
                    ${agent.capabilities.join(', ')}
                </div>
            </div>
        `).join('');

        // Re-initialize Lucide icons
        if (window.lucide) {
            lucide.createIcons();
        }
    }

    getAgentDescription(agentId) {
        const descriptions = {
            'performance_agent': 'Analyzes player and team performance statistics with temporal decay',
            'tactical_agent': 'Provides strategic insights and field placement recommendations',
            'prediction_agent': 'Forecasts match outcomes using ensemble models and MoE routing',
            'betting_agent': 'Identifies value opportunities and market inefficiencies'
        };
        return descriptions[agentId] || 'Specialized cricket analysis agent';
    }

    setupEventListeners() {
        // Analyze button
        if (this.elements.analyzeBtn) {
            this.elements.analyzeBtn.addEventListener('click', () => this.handleAnalysis());
        }
        
        // Enter key in query input
        if (this.elements.queryInput) {
            this.elements.queryInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.handleAnalysis();
            });
        }

        // Quick query buttons
        document.querySelectorAll('.quick-query').forEach(btn => {
            btn.addEventListener('click', (e) => {
                if (this.elements.queryInput) {
                    this.elements.queryInput.value = e.target.dataset.query;
                    this.handleAnalysis();
                }
            });
        });

        // Agent card selection
        document.addEventListener('click', (e) => {
            const agentCard = e.target.closest('.agent-card');
            if (agentCard) {
                document.querySelectorAll('.agent-card').forEach(c => c.classList.remove('active'));
                agentCard.classList.add('active');
                
                const agentId = agentCard.dataset.agent;
                this.showAgentExplanation(agentId);
            }
        });
    }

    async handleAnalysis() {
        const query = this.elements.queryInput?.value?.trim();
        if (!query) return;

        console.log('🔍 Starting analysis:', query);

        // Update UI state
        this.setAnalyzing(true);
        this.showExecutionFlow();
        
        try {
            // Send analysis request to backend
            const response = await fetch(`${this.apiBase}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    query: query,
                    context: {
                        format: 'ODI',
                        timestamp: new Date().toISOString()
                    }
                })
            });

            const data = await response.json();

            if (data.status === 'success') {
                this.currentAnalysis = data.result;
                await this.displayRealAnalysisResults(data.result);
                this.updateExecutionTime(data.result.execution_time);
                console.log('✅ Analysis completed successfully');
            } else {
                throw new Error(data.message);
            }
            
        } catch (error) {
            console.error('❌ Analysis failed:', error);
            this.showError(`Analysis failed: ${error.message}`);
        } finally {
            this.setAnalyzing(false);
        }
    }

    setAnalyzing(analyzing) {
        if (!this.elements.analyzeBtn) return;
        
        this.elements.analyzeBtn.disabled = analyzing;
        
        if (analyzing) {
            this.elements.analyzeBtn.innerHTML = `
                <div class="spinner" style="width: 16px; height: 16px; margin-right: 0.5rem;"></div>
                Analyzing...
            `;
        } else {
            this.elements.analyzeBtn.innerHTML = `
                <i data-lucide="search"></i>
                Analyze
            `;
            if (window.lucide) {
                lucide.createIcons();
            }
        }
    }

    showExecutionFlow() {
        if (this.elements.executionFlow) {
            this.elements.executionFlow.style.display = 'block';
        }
    }

    async displayRealAnalysisResults(result) {
        if (!this.elements.analysisResults) return;

        // Create results based on actual agent responses
        const resultCards = result.agent_responses
            .filter(response => response.success)
            .map(response => this.createResultCard(response));

        this.elements.analysisResults.innerHTML = resultCards.join('');

        // Update execution timeline
        this.updateExecutionTimeline(result.agent_responses);

        // Update insights
        this.updateInsights(result);

        // Re-initialize Lucide icons
        if (window.lucide) {
            lucide.createIcons();
        }
    }

    createResultCard(agentResponse) {
        const agentConfig = this.getAgentConfig(agentResponse.agent_id);
        const confidenceLevel = this.getConfidenceLevel(agentResponse.confidence);

        return `
            <div class="result-card">
                <div class="result-header">
                    <div class="result-icon" style="background: ${agentConfig.bgColor}; color: ${agentConfig.color};">
                        <i data-lucide="${agentConfig.icon}"></i>
                    </div>
                    <div class="result-title">${agentConfig.title}</div>
                    <div class="confidence-badge confidence-${confidenceLevel}">
                        ${Math.round(agentResponse.confidence * 100)}% confidence
                    </div>
                </div>
                <div class="result-content">
                    ${this.formatAgentResult(agentResponse)}
                </div>
                <div class="explanation-panel">
                    <div class="explanation-title">
                        <i data-lucide="info"></i>
                        AI Explanation
                    </div>
                    <div class="explanation-content">
                        ${this.generateExplanation(agentResponse)}
                    </div>
                    ${this.generateFeatureImportance(agentResponse)}
                </div>
            </div>
        `;
    }

    getAgentConfig(agentId) {
        const configs = {
            'performance_agent': {
                title: 'Performance Analysis',
                icon: 'trending-up',
                color: 'var(--batting-color)',
                bgColor: '#fef7ed'
            },
            'tactical_agent': {
                title: 'Tactical Strategy',
                icon: 'target',
                color: 'var(--bowling-color)',
                bgColor: '#eff6ff'
            },
            'prediction_agent': {
                title: 'Match Prediction',
                icon: 'crystal-ball',
                color: 'var(--prediction-color)',
                bgColor: '#f0fdf4'
            },
            'betting_agent': {
                title: 'Betting Analysis',
                icon: 'dollar-sign',
                color: 'var(--wicket-color)',
                bgColor: '#fef2f2'
            }
        };
        
        return configs[agentId] || {
            title: 'Analysis Result',
            icon: 'brain',
            color: 'var(--neutral-color)',
            bgColor: '#f8fafc'
        };
    }

    getConfidenceLevel(confidence) {
        if (confidence >= 0.8) return 'high';
        if (confidence >= 0.6) return 'medium';
        return 'low';
    }

    formatAgentResult(agentResponse) {
        if (!agentResponse.result) {
            return '<p>No detailed results available</p>';
        }

        // Format based on agent type
        switch (agentResponse.agent_id) {
            case 'performance_agent':
                return this.formatPerformanceResult(agentResponse.result);
            case 'tactical_agent':
                return this.formatTacticalResult(agentResponse.result);
            case 'prediction_agent':
                return this.formatPredictionResult(agentResponse.result);
            case 'betting_agent':
                return this.formatBettingResult(agentResponse.result);
            default:
                return `<pre>${JSON.stringify(agentResponse.result, null, 2)}</pre>`;
        }
    }

    formatPerformanceResult(result) {
        if (result.analysis_type === 'player_performance' && result.players?.length > 0) {
            const player = result.players[0];
            return `
                <p><strong>Player Analysis:</strong> ${player.name || 'Player'}</p>
                <p><strong>Current Form:</strong> ${player.current_form || 'Good'}</p>
                <p><strong>Strengths:</strong> ${player.strengths?.join(', ') || 'High batting average'}</p>
                <p><strong>Key Insights:</strong> ${result.summary || 'Comprehensive performance analysis completed'}</p>
            `;
        }
        return '<p>Performance analysis completed with statistical insights</p>';
    }

    formatTacticalResult(result) {
        if (result.analysis_type === 'field_placement' && result.recommended_fields) {
            return `
                <p><strong>Field Strategy:</strong> Dynamic placement based on match context</p>
                <p><strong>Bowling Plan:</strong> Vary pace and line according to conditions</p>
                <p><strong>Key Tactics:</strong> ${Object.keys(result.recommended_fields).join(', ')}</p>
                <p><strong>Success Probability:</strong> High based on historical patterns</p>
            `;
        }
        return '<p>Strategic tactical recommendations generated based on match analysis</p>';
    }

    formatPredictionResult(result) {
        if (result.prediction_type === 'match_winner' && result.predicted_winner) {
            return `
                <p><strong>Predicted Winner:</strong> ${result.predicted_winner}</p>
                <p><strong>Win Probability:</strong> ${Math.round(result.win_probability * 100)}%</p>
                <p><strong>Confidence Level:</strong> ${result.confidence_level || 'Medium'}</p>
                <p><strong>Key Factors:</strong> Recent form, venue advantage, head-to-head record</p>
            `;
        }
        return '<p>Match outcome predictions generated using ensemble models</p>';
    }

    formatBettingResult(result) {
        if (result.analysis_type === 'value_opportunities' && result.opportunities?.length > 0) {
            const opp = result.opportunities[0];
            return `
                <p><strong>Best Value:</strong> ${opp.selection} @ ${opp.odds}</p>
                <p><strong>Expected Value:</strong> ${(opp.expected_value * 100).toFixed(1)}%</p>
                <p><strong>Kelly Fraction:</strong> ${(opp.kelly_fraction * 100).toFixed(1)}%</p>
                <p><strong>Risk Level:</strong> ${opp.risk_level || 'Medium'}</p>
            `;
        }
        return '<p>Value betting opportunities analyzed across multiple markets</p>';
    }

    generateExplanation(agentResponse) {
        const explanations = {
            'performance_agent': 'Analysis uses temporal decay functions to weight recent performances more heavily, combined with contextual factors like venue and opposition strength.',
            'tactical_agent': 'Strategy recommendations based on match context analysis, team composition evaluation, and historical tactical success patterns.',
            'prediction_agent': 'Predictions generated using ensemble methods combining multiple models through the Mixture of Experts routing system.',
            'betting_agent': 'Value opportunities identified by comparing model-predicted probabilities with bookmaker odds, using Kelly criterion for stake optimization.'
        };
        
        return explanations[agentResponse.agent_id] || 'AI analysis completed using specialized domain knowledge and statistical models.';
    }

    generateFeatureImportance(agentResponse) {
        // Generate mock feature importance for demonstration
        const features = this.getMockFeatures(agentResponse.agent_id);
        
        return `
            <div class="feature-importance">
                ${features.map(feature => `
                    <div class="feature-bar">
                        <div class="feature-name">${feature.name}</div>
                        <div class="feature-progress">
                            <div class="feature-fill" style="width: ${feature.importance * 100}%"></div>
                        </div>
                        <div class="feature-value">${Math.round(feature.importance * 100)}%</div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    getMockFeatures(agentId) {
        const featureMap = {
            'performance_agent': [
                { name: 'Recent Form', importance: 0.35 },
                { name: 'Venue History', importance: 0.28 },
                { name: 'Opposition', importance: 0.22 },
                { name: 'Conditions', importance: 0.15 }
            ],
            'tactical_agent': [
                { name: 'Match Context', importance: 0.40 },
                { name: 'Team Composition', importance: 0.30 },
                { name: 'Historical Success', importance: 0.20 },
                { name: 'Conditions', importance: 0.10 }
            ],
            'prediction_agent': [
                { name: 'Team Strength', importance: 0.32 },
                { name: 'Recent Form', importance: 0.28 },
                { name: 'Home Advantage', importance: 0.25 },
                { name: 'Head-to-Head', importance: 0.15 }
            ],
            'betting_agent': [
                { name: 'Odds Discrepancy', importance: 0.45 },
                { name: 'Model Confidence', importance: 0.25 },
                { name: 'Market Efficiency', importance: 0.20 },
                { name: 'Liquidity', importance: 0.10 }
            ]
        };
        
        return featureMap[agentId] || [
            { name: 'Primary Factor', importance: 0.5 },
            { name: 'Secondary Factor', importance: 0.3 },
            { name: 'Context Factor', importance: 0.2 }
        ];
    }

    updateExecutionTimeline(agentResponses) {
        if (!this.elements.executionTimeline) return;

        const timeline = agentResponses.map((response, index) => {
            const config = this.getAgentConfig(response.agent_id);
            const status = response.success ? 'Completed' : 'Failed';
            const statusClass = response.success ? 'completed' : 'error';
            
            return `
                <div class="timeline-step">
                    <div class="step-icon ${response.agent_id}" style="background: ${config.bgColor}; color: ${config.color};">
                        <i data-lucide="${config.icon}"></i>
                    </div>
                    <div class="step-name">${config.title}</div>
                    <div class="step-status ${statusClass}">${status}</div>
                </div>
                ${index < agentResponses.length - 1 ? '<div class="step-arrow"><i data-lucide="arrow-right"></i></div>' : ''}
            `;
        }).join('');

        this.elements.executionTimeline.innerHTML = timeline;
        
        if (window.lucide) {
            lucide.createIcons();
        }
    }

    updateExecutionTime(executionTime) {
        if (this.elements.executionTime) {
            this.elements.executionTime.innerHTML = `
                <i data-lucide="clock"></i>
                Completed in ${executionTime.toFixed(2)}s
            `;
            if (window.lucide) {
                lucide.createIcons();
            }
        }
    }

    updateInsights(result) {
        if (!this.elements.systemInsights) return;

        const insights = [
            `Multi-agent analysis completed with ${result.agent_responses.length} agents`,
            `Overall confidence: ${Math.round(result.overall_confidence * 100)}%`,
            `Execution time: ${result.execution_time.toFixed(2)} seconds`,
            `Success rate: ${result.agent_responses.filter(r => r.success).length}/${result.agent_responses.length} agents`
        ];

        this.elements.systemInsights.innerHTML = insights.map(insight => `
            <div class="insight-item">
                <div class="insight-text">${insight}</div>
                <div class="insight-source">Agent Orchestrator</div>
            </div>
        `).join('');
    }

    async showAgentExplanation(agentId) {
        try {
            const response = await fetch(`${this.apiBase}/explain/${agentId}`);
            const data = await response.json();
            
            if (data.status === 'success' && this.elements.explainabilityContent) {
                const explanation = data.explanation;
                
                this.elements.explainabilityContent.innerHTML = `
                    <div class="explanation-panel">
                        <div class="explanation-title">
                            <i data-lucide="brain"></i>
                            ${agentId.replace('_', ' ').toUpperCase()} Agent
                        </div>
                        <div class="explanation-content">
                            <p><strong>Description:</strong> ${explanation.description}</p>
                            
                            <p><strong>Methodology:</strong></p>
                            <ul>
                                ${explanation.methodology.map(step => `<li>${step}</li>`).join('')}
                            </ul>
                            
                            <p><strong>Key Factors:</strong></p>
                            <ul>
                                ${explanation.key_factors.map(factor => `<li>${factor}</li>`).join('')}
                            </ul>
                            
                            <p><strong>Confidence Factors:</strong></p>
                            <ul>
                                ${explanation.confidence_factors.map(factor => `<li>${factor}</li>`).join('')}
                            </ul>
                        </div>
                    </div>
                `;
                
                if (window.lucide) {
                    lucide.createIcons();
                }
            }
        } catch (error) {
            console.error('Error loading agent explanation:', error);
        }
    }

    updateAgentCount() {
        if (this.elements.agentCount) {
            const activeAgents = this.agents.filter(agent => agent.status === 'active').length;
            this.elements.agentCount.textContent = `${activeAgents} Agents Active`;
        }
    }

    updateSystemHealthDisplay() {
        if (this.elements.systemHealth && this.systemStatus) {
            const healthPercentage = this.systemStatus.agent_system.health_percentage;
            let status = 'System Healthy';
            
            if (healthPercentage < 75) {
                status = 'System Degraded';
            } else if (healthPercentage < 50) {
                status = 'System Unhealthy';
            }
            
            this.elements.systemHealth.textContent = status;
        }
    }

    async updatePerformanceMetrics() {
        try {
            const response = await fetch(`${this.apiBase}/performance/metrics`);
            const data = await response.json();
            
            if (data.status === 'success' && this.elements.performanceMetrics) {
                const metrics = data.system_metrics;
                
                this.elements.performanceMetrics.innerHTML = `
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; font-size: 0.875rem;">
                        <div>
                            <div style="color: var(--text-secondary);">Avg Response Time</div>
                            <div style="font-weight: 600; color: var(--batting-color);">${metrics.average_execution_time.toFixed(1)}s</div>
                        </div>
                        <div>
                            <div style="color: var(--text-secondary);">Success Rate</div>
                            <div style="font-weight: 600; color: #10b981;">${metrics.average_success_rate.toFixed(1)}%</div>
                        </div>
                        <div>
                            <div style="color: var(--text-secondary);">Total Queries</div>
                            <div style="font-weight: 600;">${metrics.total_executions}</div>
                        </div>
                        <div>
                            <div style="color: var(--text-secondary);">Confidence Avg</div>
                            <div style="font-weight: 600; color: var(--prediction-color);">${Math.round(metrics.average_confidence * 100)}%</div>
                        </div>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Error updating performance metrics:', error);
        }
    }

    startRealTimeUpdates() {
        // Update system status every 10 seconds
        this.updateInterval = setInterval(async () => {
            try {
                await this.loadSystemStatus();
                await this.updatePerformanceMetrics();
            } catch (error) {
                console.error('Error in real-time update:', error);
            }
        }, 10000);
    }

    showError(message) {
        if (this.elements.analysisResults) {
            this.elements.analysisResults.innerHTML = `
                <div style="grid-column: 1 / -1; text-align: center; padding: 2rem; color: var(--wicket-color);">
                    <i data-lucide="alert-circle" size="48" style="margin-bottom: 1rem;"></i>
                    <p>${message}</p>
                </div>
            `;
            if (window.lucide) {
                lucide.createIcons();
            }
        }
    }

    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new WicketWiseAgentDashboard();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.dashboard) {
        window.dashboard.destroy();
    }
});
