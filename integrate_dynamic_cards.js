// Purpose: Integration script to replace static cards with dynamic ones
// Author: WicketWise Team, Last Modified: August 19, 2024

/**
 * Dynamic Player Cards Integration for WicketWise Dashboard
 * 
 * This script provides functions to integrate the dynamic player card system
 * with the existing wicketwise_dashboard.html
 */

const DYNAMIC_CARDS_API = 'http://127.0.0.1:5003/api/cards';

class DynamicCardsIntegration {
    constructor() {
        this.cache = new Map();
        this.currentPersona = 'betting';
    }

    /**
     * Initialize dynamic cards system
     */
    async initialize() {
        try {
            // Check if API is available
            const health = await this.checkHealth();
            if (health.success) {
                console.log('🎴 Dynamic Cards System initialized:', health);
                return true;
            }
        } catch (error) {
            console.warn('⚠️ Dynamic Cards API not available, using fallback');
        }
        return false;
    }

    /**
     * Check dynamic cards API health
     */
    async checkHealth() {
        const response = await fetch(`${DYNAMIC_CARDS_API}/health`);
        return await response.json();
    }

    /**
     * Generate dynamic player card
     */
    async generatePlayerCard(playerName, persona = null) {
        const selectedPersona = persona || this.currentPersona;
        
        try {
            // Check cache first
            const cacheKey = `${playerName}_${selectedPersona}`;
            if (this.cache.has(cacheKey)) {
                const cached = this.cache.get(cacheKey);
                // Use cached data if less than 5 minutes old
                if (Date.now() - cached.timestamp < 5 * 60 * 1000) {
                    console.log('📦 Using cached card for', playerName);
                    return cached.data;
                }
            }

            console.log('🎴 Generating dynamic card for', playerName, 'persona:', selectedPersona);
            
            const response = await fetch(`${DYNAMIC_CARDS_API}/generate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    player_name: playerName,
                    persona: selectedPersona
                })
            });

            const result = await response.json();
            
            if (result.success) {
                // Cache the result
                this.cache.set(cacheKey, {
                    data: result.card_data,
                    timestamp: Date.now()
                });
                
                return result.card_data;
            } else {
                throw new Error(result.error || 'Failed to generate card');
            }

        } catch (error) {
            console.error('Error generating dynamic card:', error);
            return this.generateFallbackCard(playerName);
        }
    }

    /**
     * Search for players with autocomplete
     */
    async searchPlayers(query, limit = 5) {
        try {
            const response = await fetch(`${DYNAMIC_CARDS_API}/autocomplete?partial=${encodeURIComponent(query)}&limit=${limit}`);
            const result = await response.json();
            
            if (result.success) {
                return result.suggestions;
            }
        } catch (error) {
            console.error('Error searching players:', error);
        }
        return [];
    }

    /**
     * Get popular players for quick access
     */
    async getPopularPlayers() {
        try {
            const response = await fetch(`${DYNAMIC_CARDS_API}/popular`);
            const result = await response.json();
            
            if (result.success) {
                return result.popular_players;
            }
        } catch (error) {
            console.error('Error getting popular players:', error);
        }
        return [];
    }

    /**
     * Update persona for all cards
     */
    setPersona(persona) {
        this.currentPersona = persona;
        console.log('🎭 Switched to persona:', persona);
    }

    /**
     * Replace static player card with dynamic one
     */
    async replacePlayerCard(cardElement, playerName) {
        try {
            const cardData = await this.generatePlayerCard(playerName);
            if (cardData) {
                const newCardHtml = this.generateCardHTML(cardData);
                cardElement.innerHTML = newCardHtml;
                
                // Add dynamic styling based on persona
                cardElement.className = `player-card persona-${this.currentPersona} ${cardElement.className}`;
                
                console.log('✅ Replaced static card for', playerName);
                return true;
            }
        } catch (error) {
            console.error('Error replacing player card:', error);
        }
        return false;
    }

    /**
     * Generate HTML for dynamic player card
     */
    generateCardHTML(cardData) {
        return `
            <div class="dynamic-card-header">
                <div class="player-info">
                    <img src="${cardData.profile_image_url}" 
                         alt="${cardData.player_name}"
                         class="player-avatar"
                         onerror="this.src='https://via.placeholder.com/60x60?text=${cardData.player_name.replace(' ', '+')}&bg=1f2937&color=ffffff'">
                    <div>
                        <h3 class="player-name">${cardData.player_name}</h3>
                        <p class="player-form">${cardData.recent_form} (${cardData.form_rating}/10)</p>
                    </div>
                </div>
                <div class="player-stats">
                    <div class="stat">
                        <span class="stat-label">Avg</span>
                        <span class="stat-value">${cardData.batting_avg}</span>
                    </div>
                    <div class="stat">
                        <span class="stat-label">SR</span>
                        <span class="stat-value">${cardData.strike_rate}</span>
                    </div>
                </div>
            </div>
            
            <div class="dynamic-card-body">
                <div class="situational-stats">
                    <div class="stat-row">
                        <span>Powerplay SR:</span>
                        <span>${cardData.powerplay_sr}</span>
                    </div>
                    <div class="stat-row">
                        <span>Death Overs SR:</span>
                        <span>${cardData.death_overs_sr}</span>
                    </div>
                    <div class="stat-row">
                        <span>vs Pace:</span>
                        <span>${cardData.vs_pace_avg}</span>
                    </div>
                    <div class="stat-row">
                        <span>Pressure Rating:</span>
                        <span>${cardData.pressure_rating}/10</span>
                    </div>
                </div>
                
                ${cardData.current_match_status !== 'Not Playing' ? `
                <div class="live-data">
                    <div class="live-indicator">🔴 Live: ${cardData.current_match_status}</div>
                    ${cardData.last_6_balls.length > 0 ? `
                    <div class="last-6-balls">
                        ${cardData.last_6_balls.map(ball => `
                            <span class="ball ${this.getBallClass(ball)}">${ball}</span>
                        `).join('')}
                    </div>
                    ` : ''}
                </div>
                ` : ''}
                
                ${this.currentPersona === 'betting' && cardData.value_opportunities.length > 0 ? `
                <div class="betting-intelligence">
                    <h4>🎰 Value Opportunities</h4>
                    ${cardData.value_opportunities.map(opp => `
                        <div class="opportunity">
                            <div class="opportunity-title">${opp.market}</div>
                            <div class="opportunity-ev ${opp.expected_value > 0 ? 'positive' : 'negative'}">
                                EV: ${opp.expected_value > 0 ? '+' : ''}${opp.expected_value}%
                            </div>
                            <div class="opportunity-confidence">Confidence: ${opp.confidence}%</div>
                        </div>
                    `).join('')}
                </div>
                ` : ''}
            </div>
            
            <div class="dynamic-card-footer">
                <div class="last-updated">Updated: ${new Date(cardData.last_updated).toLocaleTimeString()}</div>
                <div class="data-sources">${cardData.data_sources.join(', ')}</div>
            </div>
        `;
    }

    /**
     * Get CSS class for ball outcome
     */
    getBallClass(ball) {
        switch(ball) {
            case '4': return 'four';
            case '6': return 'six';
            case 'W': return 'wicket';
            case '0': case '.': return 'dot';
            default: return 'runs';
        }
    }

    /**
     * Generate fallback card when dynamic system unavailable
     */
    generateFallbackCard(playerName) {
        return {
            player_name: playerName,
            batting_avg: 35.0,
            strike_rate: 125.0,
            recent_form: "Unknown",
            form_rating: 7.0,
            powerplay_sr: 120.0,
            death_overs_sr: 140.0,
            vs_pace_avg: 33.0,
            vs_spin_avg: 30.0,
            pressure_rating: 7.0,
            current_match_status: "Not Playing",
            last_6_balls: [],
            value_opportunities: [],
            profile_image_url: `https://via.placeholder.com/60x60?text=${playerName.replace(' ', '+')}&bg=1f2937&color=ffffff`,
            last_updated: new Date().toISOString(),
            data_sources: ['Fallback']
        };
    }

    /**
     * Add autocomplete to search inputs
     */
    addAutocompleteToInput(inputElement, onSelect = null) {
        let autocompleteTimeout = null;
        let autocompleteContainer = null;

        // Create autocomplete container
        autocompleteContainer = document.createElement('div');
        autocompleteContainer.className = 'autocomplete-container';
        autocompleteContainer.style.cssText = `
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: white;
            border: 1px solid #ccc;
            border-radius: 4px;
            max-height: 200px;
            overflow-y: auto;
            z-index: 1000;
            display: none;
        `;
        
        // Position relative to input
        inputElement.parentElement.style.position = 'relative';
        inputElement.parentElement.appendChild(autocompleteContainer);

        // Add input listener
        inputElement.addEventListener('input', (e) => {
            const query = e.target.value.trim();
            
            if (query.length < 2) {
                autocompleteContainer.style.display = 'none';
                return;
            }

            // Debounce the search
            clearTimeout(autocompleteTimeout);
            autocompleteTimeout = setTimeout(async () => {
                const suggestions = await this.searchPlayers(query, 5);
                this.displayAutocompleteSuggestions(autocompleteContainer, suggestions, (suggestion) => {
                    inputElement.value = suggestion;
                    autocompleteContainer.style.display = 'none';
                    if (onSelect) onSelect(suggestion);
                });
            }, 300);
        });

        // Hide on blur
        inputElement.addEventListener('blur', () => {
            setTimeout(() => {
                autocompleteContainer.style.display = 'none';
            }, 150);
        });
    }

    /**
     * Display autocomplete suggestions
     */
    displayAutocompleteSuggestions(container, suggestions, onSelect) {
        if (suggestions.length === 0) {
            container.style.display = 'none';
            return;
        }

        container.innerHTML = suggestions.map(suggestion => `
            <div class="autocomplete-item" style="padding: 8px 12px; cursor: pointer; border-bottom: 1px solid #eee;">
                ${suggestion}
            </div>
        `).join('');

        // Add click handlers
        container.querySelectorAll('.autocomplete-item').forEach((item, index) => {
            item.addEventListener('click', () => onSelect(suggestions[index]));
            item.addEventListener('mouseover', () => {
                item.style.backgroundColor = '#f5f5f5';
            });
            item.addEventListener('mouseout', () => {
                item.style.backgroundColor = 'white';
            });
        });

        container.style.display = 'block';
    }
}

// Global instance for easy access
window.dynamicCards = new DynamicCardsIntegration();

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dynamicCards.initialize();
});

// Example integration functions for wicketwise_dashboard.html
window.integrateWithMainDashboard = function() {
    // Replace Virat Kohli static card with dynamic one
    const viratCard = document.querySelector('[data-player="virat-kohli"]');
    if (viratCard) {
        window.dynamicCards.replacePlayerCard(viratCard, 'Virat Kohli');
    }

    // Add autocomplete to intelligence engine search
    const searchInput = document.getElementById('kg-query-input');
    if (searchInput) {
        window.dynamicCards.addAutocompleteToInput(searchInput, (playerName) => {
            console.log('Selected player:', playerName);
        });
    }

    // Update persona switching
    const personaBtns = document.querySelectorAll('.persona-btn');
    personaBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            const persona = e.target.dataset.persona;
            window.dynamicCards.setPersona(persona);
            
            // Refresh all dynamic cards
            document.querySelectorAll('.dynamic-card').forEach(card => {
                const playerName = card.dataset.playerName;
                if (playerName) {
                    window.dynamicCards.replacePlayerCard(card, playerName);
                }
            });
        });
    });
};

console.log('🎴 Dynamic Cards Integration loaded. Call integrateWithMainDashboard() to integrate.');
