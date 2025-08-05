// WicketWise Cricket AI - Interactive JavaScript
// Author: Assistant, Last Modified: 2024-12-19

class WicketWiseApp {
    constructor() {
        this.currentPage = 'dashboard';
        this.init();
    }

    init() {
        this.setupNavigation();
        this.setupAnimations();
        this.startDataUpdates();
        this.setupInteractions();
    }

    setupNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        
        navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                // Remove active class from all items
                navItems.forEach(nav => nav.classList.remove('active'));
                
                // Add active class to clicked item
                e.target.classList.add('active');
                
                // Get the page from data attribute
                const page = e.target.getAttribute('data-page');
                this.switchPage(page);
            });
        });
    }

    switchPage(page) {
        this.currentPage = page;
        
        // Add page switching animation
        const mainContent = document.querySelector('.main-content');
        mainContent.style.opacity = '0.7';
        mainContent.style.transform = 'translateY(10px)';
        
        setTimeout(() => {
            // Here you would load different content based on the page
            this.loadPageContent(page);
            
            mainContent.style.opacity = '1';
            mainContent.style.transform = 'translateY(0)';
        }, 200);
    }

    loadPageContent(page) {
        // This would typically load different content for each page
        console.log(`Loading page: ${page}`);
        
        // Example: Update page title or content based on selection
        const sections = document.querySelectorAll('.main-content section');
        sections.forEach(section => {
            section.style.display = 'block';
        });
        
        // You could hide/show different sections based on the page
        switch(page) {
            case 'live-match':
                this.focusOnLiveMatch();
                break;
            case 'predictions':
                this.focusOnPredictions();
                break;
            case 'analytics':
                this.focusOnAnalytics();
                break;
            case 'betting':
                this.focusOnBetting();
                break;
            default:
                // Dashboard shows everything
                break;
        }
    }

    focusOnLiveMatch() {
        // Highlight live match sections
        const matchStatus = document.querySelector('.match-status');
        const statsSection = document.querySelector('.stats-section');
        
        matchStatus.style.order = '1';
        statsSection.style.order = '2';
    }

    focusOnPredictions() {
        // Highlight prediction-related content
        const predictionCard = document.querySelector('.prediction-card');
        const probabilitySection = document.querySelector('.probability-section');
        
        predictionCard.style.boxShadow = '0 15px 50px rgba(74, 144, 226, 0.4)';
        probabilitySection.style.order = '1';
    }

    focusOnAnalytics() {
        // Highlight player analytics
        const playersSection = document.querySelector('.players-section');
        playersSection.style.order = '1';
    }

    focusOnBetting() {
        // Highlight betting insights
        const bettingCard = document.querySelector('.insight-card.betting');
        if (bettingCard) {
            bettingCard.style.boxShadow = '0 15px 50px rgba(80, 200, 120, 0.4)';
        }
    }

    setupAnimations() {
        // Animate cards on scroll
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, observerOptions);

        // Observe all cards and sections
        const animatedElements = document.querySelectorAll('.card, .stat-item, .player-card, .insight-card');
        animatedElements.forEach(el => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(20px)';
            el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            observer.observe(el);
        });

        // Animate probability bar
        setTimeout(() => {
            const probabilityFill = document.querySelector('.probability-fill');
            if (probabilityFill) {
                probabilityFill.style.width = '68%';
            }
        }, 1000);
    }

    setupInteractions() {
        // Add click handlers for interactive elements
        const ctaButton = document.querySelector('.cta-button');
        if (ctaButton) {
            ctaButton.addEventListener('click', this.startAnalyzing.bind(this));
        }

        // Add hover effects for cards
        const cards = document.querySelectorAll('.card, .player-card, .insight-card');
        cards.forEach(card => {
            card.addEventListener('mouseenter', (e) => {
                e.target.style.transform = 'translateY(-8px) scale(1.02)';
            });
            
            card.addEventListener('mouseleave', (e) => {
                e.target.style.transform = 'translateY(0) scale(1)';
            });
        });

        // Add click effects for stat items
        const statItems = document.querySelectorAll('.stat-item');
        statItems.forEach(item => {
            item.addEventListener('click', (e) => {
                // Animate clicked stat
                e.target.style.transform = 'scale(0.95)';
                setTimeout(() => {
                    e.target.style.transform = 'scale(1)';
                }, 150);
                
                // Show detailed info (you could expand this)
                this.showStatDetails(e.target);
            });
        });
    }

    startAnalyzing() {
        // Animate the CTA button
        const button = document.querySelector('.cta-button');
        button.textContent = 'Analyzing...';
        button.style.background = 'linear-gradient(45deg, #50C878, #4A90E2)';
        
        // Simulate analysis process
        setTimeout(() => {
            button.textContent = 'Analysis Complete!';
            button.style.background = 'linear-gradient(45deg, #50C878, #10B981)';
            
            // Show some visual feedback
            this.showAnalysisResults();
        }, 2000);
        
        setTimeout(() => {
            button.textContent = 'Start Analyzing';
            button.style.background = 'linear-gradient(45deg, #4A90E2, #50C878)';
        }, 4000);
    }

    showAnalysisResults() {
        // Add a success indicator
        const heroSection = document.querySelector('.hero-section');
        const successMsg = document.createElement('div');
        successMsg.innerHTML = '✅ Analysis Complete! Updated predictions available below.';
        successMsg.style.cssText = `
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(16, 185, 129, 0.2);
            color: #10B981;
            padding: 12px 24px;
            border-radius: 25px;
            border: 1px solid rgba(16, 185, 129, 0.3);
            font-weight: 500;
            animation: slideInDown 0.5s ease;
        `;
        
        heroSection.appendChild(successMsg);
        
        // Remove the message after 3 seconds
        setTimeout(() => {
            successMsg.remove();
        }, 3000);
        
        // Update some stats to show "new" data
        this.updateStats();
    }

    updateStats() {
        const statValues = document.querySelectorAll('.stat-value');
        const probabilityMain = document.querySelector('.probability-main');
        const wicketProb = document.querySelector('.wicket-prob');
        
        // Simulate slight changes in stats
        if (probabilityMain) {
            probabilityMain.textContent = '71%';
            probabilityMain.style.color = '#50C878';
        }
        
        if (wicketProb) {
            wicketProb.textContent = '19%';
        }
        
        // Update probability bar
        const probabilityFill = document.querySelector('.probability-fill');
        if (probabilityFill) {
            probabilityFill.style.width = '71%';
        }
        
        // Update labels
        const teamIndia = document.querySelector('.team-india');
        const teamAustralia = document.querySelector('.team-australia');
        if (teamIndia) teamIndia.textContent = 'India: 71%';
        if (teamAustralia) teamAustralia.textContent = 'Australia: 29%';
    }

    showStatDetails(statElement) {
        const statLabel = statElement.querySelector('.stat-label').textContent;
        const statValue = statElement.querySelector('.stat-value').textContent;
        
        // Create a tooltip or modal with more details
        const tooltip = document.createElement('div');
        tooltip.innerHTML = `
            <h4>${statLabel}</h4>
            <p>Current: ${statValue}</p>
            <p>Previous: ${this.getPreviousValue(statLabel)}</p>
            <p>Trend: ${this.getTrend(statLabel)}</p>
        `;
        tooltip.style.cssText = `
            position: absolute;
            background: rgba(26, 35, 50, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            padding: 16px;
            top: -100px;
            left: 50%;
            transform: translateX(-50%);
            min-width: 200px;
            z-index: 1000;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        `;
        
        statElement.style.position = 'relative';
        statElement.appendChild(tooltip);
        
        // Remove tooltip after 3 seconds
        setTimeout(() => {
            tooltip.remove();
        }, 3000);
    }

    getPreviousValue(statLabel) {
        // Mock previous values
        const mockData = {
            'Run Rate': '3.45',
            'Strike Rate': '138.2',
            'Boundaries': '21',
            'Sixes': '6',
            'Partnership': '123',
            'Target': '387'
        };
        return mockData[statLabel] || '—';
    }

    getTrend(statLabel) {
        // Mock trend data
        const trends = ['📈 Increasing', '📉 Decreasing', '➡️ Steady'];
        return trends[Math.floor(Math.random() * trends.length)];
    }

    startDataUpdates() {
        // Simulate live data updates every 30 seconds
        setInterval(() => {
            this.updateLiveData();
        }, 30000);
    }

    updateLiveData() {
        // Simulate real-time updates
        const elements = [
            { selector: '.match-score', content: 'India: 295/4 (80 overs)' },
            { selector: '.stat-value', index: 0, content: '3.69' }, // Run rate
            { selector: '.stat-value', index: 1, content: '144.1' }, // Strike rate
        ];
        
        elements.forEach(el => {
            const element = el.index !== undefined 
                ? document.querySelectorAll(el.selector)[el.index]
                : document.querySelector(el.selector);
                
            if (element) {
                element.textContent = el.content;
                // Add a subtle flash effect
                element.style.background = 'rgba(74, 144, 226, 0.2)';
                setTimeout(() => {
                    element.style.background = '';
                }, 1000);
            }
        });
    }
}

// Add some CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInDown {
        from {
            opacity: 0;
            transform: translateX(-50%) translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.7;
        }
    }
    
    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }
`;
document.head.appendChild(style);

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new WicketWiseApp();
});

// Add some global utility functions
window.startAnalyzing = function() {
    const app = new WicketWiseApp();
    app.startAnalyzing();
};