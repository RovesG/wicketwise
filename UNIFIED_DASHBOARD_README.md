# WicketWise Unified Dashboard

## 🎯 Overview

The WicketWise Unified Dashboard is a modern, tabbed interface that consolidates all system functionality into a single, easy-to-navigate screen. No more jumping between multiple HTML files!

## 🚀 Quick Start

```bash
./start.sh
```

The system will automatically open the unified dashboard at:
`http://localhost:8000/wicketwise_unified_dashboard.html`

## 📋 Features

### 🏠 **Dashboard Tab**
- **System Status**: Real-time monitoring of API, KG, and Model status
- **Data Statistics**: Live stats on matches, balls, and venues
- **Quick Actions**: One-click access to key features

### ⚙️ **Admin Tab**
- **Model Training**: Start T20 model training with progress tracking
- **Knowledge Graph**: Build and manage the cricket knowledge graph
- **Data Enrichment**: Enrich match data with weather and context

### 📈 **Betting Intelligence Tab**
- **Market Analysis**: Active markets and mispricing opportunities
- **DGL Controls**: Risk tolerance and stake management
- **Recent Predictions**: Performance tracking and ROI

### 💬 **AI Chat Tab**
- **Interactive Chat**: Ask questions about cricket analytics
- **Knowledge Graph Integration**: Powered by the cricket KG
- **Real-time Responses**: Instant AI assistance

### 🎮 **Simulation Tab**
- **Match Selection**: Choose from holdout matches
- **Simulation Modes**: Rapid, detailed, or slow-motion analysis
- **Results Display**: Win probabilities and confidence scores

### 📊 **Analytics Tab**
- **Performance Metrics**: Advanced analytics and visualizations
- **Historical Data**: Trends and patterns analysis

## 🎨 Design Features

### Modern UI/UX
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Cricket-themed Colors**: Professional color scheme
- **Intuitive Navigation**: Tab-based interface with clear icons
- **Real-time Updates**: Live status indicators and progress bars

### Accessibility
- **Keyboard Navigation**: Full tab support
- **Screen Reader Friendly**: Proper ARIA labels
- **High Contrast**: 4.5:1 minimum contrast ratio
- **Mobile Optimized**: Touch-friendly interface

## 🔧 Technical Details

### Architecture
- **Single Page Application**: No page reloads
- **RESTful API Integration**: Real-time data from Flask backend
- **Modular JavaScript**: Clean, maintainable code structure
- **Progressive Enhancement**: Works without JavaScript for basic features

### API Endpoints Used
- `GET /api/training-pipeline/stats` - System statistics
- `POST /api/train-model` - Start model training
- `POST /api/build-knowledge-graph` - Build KG
- `POST /api/kg-chat` - AI chat functionality
- `GET /api/simulation/holdout-matches` - Simulation data
- `POST /api/simulation/run` - Run simulations

### Browser Support
- **Modern Browsers**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **JavaScript Required**: For full functionality
- **CSS Grid Support**: For responsive layouts

## 🔄 Migration from Old Screens

### Replaced Files
- `wicketwise_dashboard.html` → Unified Dashboard Tab
- `wicketwise_admin_redesigned.html` → Admin Tab
- `betting_player_ui.html` → Betting Intelligence Tab
- `wicketwise_agent_dashboard.html` → AI Chat Tab

### Legacy Access
Old screens are still available but deprecated:
- Legacy Admin: `http://localhost:8000/wicketwise_admin_redesigned.html`

## 🛠️ Customization

### Adding New Tabs
1. Add tab button in navigation
2. Create tab content div
3. Implement `load[TabName]Data()` function
4. Add tab switching logic

### Styling
- CSS variables in `:root` for easy theming
- Modular CSS classes for components
- Responsive breakpoints for mobile

## 🐛 Troubleshooting

### Common Issues
1. **Blank Dashboard**: Check if backend is running on port 5001
2. **API Errors**: Verify all services are started with `./start.sh status`
3. **Chat Not Working**: Ensure KG is built and OpenAI API key is set
4. **Simulation Errors**: Check holdout matches are available

### Debug Mode
Open browser developer tools (F12) to see console logs and network requests.

## 📈 Performance

### Optimizations
- **Lazy Loading**: Tab content loaded on demand
- **API Caching**: Reduced redundant requests
- **Efficient DOM Updates**: Minimal reflows and repaints
- **Progressive Enhancement**: Core functionality works without JavaScript

### Monitoring
- Real-time status indicators
- Performance metrics in console
- Error handling and user feedback

## 🔮 Future Enhancements

### Planned Features
- **Real-time Notifications**: WebSocket integration for live updates
- **Advanced Visualizations**: Charts and graphs for analytics
- **User Preferences**: Customizable dashboard layouts
- **Offline Support**: Service worker for offline functionality
- **Dark Mode**: Theme switching capability

### Integration Opportunities
- **Mobile App**: PWA conversion
- **Desktop App**: Electron wrapper
- **API Extensions**: Additional backend endpoints
- **Third-party Services**: External data sources

---

**Built with ❤️ for Cricket Intelligence**
