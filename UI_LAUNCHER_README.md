# WicketWise UI Launcher (Deprecated)

## 🎯 Overview

Streamlit-based paths are deprecated. The current UI is a Figma-derived static page served alongside the Flask admin API.

## 🚀 Run the current UI

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash start.sh
open "http://127.0.0.1:8000/wicketwise_dashboard.html"
```

### **Common Elements:**
- **Clean UI**: Professional cricket-themed interface with 🏏 icons
- **Wide Layout**: Full-width Streamlit layout for maximum screen usage
- **Expandable Sidebar**: Collapsible sidebar for future navigation
- **Version Footer**: Shows "WicketWise version 0.1" at the bottom

## Notes
- Backend API: port 5001 (`/api/*`)
- UI: static server port 8000 (`wicketwise_dashboard.html`)

### **Development Mode**

```bash
# Run with auto-reload on file changes
streamlit run ui_launcher.py --server.runOnSave true
```

## 🏗️ **File Structure**

```
ui_launcher.py              # Main Streamlit application
tests/test_ui_launcher.py   # Comprehensive test suite
UI_LAUNCHER_README.md       # This documentation file
```

## 📊 **Technical Details**

### **Dependencies**
- `streamlit >= 1.46.0` - Web app framework
- `python >= 3.8` - Python runtime

### **Configuration**
```python
st.set_page_config(
    page_title="WicketWise Cricket AI",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### **Tab Structure**
Each tab follows the same pattern:
1. `st.header()` with tab title
2. `st.write()` with placeholder text
3. Ready for future functionality integration

## Testing
Run the overall test suite:
```bash
PYTHONPATH=. pytest -q
```

### **Test Results**
```
6 passed, 0 failed ✅
- test_ui_launcher_imports PASSED
- test_ui_launcher_structure PASSED  
- test_ui_launcher_placeholder_text PASSED
- test_ui_launcher_tab_structure PASSED
- test_ui_launcher_version_footer PASSED
- test_ui_launcher_tab_names PASSED
```

## 🔧 **Customization**

### **Adding New Tabs**
```python
# Add a new tab
tab1, tab2, tab3, tab4 = st.tabs([
    "Live Match Dashboard", 
    "Simulator Mode", 
    "Admin Panel",
    "New Tab"  # Add your new tab here
])

# Add content for the new tab
with tab4:
    st.header("New Tab")
    st.write("Your content here...")
```

### **Styling**
The UI follows WicketWise design principles:
- **Cricket Theme**: 🏏 cricket ball icon and terminology
- **Professional Look**: Clean, modern interface
- **Responsive Design**: Works on desktop and mobile
- **Consistent Branding**: WicketWise version footer

## 📱 **Browser Support**

The UI launcher works with:
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## 🛠️ **Development Guidelines**

### **Code Structure**
```python
# 1. Imports
import streamlit as st

# 2. Page configuration
st.set_page_config(...)

# 3. Main title
st.title("🏏 WicketWise Cricket AI")

# 4. Tab creation
tab1, tab2, tab3 = st.tabs([...])

# 5. Tab content
with tab1:
    st.header("Tab Name")
    st.write("Content...")

# 6. Footer
st.markdown("---")
st.markdown("**WicketWise version 0.1**")
```

### **Adding Functionality**
When adding real functionality to tabs:

1. **Keep it Simple**: Start with basic functionality
2. **Use st.columns()**: For layout organization
3. **Add st.sidebar**: For controls and filters
4. **Include Error Handling**: Use st.error() for user feedback
5. **Add Loading States**: Use st.spinner() for long operations

## 🚦 **Performance**

### **Load Time**
- **Initial Load**: < 2 seconds
- **Tab Switching**: Instant
- **Memory Usage**: < 50MB

### **Scalability**
- **Concurrent Users**: Streamlit handles multiple sessions
- **Data Volume**: Placeholder for future optimization
- **Real-time Updates**: Ready for st.rerun() integration

## 🔮 **Future Enhancements**

### **Live Match Dashboard**
- Real-time match data integration
- Live prediction updates
- Interactive charts and graphs
- Match timeline visualization

### **Simulator Mode**
- Match scenario simulation
- What-if analysis tools
- Custom team/player configurations
- Monte Carlo simulations

### **Admin Panel**
- System configuration options
- User management
- API key management
- Performance monitoring

## 📚 **Resources**

### **Documentation**
- [Streamlit Documentation](https://docs.streamlit.io/)
- [WicketWise System Architecture](COMPLETE_TRAINING_PIPELINE_SUMMARY.md)
- [Testing Guide](tests/test_ui_launcher.py)

### **Related Files**
- `ui_streamlit.py` - Alternative UI implementation
- `demo_complete_workflow.py` - Full system demonstration
- `crickformers/` - Core ML models and training

## Status
This document is kept for historical reference. Prefer the main README instructions for launching the current UI.