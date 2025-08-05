# 🎨 Figma to Wicketwise Integration Guide

## **Quick Start: 3 Methods to Import Your Figma Design**

### **Method 1: Using Figma Plugins (Fastest)**

#### Step 1: Export from Figma
1. **Install these Figma plugins:**
   - "Figma to Code" (by Builder.io) - Best for React/HTML
   - "Locofy Lightning" - Good for responsive designs  
   - "TeleportHQ" - Good for CSS extraction

2. **Export Process:**
   ```
   Figma → Select frames → Plugin → Export as HTML/CSS
   ```

3. **Save exported files to:**
   ```
   wicketwise/
   ├── ui/
   │   ├── assets/
   │   │   ├── figma_exports/
   │   │   │   ├── components.html
   │   │   │   ├── styles.css  
   │   │   │   └── assets/
   ```

#### Step 2: Convert to Streamlit
```bash
# Run the converter tool I created for you
streamlit run figma_to_streamlit_converter.py
```

### **Method 2: Manual Integration (More Control)**

#### Step 1: Extract Design Tokens
From your Figma design, extract:

```python
# Add to ui_style.py
FIGMA_DESIGN_TOKENS = {
    "colors": {
        "primary": "#4A90E2",      # Your primary color
        "secondary": "#F5A623",    # Your secondary color  
        "background": "#1A2332",   # Your background
        "surface": "#252B37",      # Card/surface color
        "text_primary": "#FFFFFF", # Primary text
        "text_secondary": "#B8BCC8" # Secondary text
    },
    "typography": {
        "font_family": "'YourFigmaFont', 'Inter', sans-serif",
        "sizes": {
            "h1": "2.5rem",
            "h2": "2rem", 
            "h3": "1.5rem",
            "body": "1rem",
            "small": "0.875rem"
        },
        "weights": {
            "light": 300,
            "regular": 400,
            "medium": 500,
            "semibold": 600,
            "bold": 700
        }
    },
    "spacing": {
        "xs": "4px",
        "sm": "8px",
        "md": "16px", 
        "lg": "24px",
        "xl": "32px",
        "xxl": "48px"
    },
    "borders": {
        "radius": {
            "sm": "8px",
            "md": "16px", 
            "lg": "24px",
            "full": "9999px"
        },
        "width": "1px"
    },
    "shadows": {
        "sm": "0 2px 4px rgba(0,0,0,0.1)",
        "md": "0 4px 12px rgba(0,0,0,0.15)",
        "lg": "0 8px 32px rgba(0,0,0,0.3)"
    }
}
```

#### Step 2: Create Figma Component Functions
Add to your `ui_style.py`:

```python
def render_figma_hero_section(title: str, subtitle: str, cta_text: str = "Get Started"):
    """Example: Convert your Figma hero section"""
    
    hero_html = f"""
    <div style="
        background: linear-gradient(135deg, {FIGMA_DESIGN_TOKENS['colors']['primary']} 0%, {FIGMA_DESIGN_TOKENS['colors']['secondary']} 100%);
        padding: {FIGMA_DESIGN_TOKENS['spacing']['xxl']} {FIGMA_DESIGN_TOKENS['spacing']['xl']};
        border-radius: {FIGMA_DESIGN_TOKENS['borders']['radius']['lg']};
        text-align: center;
        margin-bottom: {FIGMA_DESIGN_TOKENS['spacing']['xl']};
        box-shadow: {FIGMA_DESIGN_TOKENS['shadows']['lg']};
    ">
        <h1 style="
            font-family: {FIGMA_DESIGN_TOKENS['typography']['font_family']};
            font-size: {FIGMA_DESIGN_TOKENS['typography']['sizes']['h1']};
            font-weight: {FIGMA_DESIGN_TOKENS['typography']['weights']['bold']};
            color: {FIGMA_DESIGN_TOKENS['colors']['text_primary']};
            margin-bottom: {FIGMA_DESIGN_TOKENS['spacing']['md']};
        ">{title}</h1>
        
        <p style="
            font-family: {FIGMA_DESIGN_TOKENS['typography']['font_family']};
            font-size: {FIGMA_DESIGN_TOKENS['typography']['sizes']['body']};
            color: {FIGMA_DESIGN_TOKENS['colors']['text_primary']};
            opacity: 0.9;
            margin-bottom: {FIGMA_DESIGN_TOKENS['spacing']['lg']};
        ">{subtitle}</p>
        
        <button style="
            background: rgba(255,255,255,0.2);
            border: 2px solid rgba(255,255,255,0.3);
            color: {FIGMA_DESIGN_TOKENS['colors']['text_primary']};
            padding: {FIGMA_DESIGN_TOKENS['spacing']['md']} {FIGMA_DESIGN_TOKENS['spacing']['lg']};
            border-radius: {FIGMA_DESIGN_TOKENS['borders']['radius']['md']};
            font-family: {FIGMA_DESIGN_TOKENS['typography']['font_family']};
            font-weight: {FIGMA_DESIGN_TOKENS['typography']['weights']['semibold']};
            cursor: pointer;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        " onmouseover="this.style.background='rgba(255,255,255,0.3)'" 
           onmouseout="this.style.background='rgba(255,255,255,0.2)'">{cta_text}</button>
    </div>
    """
    
    st.markdown(hero_html, unsafe_allow_html=True)
```

#### Step 3: Update Your Main UI
In `ui_launcher.py`, import and use your new components:

```python
from ui_style import render_figma_hero_section, FIGMA_DESIGN_TOKENS

# In your main app function:
def main():
    # Apply Figma theme
    apply_figma_global_styles()
    
    # Use your Figma components
    render_figma_hero_section(
        "WicketWise Cricket AI", 
        "Advanced cricket analytics powered by AI",
        "Start Analyzing"
    )
```

### **Method 3: Assets + Manual Styling (For Complex Designs)**

#### Step 1: Export Assets from Figma
1. **Export all images as PNG/SVG:**
   - Right-click → Export → PNG (2x for retina)
   - Save to `ui/assets/images/`

2. **Export icons as SVG:**
   - Save to `ui/assets/icons/`

3. **Copy exact CSS values:**
   - Colors, fonts, spacing, shadows

#### Step 2: Load Assets in Streamlit
```python
import base64
from pathlib import Path

def load_figma_image(image_path: str) -> str:
    """Load Figma-exported image for Streamlit"""
    with open(f"ui/assets/images/{image_path}", "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode()
        return f"data:image/png;base64,{encoded}"

def render_figma_card_with_image(title: str, image_name: str):
    """Example card with Figma-exported image"""
    
    image_data = load_figma_image(image_name)
    
    card_html = f"""
    <div style="
        background: #252B37;
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin: 16px 0;
    ">
        <img src="{image_data}" style="
            width: 100%;
            height: 200px;
            object-fit: cover;
        "/>
        <div style="padding: 24px;">
            <h3 style="
                color: #FFFFFF;
                font-family: 'Inter', sans-serif;
                font-weight: 600;
                margin: 0;
            ">{title}</h3>
        </div>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
```

## **Integration Checklist**

- [ ] **Export from Figma** (using plugin or manual)
- [ ] **Extract design tokens** (colors, fonts, spacing)
- [ ] **Create asset folders** (`ui/assets/images/`, `ui/assets/icons/`)
- [ ] **Add custom components** to `ui_style.py`
- [ ] **Update theme** in `ui_theme.py`
- [ ] **Import in main UI** (`ui_launcher.py`)
- [ ] **Test responsiveness** on different screen sizes

## **Pro Tips**

1. **Keep it modular:** Create one function per Figma component
2. **Use design tokens:** Don't hardcode colors/spacing
3. **Test early:** Preview each component as you convert it
4. **Stay responsive:** Figma designs might need mobile adjustments
5. **Optimize assets:** Compress images, use SVGs when possible

## **Next Steps**

1. **Run the converter tool:**
   ```bash
   streamlit run figma_to_streamlit_converter.py
   ```

2. **Start with one component** and gradually add more

3. **Share your Figma file** if you need help with specific components

---

**Need help?** Share your Figma file or exported code, and I can help convert specific components!