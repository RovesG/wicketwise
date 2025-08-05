# Wicketwise UI Assets

## Directory Structure

```
ui/assets/
├── figma_exports/     # Raw exports from Figma
├── images/           # Optimized images (PNG, JPG)
├── icons/            # SVG icons
├── fonts/            # Custom font files
├── styles/           # CSS files
└── figma_config.json # Design tokens and config
```

## Usage

1. **Export from Figma:**
   - Use "Figma to Code" plugin
   - Save exports to `figma_exports/`

2. **Optimize assets:**
   - Compress images
   - Convert to appropriate formats
   - Move to respective folders

3. **Update config:**
   - Edit `figma_config.json` with your design tokens
   - Map Figma components to Streamlit functions

## Integration Tools

- `figma_to_streamlit_converter.py` - Automated conversion
- `demo_figma_integration.py` - Preview integration
- `FIGMA_INTEGRATION_STEPS.md` - Detailed guide
