# Purpose: Setup script to prepare Figma integration tools for Wicketwise
# Author: Assistant, Last Modified: 2024-12-19

import os
import sys
from pathlib import Path
import subprocess

def setup_figma_integration():
    """
    Setup Figma integration tools and directories
    """
    
    print("🎨 Setting up Figma Integration for Wicketwise...")
    
    # Create necessary directories
    directories = [
        "ui/assets/figma_exports",
        "ui/assets/images", 
        "ui/assets/icons",
        "ui/assets/fonts",
        "ui/assets/styles"
    ]
    
    print("\n📁 Creating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")
    
    # Check for required packages
    print("\n📦 Checking required packages...")
    required_packages = [
        "streamlit",
        "streamlit-option-menu", 
        "streamlit-extras"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package} is missing")
    
    # Install missing packages
    if missing_packages:
        print(f"\n🔧 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"   ✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"   ❌ Failed to install {package}")
    
    # Create .gitignore for assets if not exists
    gitignore_path = Path("ui/assets/.gitignore")
    if not gitignore_path.exists():
        print("\n📝 Creating .gitignore for assets...")
        with open(gitignore_path, 'w') as f:
            f.write("""# Figma exports
figma_exports/
*.fig

# Large assets
*.psd
*.sketch
*.ai

# Temporary files
*.tmp
*.cache
""")
        print("   ✅ Created ui/assets/.gitignore")
    
    # Create sample figma_config.json
    config_path = Path("ui/assets/figma_config.json")
    if not config_path.exists():
        print("\n⚙️ Creating sample Figma config...")
        sample_config = {
            "design_tokens": {
                "colors": {
                    "primary": "#4A90E2",
                    "secondary": "#50C878", 
                    "background": "#0F1419",
                    "surface": "#252B37",
                    "text_primary": "#FFFFFF",
                    "text_secondary": "#B8BCC8"
                },
                "typography": {
                    "font_family": "Inter, sans-serif",
                    "sizes": {
                        "h1": "2.5rem",
                        "h2": "2rem",
                        "h3": "1.5rem", 
                        "body": "1rem"
                    }
                },
                "spacing": {
                    "sm": "8px",
                    "md": "16px", 
                    "lg": "24px",
                    "xl": "32px"
                }
            },
            "component_mapping": {
                "hero_section": "render_figma_hero_section",
                "card": "render_figma_card",
                "button": "render_figma_button"
            }
        }
        
        import json
        with open(config_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        print("   ✅ Created ui/assets/figma_config.json")
    
    # Create README for assets
    readme_path = Path("ui/assets/README.md")
    if not readme_path.exists():
        print("\n📚 Creating assets README...")
        readme_content = """# Wicketwise UI Assets

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
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print("   ✅ Created ui/assets/README.md")
    
    print("\n🎉 Figma integration setup complete!")
    print("\n📋 Next Steps:")
    print("   1. Export your Figma design using plugins")
    print("   2. Run: streamlit run demo_figma_integration.py")
    print("   3. Use: streamlit run figma_to_streamlit_converter.py")
    print("   4. Read: FIGMA_INTEGRATION_STEPS.md")
    print("\n💡 Need help? Share your Figma file or exported code!")

if __name__ == "__main__":
    setup_figma_integration()