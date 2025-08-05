# Purpose: Automated converter for Figma exports to Streamlit components
# Author: Assistant, Last Modified: 2024-12-19

import streamlit as st
import json
import re
from pathlib import Path
from typing import Dict, List, Any

class FigmaToStreamlitConverter:
    """
    Converts Figma-exported code into Streamlit-compatible components
    """
    
    def __init__(self):
        self.design_tokens = {}
        self.components = {}
        
    def parse_figma_export(self, figma_export_path: str):
        """
        Parse exported Figma files (HTML, CSS, JSON)
        """
        export_path = Path(figma_export_path)
        
        # Look for common Figma export files
        html_files = list(export_path.glob("*.html"))
        css_files = list(export_path.glob("*.css"))
        json_files = list(export_path.glob("*.json"))
        
        parsed_data = {
            "html": [],
            "css": [],
            "design_tokens": {}
        }
        
        # Parse HTML files
        for html_file in html_files:
            with open(html_file, 'r', encoding='utf-8') as f:
                parsed_data["html"].append({
                    "file": html_file.name,
                    "content": f.read()
                })
        
        # Parse CSS files
        for css_file in css_files:
            with open(css_file, 'r', encoding='utf-8') as f:
                parsed_data["css"].append({
                    "file": css_file.name,
                    "content": f.read()
                })
        
        # Parse design tokens if available
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    tokens = json.load(f)
                    parsed_data["design_tokens"] = tokens
            except json.JSONDecodeError:
                st.warning(f"Could not parse {json_file.name} as design tokens")
        
        return parsed_data
    
    def extract_design_tokens_from_css(self, css_content: str) -> Dict:
        """
        Extract design tokens from CSS content
        """
        tokens = {
            "colors": {},
            "fonts": {},
            "spacing": {},
            "borders": {}
        }
        
        # Extract CSS custom properties (variables)
        css_vars = re.findall(r'--([^:]+):\s*([^;]+);', css_content)
        
        for var_name, var_value in css_vars:
            if 'color' in var_name.lower():
                tokens["colors"][var_name] = var_value.strip()
            elif any(unit in var_value for unit in ['px', 'rem', 'em']):
                if 'radius' in var_name.lower():
                    tokens["borders"][var_name] = var_value.strip()
                else:
                    tokens["spacing"][var_name] = var_value.strip()
            elif any(font in var_value.lower() for font in ['font', 'family']):
                tokens["fonts"][var_name] = var_value.strip()
        
        # Extract direct color values
        colors = re.findall(r'#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3}|rgb\([^)]+\)|rgba\([^)]+\)', css_content)
        for i, color in enumerate(set(colors)):
            tokens["colors"][f"extracted_color_{i}"] = color
            
        return tokens
    
    def convert_to_streamlit_component(self, component_name: str, html_content: str, css_content: str) -> str:
        """
        Convert HTML/CSS to Streamlit component function
        """
        
        # Clean up HTML for Streamlit
        cleaned_html = self.clean_html_for_streamlit(html_content)
        cleaned_css = self.clean_css_for_streamlit(css_content)
        
        # Generate Python function
        component_function = f'''
def render_{component_name.lower().replace("-", "_").replace(" ", "_")}(data: Dict[str, Any] = None, **kwargs):
    """
    Streamlit component converted from Figma design: {component_name}
    """
    
    # Merge any passed data with default styling
    if data is None:
        data = {{}}
    
    # Apply any custom styling from kwargs
    custom_styles = ""
    for key, value in kwargs.items():
        if key.startswith('style_'):
            style_prop = key.replace('style_', '').replace('_', '-')
            custom_styles += f"{style_prop}: {value}; "
    
    # Component HTML with embedded CSS
    component_html = f"""
    <style>
    {cleaned_css}
    .figma-component-{component_name.lower().replace(" ", "-")} {{
        {{custom_styles}}
    }}
    </style>
    <div class="figma-component-{component_name.lower().replace(" ", "-")}">
        {cleaned_html}
    </div>
    """
    
    st.markdown(component_html, unsafe_allow_html=True)
    
    return component_html
'''
        
        return component_function
    
    def clean_html_for_streamlit(self, html_content: str) -> str:
        """
        Clean HTML content for Streamlit compatibility
        """
        # Remove script tags (Streamlit doesn't allow them)
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
        
        # Remove problematic attributes
        html_content = re.sub(r'on\w+="[^"]*"', '', html_content)  # Remove event handlers
        
        # Convert data placeholders to Python f-string format
        html_content = re.sub(r'\{\{(\w+)\}\}', r'{data.get("\1", "")}', html_content)
        
        return html_content
    
    def clean_css_for_streamlit(self, css_content: str) -> str:
        """
        Clean CSS content for Streamlit compatibility
        """
        # Remove @import statements (can cause issues)
        css_content = re.sub(r'@import[^;]+;', '', css_content)
        
        # Scope CSS to avoid conflicts with Streamlit's default styles
        # This is a simple approach - you might need more sophisticated scoping
        css_content = re.sub(r'(^|\n)([^@\{\n][^{]*)\{', r'\1.figma-component \2{', css_content)
        
        return css_content
    
    def generate_streamlit_module(self, components: List[Dict], output_path: str = "figma_components.py"):
        """
        Generate a complete Python module with all converted components
        """
        
        module_content = '''# Purpose: Streamlit components converted from Figma designs
# Author: Auto-generated by FigmaToStreamlitConverter
# Last Modified: Auto-generated

import streamlit as st
from typing import Dict, Any, Optional

"""
Figma-to-Streamlit Component Library
Generated automatically from Figma exports
"""

'''
        
        for component in components:
            module_content += component["function_code"] + "\n\n"
        
        # Add utility functions
        module_content += '''
def apply_figma_theme():
    """Apply global Figma theme to Streamlit app"""
    st.markdown("""
    <style>
    /* Global Figma theme styles */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit default elements for cleaner Figma integration */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def load_figma_fonts():
    """Load custom fonts from Figma"""
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)
'''
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(module_content)
        
        return output_path

# Streamlit UI for the converter
def main():
    st.title("🎨 Figma to Streamlit Converter")
    st.markdown("Convert your Figma exports into Streamlit components")
    
    converter = FigmaToStreamlitConverter()
    
    # File upload section
    st.subheader("1. Upload Figma Export")
    
    uploaded_files = st.file_uploader(
        "Upload Figma export files (HTML, CSS, JSON)",
        accept_multiple_files=True,
        type=['html', 'css', 'json']
    )
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} files")
        
        # Process files
        parsed_data = {"html": [], "css": [], "design_tokens": {}}
        
        for file in uploaded_files:
            content = file.read().decode('utf-8')
            
            if file.name.endswith('.html'):
                parsed_data["html"].append({"file": file.name, "content": content})
            elif file.name.endswith('.css'):
                parsed_data["css"].append({"file": file.name, "content": content})
                # Extract design tokens from CSS
                tokens = converter.extract_design_tokens_from_css(content)
                parsed_data["design_tokens"].update(tokens)
            elif file.name.endswith('.json'):
                try:
                    tokens = json.loads(content)
                    parsed_data["design_tokens"].update(tokens)
                except json.JSONDecodeError:
                    st.warning(f"Could not parse {file.name} as JSON")
        
        # Display extracted data
        st.subheader("2. Extracted Design Tokens")
        if parsed_data["design_tokens"]:
            st.json(parsed_data["design_tokens"])
        else:
            st.info("No design tokens found. You can manually add them.")
        
        # Convert components
        st.subheader("3. Generated Components")
        
        components = []
        for i, html_data in enumerate(parsed_data["html"]):
            component_name = html_data["file"].replace('.html', '')
            css_content = parsed_data["css"][i]["content"] if i < len(parsed_data["css"]) else ""
            
            function_code = converter.convert_to_streamlit_component(
                component_name, 
                html_data["content"], 
                css_content
            )
            
            components.append({
                "name": component_name,
                "function_code": function_code
            })
            
            st.code(function_code, language='python')
        
        # Generate module
        if st.button("Generate Streamlit Module"):
            output_path = converter.generate_streamlit_module(components)
            st.success(f"Generated {output_path}")
            
            # Provide download
            with open(output_path, 'r') as f:
                st.download_button(
                    "Download Generated Module",
                    f.read(),
                    file_name="figma_components.py",
                    mime="text/python"
                )

if __name__ == "__main__":
    main()