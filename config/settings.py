# Purpose: Centralized configuration for WicketWise Cricket Analytics
# Author: WicketWise Team, Last Modified: 2025-08-17

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class WicketWiseSettings:
    """
    Centralized configuration for WicketWise system.
    Uses environment variables with sensible defaults.
    """
    
    # Server Configuration
    BACKEND_HOST: str = os.getenv('WICKETWISE_BACKEND_HOST', '127.0.0.1')
    BACKEND_PORT: int = int(os.getenv('WICKETWISE_BACKEND_PORT', '5001'))
    FRONTEND_PORT: int = int(os.getenv('WICKETWISE_FRONTEND_PORT', '8000'))
    
    # Data Paths (configurable via environment)
    DATA_DIR: str = os.getenv('WICKETWISE_DATA_DIR', '/Users/shamusrae/Library/Mobile Documents/com~apple~CloudDocs/Cricket /Data')
    MODELS_DIR: str = os.getenv('WICKETWISE_MODELS_DIR', 'models')
    ARTIFACTS_DIR: str = os.getenv('WICKETWISE_ARTIFACTS_DIR', 'artifacts')
    REPORTS_DIR: str = os.getenv('WICKETWISE_REPORTS_DIR', 'reports')
    
    # API Configuration
    CORS_ENABLED: bool = os.getenv('WICKETWISE_CORS_ENABLED', 'true').lower() == 'true'
    DEBUG_MODE: bool = os.getenv('WICKETWISE_DEBUG', 'true').lower() == 'true'
    
    # API Keys (handled by env_manager, but documented here)
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    BETFAIR_API_KEY: Optional[str] = os.getenv('BETFAIR_API_KEY')
    
    @property
    def backend_url(self) -> str:
        """Full backend URL for API calls"""
        return f"http://{self.BACKEND_HOST}:{self.BACKEND_PORT}"
    
    @property
    def frontend_url(self) -> str:
        """Full frontend URL"""
        return f"http://{self.BACKEND_HOST}:{self.FRONTEND_PORT}"
    
    def get_data_path(self, filename: str) -> Path:
        """Get full path to data file"""
        return Path(self.DATA_DIR) / filename
    
    def get_model_path(self, filename: str) -> Path:
        """Get full path to model file"""
        return Path(self.MODELS_DIR) / filename
    
    def validate_paths(self) -> bool:
        """Validate that required paths exist"""
        data_dir = Path(self.DATA_DIR)
        models_dir = Path(self.MODELS_DIR)
        
        # Create directories if they don't exist
        models_dir.mkdir(exist_ok=True)
        Path(self.ARTIFACTS_DIR).mkdir(exist_ok=True)
        Path(self.REPORTS_DIR).mkdir(exist_ok=True)
        
        # Check if data directory exists
        if not data_dir.exists():
            print(f"⚠️  Warning: Data directory not found at {data_dir}")
            print(f"   Set WICKETWISE_DATA_DIR environment variable to correct path")
            return False
        
        return True

# Global settings instance
settings = WicketWiseSettings()

# Validate configuration on import
if not settings.validate_paths():
    print("⚠️  Configuration validation failed - some features may not work")

# API Endpoints Configuration
class APIEndpoints:
    """Centralized API endpoint definitions"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
    
    @property
    def health(self) -> str:
        return f"{self.base_url}/api/health"
    
    @property
    def build_knowledge_graph(self) -> str:
        return f"{self.base_url}/api/build-knowledge-graph"
    
    @property
    def train_model(self) -> str:
        return f"{self.base_url}/api/train-model"
    
    @property
    def kg_chat(self) -> str:
        return f"{self.base_url}/api/kg-chat"
    
    @property
    def kg_settings(self) -> str:
        return f"{self.base_url}/api/kg-settings"
    
    @property
    def aligner_settings(self) -> str:
        return f"{self.base_url}/api/aligner-settings"
    
    @property
    def training_settings(self) -> str:
        return f"{self.base_url}/api/training-settings"
    
    @property
    def alljson_settings(self) -> str:
        return f"{self.base_url}/api/alljson-settings"
    
    def operation_status(self, operation_id: str) -> str:
        return f"{self.base_url}/api/operation-status/{operation_id}"

# Global API endpoints instance
api = APIEndpoints(settings.backend_url)
