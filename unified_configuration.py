#!/usr/bin/env python3
"""
Unified Configuration System for WicketWise
Centralized, type-safe, environment-aware configuration management

Author: WicketWise Team, Last Modified: 2025-01-21
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class ServerConfig:
    """Server configuration"""
    host: str = "127.0.0.1"
    backend_port: int = 5001
    frontend_port: int = 8000
    cors_enabled: bool = True
    debug_mode: bool = False
    workers: int = 4
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    request_timeout: int = 300  # 5 minutes

@dataclass
class DataConfig:
    """Data paths and file configuration"""
    data_dir: str = "/data/cricket"
    models_dir: str = "models"
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"
    cache_dir: str = "cache"
    logs_dir: str = "logs"
    
    # File patterns with version support
    cricket_data_pattern: str = "joined_ball_by_ball_data_v{version}.csv"
    nvplay_data_pattern: str = "nvplay_data_v{version}.csv"
    decimal_data_pattern: str = "decimal_data_v{version}.csv"
    people_data_pattern: str = "people.csv"
    
    # Current data version
    data_version: int = 3
    
    # File size limits
    max_upload_size: int = 500 * 1024 * 1024  # 500MB
    max_csv_rows: int = 50_000_000  # 50M rows
    
    def get_cricket_data_path(self) -> Path:
        """Get current cricket data file path"""
        filename = self.cricket_data_pattern.format(version=self.data_version)
        return Path(self.data_dir) / filename
    
    def get_nvplay_data_path(self) -> Path:
        """Get current nvplay data file path"""
        filename = self.nvplay_data_pattern.format(version=self.data_version)
        return Path(self.data_dir) / filename
    
    def get_decimal_data_path(self) -> Path:
        """Get current decimal data file path"""
        filename = self.decimal_data_pattern.format(version=self.data_version)
        return Path(self.data_dir) / filename

@dataclass
class ModelConfig:
    """ML Model configuration"""
    # GNN Architecture
    gnn_dimensions: Dict[str, int] = field(default_factory=lambda: {
        "query_dim": 128,
        "batter_dim": 128,
        "bowler_dim": 128,
        "venue_dim": 64,
        "weather_dim": 64,
        "coord_dim": 32,
        "squad_dim": 48
    })
    
    gnn_attention: Dict[str, Any] = field(default_factory=lambda: {
        "heads": 4,
        "dim": 128,
        "dropout": 0.1
    })
    
    # Transformer Configuration
    transformer_config: Dict[str, Any] = field(default_factory=lambda: {
        "d_model": 512,
        "nhead": 8,
        "num_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "max_sequence_length": 1000
    })
    
    # Training Configuration
    training: Dict[str, Any] = field(default_factory=lambda: {
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "epochs": 100,
        "early_stopping_patience": 10,
        "gradient_clip": 1.0,
        "warmup_steps": 1000
    })
    
    # Model paths
    model_save_path: str = "models/crickformer_latest.pt"
    checkpoint_dir: str = "models/checkpoints"
    tensorboard_log_dir: str = "logs/tensorboard"

@dataclass
class APIConfig:
    """External API configuration"""
    # OpenAI Configuration
    openai: Dict[str, Any] = field(default_factory=lambda: {
        "model": "gpt-4o",
        "temperature": 0.1,
        "max_tokens": 4000,
        "rate_limit_delay": 1.0,
        "max_retries": 3,
        "timeout": 30
    })
    
    # Enrichment Pipeline Configuration
    enrichment: Dict[str, Any] = field(default_factory=lambda: {
        "max_concurrent": 10,
        "batch_size": 50,
        "cache_ttl": 86400,  # 24 hours
        "max_connections": 100,
        "max_connections_per_host": 10
    })
    
    # Betfair API Configuration
    betfair: Dict[str, Any] = field(default_factory=lambda: {
        "app_key": "",
        "username": "",
        "password": "",
        "cert_file": "",
        "key_file": "",
        "timeout": 30,
        "max_retries": 3
    })

@dataclass
class SecurityConfig:
    """Security configuration"""
    # JWT Configuration
    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    
    # Rate Limiting
    rate_limits: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "enrich_matches": {"hour": 10, "day": 50},
        "build_kg": {"hour": 3, "day": 10},
        "train_model": {"hour": 1, "day": 5},
        "default": {"minute": 60, "hour": 1000}
    })
    
    # Password Policy
    password_policy: Dict[str, Any] = field(default_factory=lambda: {
        "min_length": 8,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_digits": True,
        "require_special": True,
        "max_age_days": 90
    })
    
    # Security Headers
    security_headers: Dict[str, str] = field(default_factory=lambda: {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'"
    })

@dataclass
class PerformanceConfig:
    """Performance and optimization configuration"""
    # Caching Configuration
    cache: Dict[str, Any] = field(default_factory=lambda: {
        "redis_url": "redis://localhost:6379/0",
        "default_ttl": 3600,
        "max_memory": "256mb",
        "eviction_policy": "allkeys-lru"
    })
    
    # Database Configuration
    database: Dict[str, Any] = field(default_factory=lambda: {
        "pool_size": 20,
        "max_overflow": 30,
        "pool_timeout": 30,
        "pool_recycle": 3600,
        "echo": False
    })
    
    # Async Configuration
    async_config: Dict[str, Any] = field(default_factory=lambda: {
        "max_workers": 10,
        "queue_size": 1000,
        "timeout": 300
    })
    
    # Memory Management
    memory: Dict[str, Any] = field(default_factory=lambda: {
        "chunk_size": 50000,
        "max_memory_usage": "8GB",
        "gc_threshold": 0.8
    })

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Log files
    access_log: str = "logs/access.log"
    error_log: str = "logs/error.log"
    security_log: str = "logs/security.log"
    performance_log: str = "logs/performance.log"
    
    # Log rotation
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # Structured logging
    structured: bool = True
    include_trace: bool = False

@dataclass
class MatchingConfig:
    """Match alignment and fuzzy matching configuration"""
    # Similarity thresholds
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "team_similarity": 0.7,
        "venue_similarity": 0.6,
        "player_similarity": 0.8,
        "exact_match": 1.0,
        "fuzzy_match": 0.6
    })
    
    # Aligner settings
    aligner: Dict[str, Any] = field(default_factory=lambda: {
        "strategy": "hybrid",  # hybrid, fingerprint, dna_hash, llm
        "fingerprint_length": 50,
        "max_distance": 5,
        "use_llm_fallback": True
    })
    
    # Team aliases for better matching
    team_aliases: Dict[str, List[str]] = field(default_factory=lambda: {
        "Royal Challengers Bangalore": ["Royal Challengers Bengaluru", "RCB", "Bangalore", "Bengaluru"],
        "Chennai Super Kings": ["CSK", "Chennai", "Super Kings"],
        "Mumbai Indians": ["MI", "Mumbai"],
        "Kolkata Knight Riders": ["KKR", "Kolkata", "Knight Riders"],
        "Delhi Capitals": ["DC", "Delhi Daredevils", "DD", "Delhi"],
        "Rajasthan Royals": ["RR", "Rajasthan"],
        "Punjab Kings": ["PBKS", "Kings XI Punjab", "KXIP", "Punjab"],
        "Sunrisers Hyderabad": ["SRH", "Hyderabad", "Sunrisers"]
    })

class WicketWiseConfig:
    """Main configuration class"""
    
    def __init__(self, config_file: Optional[str] = None, environment: Optional[Environment] = None):
        self.environment = environment or Environment.DEVELOPMENT
        self.config_file = config_file
        self.loaded_at = datetime.utcnow()
        
        # Initialize configuration sections
        self.server = ServerConfig()
        self.data = DataConfig()
        self.models = ModelConfig()
        self.apis = APIConfig()
        self.security = SecurityConfig()
        self.performance = PerformanceConfig()
        self.logging = LoggingConfig()
        self.matching = MatchingConfig()
        
        # Load configuration
        self._load_configuration()
        self._validate_configuration()
    
    def _load_configuration(self):
        """Load configuration from various sources"""
        # 1. Load from file if specified
        if self.config_file:
            self._load_from_file(self.config_file)
        
        # 2. Load environment-specific config
        env_config_file = f"config/wicketwise_{self.environment.value}.yaml"
        if Path(env_config_file).exists():
            self._load_from_file(env_config_file)
        
        # 3. Override with environment variables
        self._load_from_environment()
        
        logger.info(f"Configuration loaded for {self.environment.value} environment")
    
    def _load_from_file(self, config_file: str):
        """Load configuration from YAML or JSON file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    logger.error(f"Unsupported configuration file format: {config_path.suffix}")
                    return
            
            self._merge_config(config_data)
            logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            # Server configuration
            'WICKETWISE_HOST': ('server', 'host'),
            'WICKETWISE_BACKEND_PORT': ('server', 'backend_port', int),
            'WICKETWISE_FRONTEND_PORT': ('server', 'frontend_port', int),
            'WICKETWISE_DEBUG': ('server', 'debug_mode', bool),
            'WICKETWISE_CORS_ENABLED': ('server', 'cors_enabled', bool),
            
            # Data configuration
            'WICKETWISE_DATA_DIR': ('data', 'data_dir'),
            'WICKETWISE_MODELS_DIR': ('data', 'models_dir'),
            'WICKETWISE_DATA_VERSION': ('data', 'data_version', int),
            
            # Security configuration
            'WICKETWISE_JWT_SECRET': ('security', 'jwt_secret_key'),
            'WICKETWISE_JWT_EXPIRY': ('security', 'jwt_expiry_hours', int),
            
            # API Keys (handled separately for security)
            'OPENAI_API_KEY': ('_api_keys', 'openai'),
            'BETFAIR_API_KEY': ('_api_keys', 'betfair'),
            'ANTHROPIC_API_KEY': ('_api_keys', 'anthropic'),
        }
        
        # API keys stored separately for security
        self._api_keys = {}
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Handle type conversion
                if len(config_path) > 2:
                    converter = config_path[2]
                    if converter == bool:
                        value = value.lower() in ('true', '1', 'yes', 'on')
                    elif converter == int:
                        value = int(value)
                    elif converter == float:
                        value = float(value)
                
                # Set configuration value
                section_name, attr_name = config_path[0], config_path[1]
                if section_name == '_api_keys':
                    self._api_keys[attr_name] = value
                else:
                    section = getattr(self, section_name)
                    setattr(section, attr_name, value)
    
    def _merge_config(self, config_data: Dict[str, Any]):
        """Merge configuration data into existing configuration"""
        for section_name, section_data in config_data.items():
            if hasattr(self, section_name) and isinstance(section_data, dict):
                section = getattr(self, section_name)
                for key, value in section_data.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
    
    def _validate_configuration(self):
        """Validate configuration values"""
        errors = []
        
        # Validate paths exist or can be created (skip in testing/development environment)
        if self.environment == Environment.PRODUCTION:
            paths_to_check = [
                self.data.data_dir,
                self.data.models_dir,
                self.data.artifacts_dir,
                self.data.reports_dir,
                self.data.cache_dir,
                self.data.logs_dir
            ]
            
            for path_str in paths_to_check:
                path = Path(path_str)
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create directory {path}: {e}")
        
        # Validate ports
        if not (1024 <= self.server.backend_port <= 65535):
            errors.append(f"Invalid backend port: {self.server.backend_port}")
        
        if not (1024 <= self.server.frontend_port <= 65535):
            errors.append(f"Invalid frontend port: {self.server.frontend_port}")
        
        # Validate JWT secret in production
        if (self.environment == Environment.PRODUCTION and 
            self.security.jwt_secret_key == "change-me-in-production"):
            errors.append("JWT secret key must be changed in production")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation passed")
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Get API key for service"""
        return self._api_keys.get(service.lower())
    
    def set_api_key(self, service: str, key: str):
        """Set API key for service"""
        self._api_keys[service.lower()] = key
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)"""
        config_dict = {}
        
        for attr_name in ['server', 'data', 'models', 'apis', 'performance', 'logging', 'matching']:
            section = getattr(self, attr_name)
            config_dict[attr_name] = asdict(section)
        
        # Add non-sensitive security config
        security_dict = asdict(self.security)
        security_dict['jwt_secret_key'] = '***REDACTED***'
        config_dict['security'] = security_dict
        
        # Add metadata
        config_dict['_metadata'] = {
            'environment': self.environment.value,
            'loaded_at': self.loaded_at.isoformat(),
            'config_file': self.config_file
        }
        
        return config_dict
    
    def save_to_file(self, file_path: str):
        """Save current configuration to file"""
        config_dict = self.to_dict()
        
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'w') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif file_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")
            raise
    
    def reload(self):
        """Reload configuration from all sources"""
        logger.info("Reloading configuration...")
        self.__init__(self.config_file, self.environment)

# Global configuration instance
_config_instance: Optional[WicketWiseConfig] = None

def get_config() -> WicketWiseConfig:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        # Check for environment variable
        env_name = os.getenv('WICKETWISE_ENV', 'development').upper()
        try:
            environment = Environment[env_name]
        except KeyError:
            environment = Environment.DEVELOPMENT
        
        _config_instance = WicketWiseConfig(environment=environment)
    return _config_instance

def init_config(config_file: Optional[str] = None, environment: Optional[Environment] = None) -> WicketWiseConfig:
    """Initialize global configuration"""
    global _config_instance
    _config_instance = WicketWiseConfig(config_file, environment)
    return _config_instance

# Convenience function for backward compatibility
def get_settings():
    """Get configuration (backward compatibility)"""
    return get_config()

# Example usage
if __name__ == "__main__":
    # Initialize configuration
    config = WicketWiseConfig(environment=Environment.DEVELOPMENT)
    
    # Print configuration summary
    print("🔧 WicketWise Configuration Summary")
    print("=" * 50)
    print(f"Environment: {config.environment.value}")
    print(f"Backend URL: http://{config.server.host}:{config.server.backend_port}")
    print(f"Frontend URL: http://{config.server.host}:{config.server.frontend_port}")
    print(f"Data Directory: {config.data.data_dir}")
    print(f"Models Directory: {config.data.models_dir}")
    print(f"Debug Mode: {config.server.debug_mode}")
    print(f"CORS Enabled: {config.server.cors_enabled}")
    
    # Save configuration template
    config.save_to_file("config/wicketwise_template.yaml")
    print("\n✅ Configuration template saved to config/wicketwise_template.yaml")
