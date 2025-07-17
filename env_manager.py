# Purpose: Environment manager for secure API key and secret management
# Author: WicketWise Team, Last Modified: 2024-01-15

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dotenv import load_dotenv, set_key, unset_key, find_dotenv
import threading

# Configure logging
logger = logging.getLogger(__name__)

class EnvManager:
    """
    Secure environment variable manager for API keys and secrets.
    
    Features:
    - Thread-safe operations
    - Automatic .env file detection
    - Fallback to environment variables
    - Secure key validation
    - Memory-only storage option
    """
    
    def __init__(self, env_file: Optional[str] = None, auto_load: bool = True):
        """
        Initialize the environment manager.
        
        Args:
            env_file: Path to .env file (auto-detected if None)
            auto_load: Whether to automatically load .env on initialization
        """
        self._lock = threading.Lock()
        self._memory_store: Dict[str, str] = {}
        self._env_file = env_file or find_dotenv() or ".env"
        self._env_path = Path(self._env_file)
        
        # Supported service names for validation
        self._supported_services = {
            'betfair': 'BETFAIR_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'huggingface': 'HUGGINGFACE_API_KEY',
            'wandb': 'WANDB_API_KEY',
            'comet': 'COMET_API_KEY'
        }
        
        if auto_load:
            self.load_env()
    
    def get_api_key(self, service_name: str) -> str:
        """
        Get API key for a specific service.
        
        Args:
            service_name: Name of the service (e.g., 'betfair', 'openai')
            
        Returns:
            API key string
            
        Raises:
            ValueError: If service name is not supported
            KeyError: If API key is not found
        """
        with self._lock:
            # Normalize service name
            service_name = service_name.lower().strip()
            
            # Check if service is supported
            if service_name not in self._supported_services:
                raise ValueError(
                    f"Unsupported service: {service_name}. "
                    f"Supported services: {list(self._supported_services.keys())}"
                )
            
            env_var_name = self._supported_services[service_name]
            
            # Try memory store first
            if service_name in self._memory_store:
                logger.debug(f"Retrieved {service_name} key from memory store")
                return self._memory_store[service_name]
            
            # Try environment variable
            api_key = os.getenv(env_var_name)
            if api_key:
                logger.debug(f"Retrieved {service_name} key from environment")
                return api_key
            
            # Key not found
            raise KeyError(f"API key for {service_name} not found in environment or memory store")
    
    def set_api_key(self, service_name: str, key: str, persist: bool = False) -> None:
        """
        Set API key for a specific service.
        
        Args:
            service_name: Name of the service (e.g., 'betfair', 'openai')
            key: API key string
            persist: Whether to persist to .env file
            
        Raises:
            ValueError: If service name is not supported or key is invalid
        """
        with self._lock:
            # Normalize service name
            service_name = service_name.lower().strip()
            
            # Validate service name
            if service_name not in self._supported_services:
                raise ValueError(
                    f"Unsupported service: {service_name}. "
                    f"Supported services: {list(self._supported_services.keys())}"
                )
            
            # Validate key
            if not key or not isinstance(key, str):
                raise ValueError("API key must be a non-empty string")
            
            key = key.strip()
            if not key:
                raise ValueError("API key cannot be empty or whitespace only")
            
            # Basic format validation
            if not self._validate_key_format(service_name, key):
                logger.warning(f"API key for {service_name} may have invalid format")
            
            # Store in memory
            self._memory_store[service_name] = key
            
            # Set in current environment
            env_var_name = self._supported_services[service_name]
            os.environ[env_var_name] = key
            
            # Persist to .env file if requested
            if persist:
                try:
                    self._write_key_to_env(env_var_name, key)
                    logger.info(f"Persisted {service_name} key to .env file")
                except Exception as e:
                    logger.error(f"Failed to persist {service_name} key: {e}")
                    raise
            
            logger.debug(f"Set {service_name} key in memory and environment")
    
    def load_env(self) -> Dict[str, str]:
        """
        Load environment variables from .env file.
        
        Returns:
            Dictionary of loaded environment variables
        """
        loaded_vars = {}
        
        with self._lock:
            if self._env_path.exists():
                try:
                    # Load .env file
                    load_dotenv(self._env_file, override=True)
                    
                    # Check which supported services were loaded
                    for service, env_var in self._supported_services.items():
                        value = os.getenv(env_var)
                        if value:
                            loaded_vars[service] = "***" + value[-4:] if len(value) > 4 else "***"
                            logger.debug(f"Loaded {service} key from .env")
                    
                    logger.info(f"Loaded {len(loaded_vars)} API keys from {self._env_file}")
                    
                except Exception as e:
                    logger.error(f"Failed to load .env file: {e}")
                    raise
            else:
                logger.warning(f".env file not found at {self._env_file}")
        
        return loaded_vars
    
    def write_env(self, backup: bool = True) -> None:
        """
        Write current API keys to .env file.
        
        Args:
            backup: Whether to create a backup of existing .env file
            
        Raises:
            PermissionError: If unable to write to .env file
            OSError: If file system operation fails
        """
        with self._lock:
            try:
                # Create backup if requested and file exists
                if backup and self._env_path.exists():
                    backup_path = self._env_path.with_suffix('.env.backup')
                    backup_path.write_text(self._env_path.read_text())
                    logger.debug(f"Created backup at {backup_path}")
                
                # Write memory store keys to .env
                keys_written = 0
                for service, key in self._memory_store.items():
                    env_var_name = self._supported_services[service]
                    self._write_key_to_env(env_var_name, key)
                    keys_written += 1
                
                # Also write any environment variables that aren't in memory store
                for service, env_var in self._supported_services.items():
                    if service not in self._memory_store:
                        value = os.getenv(env_var)
                        if value:
                            self._write_key_to_env(env_var, value)
                            keys_written += 1
                
                logger.info(f"Wrote {keys_written} API keys to {self._env_file}")
                
            except PermissionError as e:
                logger.error(f"Permission denied writing to .env file: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to write .env file: {e}")
                raise
    
    def remove_api_key(self, service_name: str, persist: bool = False) -> None:
        """
        Remove API key for a specific service.
        
        Args:
            service_name: Name of the service
            persist: Whether to remove from .env file as well
        """
        with self._lock:
            service_name = service_name.lower().strip()
            
            if service_name not in self._supported_services:
                raise ValueError(f"Unsupported service: {service_name}")
            
            env_var_name = self._supported_services[service_name]
            
            # Remove from memory store
            self._memory_store.pop(service_name, None)
            
            # Remove from environment
            os.environ.pop(env_var_name, None)
            
            # Remove from .env file if requested
            if persist and self._env_path.exists():
                try:
                    unset_key(self._env_file, env_var_name)
                    logger.info(f"Removed {service_name} key from .env file")
                except Exception as e:
                    logger.error(f"Failed to remove {service_name} key from .env: {e}")
            
            logger.debug(f"Removed {service_name} key from memory and environment")
    
    def list_available_keys(self) -> Dict[str, bool]:
        """
        List which API keys are available.
        
        Returns:
            Dictionary mapping service names to availability status
        """
        with self._lock:
            availability = {}
            
            for service, env_var in self._supported_services.items():
                # Check memory store first
                if service in self._memory_store:
                    availability[service] = True
                # Then check environment
                elif os.getenv(env_var):
                    availability[service] = True
                else:
                    availability[service] = False
            
            return availability
    
    def validate_all_keys(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate all available API keys.
        
        Returns:
            Dictionary with validation results for each service
        """
        results = {}
        
        for service in self._supported_services.keys():
            try:
                key = self.get_api_key(service)
                results[service] = {
                    'available': True,
                    'valid_format': self._validate_key_format(service, key),
                    'length': len(key),
                    'preview': key[:8] + "..." if len(key) > 8 else key
                }
            except KeyError:
                results[service] = {
                    'available': False,
                    'valid_format': False,
                    'length': 0,
                    'preview': None
                }
        
        return results
    
    def _validate_key_format(self, service_name: str, key: str) -> bool:
        """
        Validate API key format for specific services.
        
        Args:
            service_name: Name of the service
            key: API key to validate
            
        Returns:
            True if format appears valid
        """
        if not key:
            return False
        
        # Service-specific format validation
        if service_name == 'openai':
            return key.startswith('sk-') and len(key) > 20
        elif service_name == 'anthropic':
            return key.startswith('sk-ant-') and len(key) > 20
        elif service_name == 'betfair':
            return len(key) >= 20  # Betfair keys are typically long
        elif service_name == 'huggingface':
            return key.startswith('hf_') and len(key) > 10
        elif service_name == 'wandb':
            return len(key) >= 20  # W&B keys are typically long
        elif service_name == 'comet':
            return len(key) >= 20  # Comet keys are typically long
        
        # Default validation - just check it's not empty
        return len(key) > 0
    
    def _write_key_to_env(self, env_var_name: str, key: str) -> None:
        """
        Write a single key to .env file.
        
        Args:
            env_var_name: Environment variable name
            key: API key value
        """
        try:
            set_key(self._env_file, env_var_name, key)
        except Exception as e:
            logger.error(f"Failed to write {env_var_name} to .env: {e}")
            raise
    
    def get_env_file_path(self) -> Path:
        """Get the path to the .env file being used."""
        return self._env_path
    
    def clear_memory_store(self) -> None:
        """Clear all keys from memory store (but not environment or .env file)."""
        with self._lock:
            self._memory_store.clear()
            logger.debug("Cleared memory store")


# Global instance for convenience
_global_env_manager = None

def get_env_manager() -> EnvManager:
    """Get the global environment manager instance."""
    global _global_env_manager
    if _global_env_manager is None:
        _global_env_manager = EnvManager()
    return _global_env_manager

# Convenience functions using global instance
def get_api_key(service_name: str) -> str:
    """Get API key for a specific service using global env manager."""
    return get_env_manager().get_api_key(service_name)

def set_api_key(service_name: str, key: str, persist: bool = False) -> None:
    """Set API key for a specific service using global env manager."""
    get_env_manager().set_api_key(service_name, key, persist)

def load_env() -> Dict[str, str]:
    """Load environment variables from .env file using global env manager."""
    return get_env_manager().load_env()

def write_env(backup: bool = True) -> None:
    """Write current API keys to .env file using global env manager."""
    get_env_manager().write_env(backup=backup) 