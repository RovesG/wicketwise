# Purpose: Comprehensive test suite for env_manager.py
# Author: WicketWise Team, Last Modified: 2024-01-15

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
import threading
import time

# Import the module to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from env_manager import EnvManager, get_api_key, set_api_key, load_env, write_env, get_env_manager

class TestEnvManager:
    """Test suite for EnvManager class"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_env_file = Path(self.temp_dir) / ".env"
        
        # Clear any existing environment variables
        env_vars = ['BETFAIR_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 
                   'HUGGINGFACE_API_KEY', 'WANDB_API_KEY', 'COMET_API_KEY']
        for var in env_vars:
            os.environ.pop(var, None)
    
    def teardown_method(self):
        """Cleanup after each test method"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Clear environment variables
        env_vars = ['BETFAIR_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 
                   'HUGGINGFACE_API_KEY', 'WANDB_API_KEY', 'COMET_API_KEY']
        for var in env_vars:
            os.environ.pop(var, None)
    
    def test_init_with_custom_env_file(self):
        """Test EnvManager initialization with custom .env file"""
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        assert env_manager.get_env_file_path() == self.temp_env_file
    
    def test_init_with_auto_load(self):
        """Test EnvManager initialization with auto-load"""
        # Create a test .env file
        self.temp_env_file.write_text("OPENAI_API_KEY=sk-test123456789\n")
        
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=True)
        
        # Should have loaded the key
        assert env_manager.get_api_key('openai') == 'sk-test123456789'
    
    def test_set_and_get_api_key_memory_only(self):
        """Test setting and getting API key in memory only"""
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        
        # Set key in memory only
        env_manager.set_api_key('openai', 'sk-test123456789', persist=False)
        
        # Should be able to retrieve it
        assert env_manager.get_api_key('openai') == 'sk-test123456789'
        
        # Should not be in .env file
        assert not self.temp_env_file.exists()
    
    def test_set_and_get_api_key_with_persistence(self):
        """Test setting and getting API key with persistence"""
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        
        # Set key with persistence
        env_manager.set_api_key('openai', 'sk-test123456789', persist=True)
        
        # Should be able to retrieve it
        assert env_manager.get_api_key('openai') == 'sk-test123456789'
        
        # Should be in .env file
        assert self.temp_env_file.exists()
        content = self.temp_env_file.read_text()
        assert 'OPENAI_API_KEY=' in content and 'sk-test123456789' in content
    
    def test_get_api_key_not_found(self):
        """Test getting API key that doesn't exist"""
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        
        with pytest.raises(KeyError, match="API key for openai not found"):
            env_manager.get_api_key('openai')
    
    def test_set_api_key_invalid_service(self):
        """Test setting API key for unsupported service"""
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        
        with pytest.raises(ValueError, match="Unsupported service: invalid"):
            env_manager.set_api_key('invalid', 'test-key')
    
    def test_set_api_key_invalid_key(self):
        """Test setting invalid API key"""
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        
        # Empty key
        with pytest.raises(ValueError, match="API key must be a non-empty string"):
            env_manager.set_api_key('openai', '')
        
        # None key
        with pytest.raises(ValueError, match="API key must be a non-empty string"):
            env_manager.set_api_key('openai', None)
        
        # Whitespace only key
        with pytest.raises(ValueError, match="API key cannot be empty or whitespace only"):
            env_manager.set_api_key('openai', '   ')
    
    def test_load_env_file_exists(self):
        """Test loading .env file when it exists"""
        # Create test .env file
        env_content = """OPENAI_API_KEY=sk-test123456789
BETFAIR_API_KEY=betfair-test-key-123
ANTHROPIC_API_KEY=sk-ant-test123456789
"""
        self.temp_env_file.write_text(env_content)
        
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        loaded_vars = env_manager.load_env()
        
        # Should have loaded 3 keys
        assert len(loaded_vars) == 3
        assert 'openai' in loaded_vars
        assert 'betfair' in loaded_vars
        assert 'anthropic' in loaded_vars
        
        # Keys should be masked in return value
        assert loaded_vars['openai'] == '***6789'
        assert loaded_vars['betfair'] == '***-123'
    
    def test_load_env_file_not_exists(self):
        """Test loading .env file when it doesn't exist"""
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        loaded_vars = env_manager.load_env()
        
        # Should return empty dict
        assert loaded_vars == {}
    
    def test_write_env_from_memory(self):
        """Test writing environment variables to .env file"""
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        
        # Set some keys in memory
        env_manager.set_api_key('openai', 'sk-test123456789', persist=False)
        env_manager.set_api_key('betfair', 'betfair-test-key-123', persist=False)
        
        # Write to .env file
        env_manager.write_env(backup=False)
        
        # Check file contents
        assert self.temp_env_file.exists()
        content = self.temp_env_file.read_text()
        assert 'OPENAI_API_KEY=' in content and 'sk-test123456789' in content
        assert 'BETFAIR_API_KEY=' in content and 'betfair-test-key-123' in content
    
    def test_write_env_with_backup(self):
        """Test writing .env file with backup creation"""
        # Create existing .env file
        self.temp_env_file.write_text("EXISTING_KEY=existing_value\n")
        
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        env_manager.set_api_key('openai', 'sk-test123456789', persist=False)
        
        # Write with backup
        env_manager.write_env(backup=True)
        
        # Check backup was created
        backup_file = self.temp_env_file.with_suffix('.env.backup')
        assert backup_file.exists()
        assert backup_file.read_text() == "EXISTING_KEY=existing_value\n"
    
    def test_remove_api_key_memory_only(self):
        """Test removing API key from memory only"""
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        
        # Set and then remove key
        env_manager.set_api_key('openai', 'sk-test123456789', persist=False)
        env_manager.remove_api_key('openai', persist=False)
        
        # Should not be available anymore
        with pytest.raises(KeyError):
            env_manager.get_api_key('openai')
    
    def test_remove_api_key_with_persistence(self):
        """Test removing API key with persistence"""
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        
        # Set key with persistence
        env_manager.set_api_key('openai', 'sk-test123456789', persist=True)
        
        # Remove with persistence
        env_manager.remove_api_key('openai', persist=True)
        
        # Should not be available
        with pytest.raises(KeyError):
            env_manager.get_api_key('openai')
        
        # Should not be in .env file
        if self.temp_env_file.exists():
            content = self.temp_env_file.read_text()
            assert 'OPENAI_API_KEY' not in content
    
    def test_list_available_keys(self):
        """Test listing available API keys"""
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        
        # Initially no keys available
        availability = env_manager.list_available_keys()
        assert all(not available for available in availability.values())
        
        # Set some keys
        env_manager.set_api_key('openai', 'sk-test123456789', persist=False)
        env_manager.set_api_key('betfair', 'betfair-test-key', persist=False)
        
        # Check availability
        availability = env_manager.list_available_keys()
        assert availability['openai'] is True
        assert availability['betfair'] is True
        assert availability['anthropic'] is False
    
    def test_validate_all_keys(self):
        """Test validating all API keys"""
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        
        # Set keys with different formats
        env_manager.set_api_key('openai', 'sk-test123456789012345', persist=False)  # Valid format
        env_manager.set_api_key('betfair', 'betfair-test-key-123456789', persist=False)  # Valid length
        env_manager.set_api_key('anthropic', 'invalid-key', persist=False)  # Invalid format
        
        validation_results = env_manager.validate_all_keys()
        
        # OpenAI should be valid
        assert validation_results['openai']['available'] is True
        assert validation_results['openai']['valid_format'] is True
        
        # Betfair should be available but format validation is basic
        assert validation_results['betfair']['available'] is True
        assert validation_results['betfair']['valid_format'] is True
        
        # Anthropic should be available but invalid format
        assert validation_results['anthropic']['available'] is True
        assert validation_results['anthropic']['valid_format'] is False
        
        # Huggingface should not be available
        assert validation_results['huggingface']['available'] is False
    
    def test_key_format_validation(self):
        """Test API key format validation"""
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        
        # Test OpenAI format validation
        assert env_manager._validate_key_format('openai', 'sk-test123456789012345') is True
        assert env_manager._validate_key_format('openai', 'invalid-key') is False
        
        # Test Anthropic format validation
        assert env_manager._validate_key_format('anthropic', 'sk-ant-test123456789012345') is True
        assert env_manager._validate_key_format('anthropic', 'sk-test123456789') is False
        
        # Test Huggingface format validation
        assert env_manager._validate_key_format('huggingface', 'hf_test123456789') is True
        assert env_manager._validate_key_format('huggingface', 'invalid-key') is False
    
    def test_thread_safety(self):
        """Test thread safety of EnvManager operations"""
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        
        def set_keys(service_suffix):
            for i in range(10):
                env_manager.set_api_key('openai', f'sk-test{service_suffix}{i:03d}', persist=False)
                time.sleep(0.001)  # Small delay to encourage race conditions
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=set_keys, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have some key set (exact value depends on thread timing)
        key = env_manager.get_api_key('openai')
        assert key.startswith('sk-test')
    
    def test_clear_memory_store(self):
        """Test clearing memory store"""
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        
        # Set keys in memory
        env_manager.set_api_key('openai', 'sk-test123456789', persist=False)
        env_manager.set_api_key('betfair', 'betfair-test-key', persist=False)
        
        # Clear memory store
        env_manager.clear_memory_store()
        
        # Keys should still be in environment but not in memory store
        # This means they should still be accessible via get_api_key
        assert env_manager.get_api_key('openai') == 'sk-test123456789'
        assert env_manager.get_api_key('betfair') == 'betfair-test-key'
    
    def test_fallback_to_environment_variables(self):
        """Test fallback to environment variables when not in memory store"""
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        
        # Set environment variable directly
        os.environ['OPENAI_API_KEY'] = 'sk-env-test123456789'
        
        # Should be able to retrieve from environment
        assert env_manager.get_api_key('openai') == 'sk-env-test123456789'
    
    @patch('env_manager.set_key')
    def test_write_env_file_error_handling(self, mock_set_key):
        """Test error handling when writing to .env file fails"""
        mock_set_key.side_effect = PermissionError("Permission denied")
        
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        env_manager.set_api_key('openai', 'sk-test123456789', persist=False)
        
        # Should raise PermissionError
        with pytest.raises(PermissionError):
            env_manager.write_env()
    
    @patch('env_manager.load_dotenv')
    def test_load_env_file_error_handling(self, mock_load_dotenv):
        """Test error handling when loading .env file fails"""
        mock_load_dotenv.side_effect = Exception("File corrupted")
        
        # Create .env file so it exists
        self.temp_env_file.write_text("OPENAI_API_KEY=test\n")
        
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        
        # Should raise the exception
        with pytest.raises(Exception, match="File corrupted"):
            env_manager.load_env()


class TestConvenienceFunctions:
    """Test suite for convenience functions"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Clear any existing environment variables
        env_vars = ['BETFAIR_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
        for var in env_vars:
            os.environ.pop(var, None)
        
        # Reset global env manager
        import env_manager
        env_manager._global_env_manager = None
    
    def teardown_method(self):
        """Cleanup after each test method"""
        # Clear environment variables
        env_vars = ['BETFAIR_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
        for var in env_vars:
            os.environ.pop(var, None)
        
        # Reset global env manager
        import env_manager
        env_manager._global_env_manager = None
    
    def test_global_env_manager_singleton(self):
        """Test that global env manager is a singleton"""
        manager1 = get_env_manager()
        manager2 = get_env_manager()
        
        assert manager1 is manager2
    
    def test_convenience_set_and_get_api_key(self):
        """Test convenience functions for setting and getting API keys"""
        # Set key using convenience function
        set_api_key('openai', 'sk-test123456789', persist=False)
        
        # Get key using convenience function
        key = get_api_key('openai')
        assert key == 'sk-test123456789'
    
    def test_convenience_load_env(self):
        """Test convenience function for loading .env"""
        # Set environment variable
        os.environ['OPENAI_API_KEY'] = 'sk-test123456789'
        
        # Load using convenience function
        loaded_vars = load_env()
        
        # Should have loaded the key
        assert 'openai' in loaded_vars
    
    @patch('env_manager.EnvManager.write_env')
    def test_convenience_write_env(self, mock_write_env):
        """Test convenience function for writing .env"""
        # Call convenience function
        write_env(backup=False)
        
        # Should have called the manager's write_env method
        mock_write_env.assert_called_once_with(backup=False)


class TestEdgeCases:
    """Test suite for edge cases and error conditions"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_env_file = Path(self.temp_dir) / ".env"
    
    def teardown_method(self):
        """Cleanup after each test method"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_case_insensitive_service_names(self):
        """Test that service names are case insensitive"""
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        
        # Set with different cases
        env_manager.set_api_key('OpenAI', 'sk-test123456789', persist=False)
        env_manager.set_api_key('BETFAIR', 'betfair-test-key', persist=False)
        
        # Get with different cases
        assert env_manager.get_api_key('openai') == 'sk-test123456789'
        assert env_manager.get_api_key('betfair') == 'betfair-test-key'
        assert env_manager.get_api_key('OPENAI') == 'sk-test123456789'
    
    def test_whitespace_handling(self):
        """Test handling of whitespace in service names and keys"""
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=False)
        
        # Set with whitespace
        env_manager.set_api_key('  openai  ', '  sk-test123456789  ', persist=False)
        
        # Should normalize and work
        assert env_manager.get_api_key('openai') == 'sk-test123456789'
        assert env_manager.get_api_key('  openai  ') == 'sk-test123456789'
    
    def test_empty_env_file(self):
        """Test handling of empty .env file"""
        # Clear any existing environment variables first
        env_vars = ['BETFAIR_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 
                   'HUGGINGFACE_API_KEY', 'WANDB_API_KEY', 'COMET_API_KEY']
        for var in env_vars:
            os.environ.pop(var, None)
        
        # Create empty .env file
        self.temp_env_file.write_text("")
        
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=True)
        loaded_vars = env_manager.load_env()
        
        # Should handle empty file gracefully
        assert loaded_vars == {}
    
    def test_malformed_env_file(self):
        """Test handling of malformed .env file"""
        # Create malformed .env file
        malformed_content = """OPENAI_API_KEY=sk-test123456789
INVALID_LINE_WITHOUT_EQUALS
BETFAIR_API_KEY=betfair-test-key
=VALUE_WITHOUT_KEY
"""
        self.temp_env_file.write_text(malformed_content)
        
        env_manager = EnvManager(env_file=str(self.temp_env_file), auto_load=True)
        
        # Should still load valid keys
        assert env_manager.get_api_key('openai') == 'sk-test123456789'
        assert env_manager.get_api_key('betfair') == 'betfair-test-key'


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 