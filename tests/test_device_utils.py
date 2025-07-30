# Purpose: Unit tests for device detection and logging utilities
# Author: WicketWise Team, Last Modified: 2024-07-19

import pytest
import torch
import platform
from unittest.mock import patch, MagicMock
from io import StringIO
import sys

# Import the functions to test
import sys
sys.path.append('.')
from device_utils import get_best_device, log_device_details, get_device_info, set_device_for_model


class TestGetBestDevice:
    """Test suite for get_best_device function."""
    
    @patch('torch.cuda.is_available')
    def test_cuda_selected_when_available(self, mock_cuda_available):
        """Test that CUDA is selected when available."""
        mock_cuda_available.return_value = True
        
        device = get_best_device()
        
        assert device.type == "cuda", "Should select CUDA when available"
        mock_cuda_available.assert_called_once()
    
    @patch('device_utils._check_mps_available')
    @patch('platform.system')
    @patch('torch.cuda.is_available')
    def test_mps_selected_on_darwin_when_available(self, mock_cuda_available, mock_platform_system, mock_mps_available):
        """Test that MPS is selected on Darwin when CUDA is not available but MPS is."""
        mock_cuda_available.return_value = False
        mock_platform_system.return_value = "Darwin"
        mock_mps_available.return_value = True
        
        device = get_best_device()
        
        assert device.type == "mps", "Should select MPS on Darwin when available and CUDA is not"
        mock_cuda_available.assert_called_once()
        mock_platform_system.assert_called_once()
        mock_mps_available.assert_called_once()
    
    @patch('device_utils._check_mps_available')
    @patch('platform.system')
    @patch('torch.cuda.is_available')
    def test_cpu_selected_when_mps_unavailable_on_darwin(self, mock_cuda_available, mock_platform_system, mock_mps_available):
        """Test that CPU is selected on Darwin when neither CUDA nor MPS are available."""
        mock_cuda_available.return_value = False
        mock_platform_system.return_value = "Darwin"
        mock_mps_available.return_value = False
        
        device = get_best_device()
        
        assert device.type == "cpu", "Should select CPU when neither CUDA nor MPS are available"
        mock_cuda_available.assert_called_once()
        mock_platform_system.assert_called_once()
        mock_mps_available.assert_called_once()
    
    @patch('platform.system')
    @patch('torch.cuda.is_available')
    def test_cpu_selected_on_non_darwin_without_cuda(self, mock_cuda_available, mock_platform_system):
        """Test that CPU is selected on non-Darwin systems without CUDA."""
        mock_cuda_available.return_value = False
        mock_platform_system.return_value = "Linux"
        
        device = get_best_device()
        
        assert device.type == "cpu", "Should select CPU on non-Darwin systems without CUDA"
        mock_cuda_available.assert_called_once()
        mock_platform_system.assert_called_once()
    
    @patch('device_utils._check_mps_available')
    @patch('platform.system')
    @patch('torch.cuda.is_available')
    def test_cuda_priority_over_mps(self, mock_cuda_available, mock_platform_system, mock_mps_available):
        """Test that CUDA has priority over MPS when both are available."""
        mock_cuda_available.return_value = True
        mock_platform_system.return_value = "Darwin"
        mock_mps_available.return_value = True
        
        device = get_best_device()
        
        assert device.type == "cuda", "Should select CUDA over MPS when both are available"
        mock_cuda_available.assert_called_once()
        # platform.system and _check_mps_available should not be called when CUDA is available
        mock_platform_system.assert_not_called()
        mock_mps_available.assert_not_called()
    
    @patch('torch.cuda.is_available')
    def test_return_type_is_torch_device(self, mock_cuda_available):
        """Test that the function returns a torch.device object."""
        mock_cuda_available.return_value = False
        
        device = get_best_device()
        
        assert isinstance(device, torch.device), "Should return a torch.device object"


class TestLogDeviceDetails:
    """Test suite for log_device_details function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.held, sys.stdout = sys.stdout, StringIO()
    
    def tearDown(self):
        """Clean up test fixtures."""
        sys.stdout = self.held
    
    @patch('torch.cuda.get_device_name')
    def test_cuda_device_logging(self, mock_get_device_name):
        """Test logging for CUDA device."""
        mock_get_device_name.return_value = "NVIDIA GeForce RTX 3080"
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        cuda_device = torch.device("cuda")
        log_device_details(cuda_device)
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        assert "🖥 Using CUDA GPU: NVIDIA GeForce RTX 3080" in output
        mock_get_device_name.assert_called_once_with(0)
    
    @patch('torch.cuda.get_device_name')
    def test_cuda_device_with_index_logging(self, mock_get_device_name):
        """Test logging for CUDA device with specific index."""
        mock_get_device_name.return_value = "NVIDIA GeForce RTX 4090"
        
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        cuda_device = torch.device("cuda:1")
        log_device_details(cuda_device)
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        assert "🖥 Using CUDA GPU: NVIDIA GeForce RTX 4090" in output
        mock_get_device_name.assert_called_once_with(1)
    
    def test_mps_device_logging(self):
        """Test logging for MPS device."""
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        mps_device = torch.device("mps")
        log_device_details(mps_device)
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        assert "🍏 Using Apple MPS (Metal) backend" in output
    
    def test_cpu_device_logging(self):
        """Test logging for CPU device."""
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        cpu_device = torch.device("cpu")
        log_device_details(cpu_device)
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        assert "💻 Using CPU" in output
    
    def test_unknown_device_logging(self):
        """Test logging for unknown device types."""
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        # Create a mock device with unknown type
        unknown_device = MagicMock()
        unknown_device.type = "unknown"
        
        log_device_details(unknown_device)
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        assert "🔧 Using device:" in output


class TestGetDeviceInfo:
    """Test suite for get_device_info function."""
    
    @patch('torch.cuda.memory_reserved')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.current_device')
    @patch('torch.cuda.device_count')
    @patch('torch.cuda.is_available')
    @patch('torch.backends.mps.is_available')
    @patch('platform.release')
    @patch('platform.system')
    @patch('device_utils.get_best_device')
    def test_device_info_with_cuda(self, mock_get_best_device, mock_platform_system, mock_platform_release,
                                   mock_mps_available, mock_cuda_available, mock_cuda_device_count,
                                   mock_cuda_current_device, mock_cuda_get_device_name,
                                   mock_cuda_memory_allocated, mock_cuda_memory_reserved):
        """Test get_device_info when CUDA is available."""
        # Setup mocks
        mock_get_best_device.return_value = torch.device("cuda")
        mock_platform_system.return_value = "Linux"
        mock_platform_release.return_value = "5.4.0"
        mock_mps_available.return_value = False
        mock_cuda_available.return_value = True
        mock_cuda_device_count.return_value = 2
        mock_cuda_current_device.return_value = 0
        mock_cuda_get_device_name.return_value = "NVIDIA RTX 3080"
        mock_cuda_memory_allocated.return_value = 1024 * 1024 * 1024  # 1GB
        mock_cuda_memory_reserved.return_value = 2 * 1024 * 1024 * 1024  # 2GB
        
        info = get_device_info()
        
        # Verify basic info
        assert info["best_device"] == "cuda"
        assert info["cuda_available"] is True
        assert info["mps_available"] is False
        assert info["platform"] == "Linux"
        assert info["platform_release"] == "5.4.0"
        assert "torch_version" in info
        
        # Verify CUDA-specific info
        assert info["cuda_device_count"] == 2
        assert info["cuda_current_device"] == 0
        assert info["cuda_device_name"] == "NVIDIA RTX 3080"
        assert info["cuda_memory_allocated"] == 1024 * 1024 * 1024
        assert info["cuda_memory_reserved"] == 2 * 1024 * 1024 * 1024
    
    @patch('device_utils._check_mps_available')
    @patch('torch.cuda.is_available')
    @patch('platform.release')
    @patch('platform.system')
    @patch('device_utils.get_best_device')
    def test_device_info_without_cuda(self, mock_get_best_device, mock_platform_system, mock_platform_release,
                                      mock_cuda_available, mock_mps_available):
        """Test get_device_info when CUDA is not available."""
        # Setup mocks
        mock_get_best_device.return_value = torch.device("cpu")
        mock_platform_system.return_value = "Darwin"
        mock_platform_release.return_value = "20.6.0"
        mock_cuda_available.return_value = False
        mock_mps_available.return_value = False
        
        info = get_device_info()
        
        # Verify basic info
        assert info["best_device"] == "cpu"
        assert info["cuda_available"] is False
        assert info["mps_available"] is False
        assert info["platform"] == "Darwin"
        assert info["platform_release"] == "20.6.0"
        assert "torch_version" in info
        
        # Verify no CUDA-specific info
        assert "cuda_device_count" not in info
        assert "cuda_current_device" not in info
        assert "cuda_device_name" not in info
        assert "cuda_memory_allocated" not in info
        assert "cuda_memory_reserved" not in info
    
    @patch('device_utils._check_mps_available')
    def test_device_info_without_mps_backend(self, mock_mps_available):
        """Test get_device_info when MPS backend is not available (older PyTorch)."""
        mock_mps_available.return_value = False
        
        info = get_device_info()
        
        assert info["mps_available"] is False


class TestSetDeviceForModel:
    """Test suite for set_device_for_model function."""
    
    def test_model_moved_to_specified_device(self):
        """Test that model is moved to the specified device."""
        model = torch.nn.Linear(10, 1)
        original_device = next(model.parameters()).device
        
        # Move to CPU explicitly
        result_model = set_device_for_model(model, "cpu")
        new_device = next(result_model.parameters()).device
        
        assert isinstance(result_model, torch.nn.Module), "Should return a model"
        assert new_device.type == "cpu", "Should move model to specified device"
    
    @patch('device_utils.get_best_device')
    def test_model_moved_to_best_device_when_none_specified(self, mock_get_best_device):
        """Test that model is moved to best device when none is specified."""
        mock_get_best_device.return_value = torch.device("cpu")
        
        model = torch.nn.Linear(5, 2)
        result_model = set_device_for_model(model, device=None)
        
        mock_get_best_device.assert_called_once()
        assert isinstance(result_model, torch.nn.Module), "Should return a model"
    
    def test_string_device_converted_to_torch_device(self):
        """Test that string device specifications are converted to torch.device."""
        model = torch.nn.Linear(3, 1)
        
        # Pass device as string
        result_model = set_device_for_model(model, "cpu")
        device = next(result_model.parameters()).device
        
        assert device.type == "cpu", "Should handle string device specification"
    
    @patch('device_utils.log_device_details')
    def test_device_details_logged(self, mock_log_device_details):
        """Test that device details are logged."""
        model = torch.nn.Linear(2, 1)
        
        set_device_for_model(model, "cpu")
        
        mock_log_device_details.assert_called_once()
        # Verify the device passed to log_device_details
        called_device = mock_log_device_details.call_args[0][0]
        assert called_device.type == "cpu"


class TestDeviceUtilsIntegration:
    """Integration tests for device utilities."""
    
    def test_end_to_end_device_selection_and_logging(self):
        """Test complete workflow from device selection to model setup."""
        # Get best device
        device = get_best_device()
        assert isinstance(device, torch.device)
        
        # Create and move model
        model = torch.nn.Linear(4, 2)
        model = set_device_for_model(model, device)
        
        # Verify model is on correct device
        model_device = next(model.parameters()).device
        assert model_device.type == device.type
    
    def test_device_info_completeness(self):
        """Test that device info contains all expected keys."""
        info = get_device_info()
        
        required_keys = [
            "best_device", "cuda_available", "mps_available",
            "platform", "platform_release", "torch_version"
        ]
        
        for key in required_keys:
            assert key in info, f"Device info should contain {key}"
    
    @patch('torch.cuda.is_available')
    @patch('device_utils._check_mps_available')
    @patch('platform.system')
    def test_device_selection_priority_order(self, mock_platform_system, mock_mps_available, mock_cuda_available):
        """Test that device selection follows the correct priority order."""
        # Test all combinations
        test_cases = [
            # (cuda_available, platform, mps_available, expected_device)
            (True, "Darwin", True, "cuda"),     # CUDA has highest priority
            (True, "Darwin", False, "cuda"),    # CUDA over no MPS
            (True, "Linux", False, "cuda"),     # CUDA on non-Darwin
            (False, "Darwin", True, "mps"),     # MPS on Darwin when no CUDA
            (False, "Darwin", False, "cpu"),    # CPU when no CUDA/MPS on Darwin
            (False, "Linux", False, "cpu"),     # CPU on non-Darwin without CUDA
        ]
        
        for cuda_avail, platform_sys, mps_avail, expected in test_cases:
            mock_cuda_available.return_value = cuda_avail
            mock_platform_system.return_value = platform_sys
            mock_mps_available.return_value = mps_avail
            
            device = get_best_device()
            assert device.type == expected, f"Failed for case: CUDA={cuda_avail}, Platform={platform_sys}, MPS={mps_avail}"


if __name__ == "__main__":
    # Run specific test for debugging
    pytest.main([__file__ + "::TestGetBestDevice::test_cuda_selected_when_available", "-v"])