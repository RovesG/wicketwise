# Purpose: Device detection and logging utilities for PyTorch
# Author: WicketWise Team, Last Modified: 2024-07-19

"""
This module provides utilities for detecting the best available PyTorch device
and logging device information. It supports CUDA, Apple MPS (Metal), and CPU
backends with automatic fallback logic.

Functions:
- get_best_device(): Detect and return the best available device
- log_device_details(): Log detailed information about the selected device
"""

import torch
import platform
from typing import Union


def _check_mps_available() -> bool:
    """
    Check if MPS (Metal Performance Shaders) backend is available.
    
    Returns:
        bool: True if MPS is available, False otherwise
    """
    try:
        return torch.backends.mps.is_available()
    except AttributeError:
        # MPS backend not available in this PyTorch version
        return False


def get_best_device() -> torch.device:
    """
    Detect and return the best available PyTorch device.
    
    Priority order:
    1. CUDA GPU (if available)
    2. Apple MPS/Metal (if on Darwin/macOS and available)
    3. CPU (fallback)
    
    Returns:
        torch.device: The best available device for computation
    """
    # Check for CUDA availability first (highest priority)
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    # Check for Apple MPS on macOS
    if platform.system() == "Darwin" and _check_mps_available():
        return torch.device("mps")
    
    # Fallback to CPU
    return torch.device("cpu")


def log_device_details(device: torch.device) -> None:
    """
    Log detailed information about the specified device.
    
    Args:
        device: The PyTorch device to log information about
    """
    device_type = device.type.lower()
    
    if device_type == "cuda":
        # Get CUDA device name
        device_name = torch.cuda.get_device_name(device.index if device.index is not None else 0)
        print(f"🖥 Using CUDA GPU: {device_name}")
    elif device_type == "mps":
        print("🍏 Using Apple MPS (Metal) backend")
    elif device_type == "cpu":
        print("💻 Using CPU")
    else:
        # Handle any other device types gracefully
        print(f"🔧 Using device: {device}")


def get_device_info() -> dict:
    """
    Get comprehensive device information for debugging and logging.
    
    Returns:
        dict: Dictionary containing device availability and system information
    """
    info = {
        "best_device": str(get_best_device()),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": _check_mps_available(),
        "platform": platform.system(),
        "platform_release": platform.release(),
        "torch_version": torch.__version__,
    }
    
    # Add CUDA-specific information if available
    if torch.cuda.is_available():
        info.update({
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_current_device": torch.cuda.current_device(),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_memory_allocated": torch.cuda.memory_allocated(0),
            "cuda_memory_reserved": torch.cuda.memory_reserved(0),
        })
    
    return info


def set_device_for_model(model: torch.nn.Module, device: Union[torch.device, str, None] = None) -> torch.nn.Module:
    """
    Move a PyTorch model to the specified device or the best available device.
    
    Args:
        model: PyTorch model to move
        device: Target device (if None, uses get_best_device())
        
    Returns:
        torch.nn.Module: Model moved to the specified device
    """
    if device is None:
        device = get_best_device()
    
    if isinstance(device, str):
        device = torch.device(device)
    
    model = model.to(device)
    log_device_details(device)
    
    return model


# Example usage and demonstration
if __name__ == "__main__":
    print("🔍 Device Detection Utility")
    print("=" * 40)
    
    # Detect best device
    best_device = get_best_device()
    print(f"Best available device: {best_device}")
    
    # Log device details
    log_device_details(best_device)
    
    # Show comprehensive device info
    print("\n📊 Device Information:")
    device_info = get_device_info()
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # Test with a simple model
    print("\n🧪 Model Device Test:")
    simple_model = torch.nn.Linear(10, 1)
    simple_model = set_device_for_model(simple_model, best_device)
    print(f"Model device: {next(simple_model.parameters()).device}")