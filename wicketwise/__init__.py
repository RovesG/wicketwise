# Purpose: Main WicketWise package initialization
# Author: WicketWise Team, Last Modified: 2024-12-07

"""
WicketWise: Advanced Cricket Analytics and Prediction System

A comprehensive cricket analysis platform combining machine learning,
real-time data processing, and tactical intelligence for T20 cricket.
"""

__version__ = "1.0.0"
__author__ = "WicketWise Team"

# Core imports
from .core import DataIngestor, FeatureGenerator, InningsPredictor

__all__ = [
    "DataIngestor",
    "FeatureGenerator", 
    "InningsPredictor",
    "__version__",
    "__author__"
] 