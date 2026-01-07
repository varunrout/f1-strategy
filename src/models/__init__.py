# F1 Tyre Degradation Modeling Module
"""
This module provides tyre degradation prediction models trained on 2022-2024 F1 data.
"""

from .degradation_model import DegradationModel
from .feature_builder import FeatureBuilder

__all__ = ['DegradationModel', 'FeatureBuilder']
