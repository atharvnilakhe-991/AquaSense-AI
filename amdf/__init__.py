"""
AquaSense AI - Adaptive Multimodal Data Fusion Engine (AMDFE)
Member 2: Data Engineering, Quality, Reliability, and Multimodal Fusion.
"""

from .quality import assess_data_quality, calculate_temporal_irregularity
from .reliability import estimate_source_reliability, compute_modality_reliability_vector
from .fusion import AdaptiveMultimodalDataFusionEngine

__all__ = [
    "assess_data_quality",
    "calculate_temporal_irregularity",
    "estimate_source_reliability",
    "compute_modality_reliability_vector",
    "AdaptiveMultimodalDataFusionEngine",
]
