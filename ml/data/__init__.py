"""
AquaSense AI - ML Data Handling
"""

from .loader import (
    load_groundwater_dataset,
    split_by_unseen_wells,
    split_by_temporal_cutoff,
)

__all__ = [
    "load_groundwater_dataset",
    "split_by_unseen_wells",
    "split_by_temporal_cutoff",
]
