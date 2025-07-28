"""
Pipeline package for image processing
"""

from .nudify_pipeline import NudifyPipeline
from .segmentation import HumanSegmentation

__all__ = ['NudifyPipeline', 'HumanSegmentation'] 