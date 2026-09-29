"""Versioned, offline evaluation of frozen Video2Scene artifacts."""
from .config import MetricConfig
from .image import compare_images
from .geometry import compare_points, sample_surface
from .instances import Instance, compare_instances
from .views import compare_views

__version__ = "0.1.0"
__all__ = ["MetricConfig", "compare_images", "compare_points", "sample_surface",
           "Instance", "compare_instances", "compare_views"]
