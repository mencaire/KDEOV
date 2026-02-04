"""
KDEOV Model Package
Knowledge Distillation for Efficient Open-Vocabulary Vision
"""

from .kdeov_model import KDEOVModel
from .components import (
    FrozenCLIPTextEncoder,
    LightweightVisualBackbone,
    ProjectionNetwork,
    CrossModalFusionModule,
    SpatialProjection,
    grid_boxes_to_image,
)
from .losses import (
    DistillationLoss,
    CrossModalAlignmentLoss,
    FeatureAlignmentLoss
)

__all__ = [
    'KDEOVModel',
    'FrozenCLIPTextEncoder',
    'LightweightVisualBackbone',
    'ProjectionNetwork',
    'CrossModalFusionModule',
    'SpatialProjection',
    'grid_boxes_to_image',
    'DistillationLoss',
    'CrossModalAlignmentLoss',
    'FeatureAlignmentLoss',
]

