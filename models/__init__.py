"""
KDEOV Model Package
Knowledge Distillation for Efficient Open-Vocabulary Vision
"""

from .kdeov_model import KDEOVModel
from .components import (
    FrozenCLIPTextEncoder,
    LightweightVisualBackbone,
    ProjectionNetwork,
    CrossModalFusionModule
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
    'DistillationLoss',
    'CrossModalAlignmentLoss',
    'FeatureAlignmentLoss',
]

