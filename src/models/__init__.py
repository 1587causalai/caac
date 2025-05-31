"""
Models module for Shared Cauchy OvR Classifier

This package contains the implementation of the novel Shared Latent Cauchy Vector
based One-vs-Rest (OvR) Classifier as described in theory_v2.md.
"""

from .shared_cauchy_ovr import SharedCauchyOvRClassifier
from .loss_functions import (
    OvRBinaryCrossEntropyLoss,
    WeightedOvRBCELoss,
    FocalOvRLoss,
    UncertaintyRegularizedLoss,
    create_loss_function
)
from .trainer import SharedCauchyOvRTrainer

__all__ = [
    'SharedCauchyOvRClassifier',
    'OvRBinaryCrossEntropyLoss',
    'WeightedOvRBCELoss', 
    'FocalOvRLoss',
    'UncertaintyRegularizedLoss',
    'create_loss_function',
    'SharedCauchyOvRTrainer'
]
