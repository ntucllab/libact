"""
Concrete query strategy classes.
"""
from __future__ import absolute_import

from .adaptive_active_learning import AdaptiveActiveLearning
from .binary_minimization import BinaryMinimization
from .cost_sensitive_reference_pair_encoding import CostSensitiveReferencePairEncoding
from .maximum_margin_reduction import MaximumLossReductionMaximalConfidence as MMC
from .multilabel_with_auxiliary_learner import MultilabelWithAuxiliaryLearner

__all__ = [
    'AdaptiveActiveLearning',
    'BinaryMinimization',
    'CostSensitiveReferencePairEncoding',
    'MMC',
    'MultilabelWithAuxiliaryLearner'
]
