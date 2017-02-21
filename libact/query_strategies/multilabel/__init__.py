"""
Concrete query strategy classes.
"""
from __future__ import absolute_import

from .maximum_margin_reduction import MaximumLossReductionMaximalConfidence as MMC
from .multilable_with_auxiliary_learner import MultilabelWithAuxiliaryLearner
from .binary_minimization import BinaryMinimization
from .adaptive_active_learning import AdaptiveActiveLearning
