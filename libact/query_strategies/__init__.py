"""
Concrete query strategy classes.
"""
from __future__ import absolute_import

import os
ON_RTD = os.environ.get('READTHEDOCS', None) == 'True'
import logging
LOGGER = logging.getLogger(__name__)

from .active_learning_by_learning import ActiveLearningByLearning
from .hintsvm import HintSVM
from .uncertainty_sampling import UncertaintySampling
from .query_by_committee import QueryByCommittee
from .quire import QUIRE
from .random_sampling import RandomSampling
from .density_weighted_uncertainty_sampling import DWUS
# don't import c extentions when on readthedocs server
if not ON_RTD:
    from ._variance_reduction import estVar
from .variance_reduction import VarianceReduction
