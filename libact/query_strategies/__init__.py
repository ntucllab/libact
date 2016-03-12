"""
Concrete query strategy classes.
"""
import os
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
import logging
logger = logging.getLogger(__name__)

from .active_learning_by_learning import ActiveLearningByLearning
from .hintsvm import HintSVM
from .uncertainty_sampling import UncertaintySampling
from .query_by_committee import QueryByCommittee
from .quire import QUIRE
from .random_sampling import RandomSampling
#don't import c extentions when on readthedocs server
if not on_rtd:
    from ._variance_reduction import estVar
from .variance_reduction import VarianceReduction
