"""
Concrete query strategy classes.
"""
import logging
logger = logging.getLogger(__name__)

from .active_learning_by_learning import ActiveLearningByLearning
try:
    from .hintsvm import HintSVM
except ImportError:
    logger.warn('HintSVM library not found, not importing.')
from .uncertainty_sampling import UncertaintySampling
from .query_by_committee import QueryByCommittee
from .quire import QUIRE
from .random_sampling import RandomSampling
from .variance_reduction import VarianceReduction
