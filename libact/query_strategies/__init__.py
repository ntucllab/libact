"""
Concrete query strategy classes.
"""

from .active_learning_by_learning import ActiveLearningByLearning
from .hintsvm import HintSVM
from .uncertainty_sampling import UncertaintySampling
from .query_by_committee import QueryByCommittee
from .quire import QUIRE
from .random_sampling import RandomSampling
from .variance_reduction import VarianceReduction
