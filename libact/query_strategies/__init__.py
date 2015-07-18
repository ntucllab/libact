"""
Concrete query strategy classes.
"""

from .uncertainty_sampling import UncertaintySampling
from .query_by_committee import QueryByCommittee
from .quire import QUIRE

try:
    from .hintsvm import HintSVM
except ImportError:
    # HintSVM library not found, not importing
    pass
