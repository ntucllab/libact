"""Hinted Support Vector Machine

This module contains a class that implements Hinted Support Vector Machine, an
active learning algorithm.

Standalone hintsvm can be retrieved from https://github.com/yangarbiter/hintsvm
"""
import numpy as np

from libact.base.interfaces import QueryStrategy
from libact.query_strategies._hintsvm import hintsvm_query
from libact.utils import inherit_docstring_from, seed_random_state, zip


class HintSVM(QueryStrategy):

    r"""Hinted Support Vector Machine

    Hinted Support Vector Machine is an active learning algorithm within the
    hined sampling framework with an extended support vector machine.

    Parameters
    ----------
    Cl : float, >0, optional (default=0.1)
        The weight of the classification error on labeled pool.

    Ch : float, >0, optional (default=0.1)
        The weight of the hint error on hint pool.

    p : float, >0 and <=1, optional (default=.5)
        The probability to select an instance from unlabeld pool to hint pool.

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, optional (default='linear')
                linear: u'\*v
                poly: (gamma\*u'\*v + coef0)^degree
                rbf: exp(-gamma\*|u-v|^2)
                sigmoid: tanh(gamma\*u'\*v + coef0)

    degree : int, optional (default=3)
        Parameter for kernel function.

    gamma : float, optional (default=0.1)
        Parameter for kernel function.

    coef0 : float, optional (default=0.)
        Parameter for kernel function.

    tol : float, optional (default=1e-3)
        Tolerance of termination criterion.

    shrinking : {0, 1}, optional (default=1)
        Whether to use the shrinking heuristics.

    cache_size : float, optional (default=100.)
        Set cache memory size in MB.

    verbose : int, optional (default=0)
        Set verbosity level for hintsvm solver.

    Attributes
    ----------
    random_states\_ : np.random.RandomState instance
        The random number generator using.

    Examples
    --------
    Here is an example of declaring a HintSVM query_strategy object:

    .. code-block:: python

       from libact.query_strategies import HintSVM

       qs = HintSVM(
            dataset, # Dataset object
            Cl=0.01,
            p=0.8,
            )

    References
    ----------
    .. [1] Li, Chun-Liang, Chun-Sung Ferng, and Hsuan-Tien Lin. "Active Learning
           with Hinted Support Vector Machine." ACML. 2012.

    .. [2] Chun-Liang Li, Chun-Sung Ferng, and Hsuan-Tien Lin. Active learning
           using hint information. Neural Computation, 27(8):1738--1765, August
           2015.

    """

    def __init__(self, *args, **kwargs):
        super(HintSVM, self).__init__(*args, **kwargs)

        # Weight on labeled data's classification error
        self.cl = kwargs.pop('Cl', 0.1)
        if self.cl <= 0:
            raise ValueError('Parameter Cl should be greater than 0.')

        # Weight on hinted data's classification error
        self.ch = kwargs.pop('Ch', 0.1)
        if self.ch <= 0:
            raise ValueError('Parameter Cl should be greater than 0.')

        # Prabability of sampling a data from unlabeled pool to hinted pool
        self.p = kwargs.pop('p', 0.5)
        if self.p > 1.0 or self.p < 0.0:
            raise ValueError(
                'Parameter p should be greater than or equal to 0 and less '
                'than or equal to 1.'
            )

        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)

        # svm solver parameters
        self.svm_params = {}
        self.svm_params['kernel'] = kwargs.pop('kernel', 'linear')
        self.svm_params['degree'] = kwargs.pop('degree', 3)
        self.svm_params['gamma'] = kwargs.pop('gamma', 0.1)
        self.svm_params['coef0'] = kwargs.pop('coef0', 0.)
        self.svm_params['tol'] = kwargs.pop('tol', 1e-3)
        self.svm_params['shrinking'] = kwargs.pop('shrinking', 1)
        self.svm_params['cache_size'] = kwargs.pop('cache_size', 100.)
        self.svm_params['verbose'] = kwargs.pop('verbose', 0)

        self.svm_params['C'] = self.cl

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        unlabeled_entry_ids, unlabeled_pool = dataset.get_unlabeled_entries()
        labeled_pool, y = dataset.get_labeled_entries()
        if len(np.unique(y)) > 2:
            raise ValueError("HintSVM query strategy support binary class "
                "active learning only. Found %s classes" % len(np.unique(y)))

        hint_pool_idx = self.random_state_.choice(
            len(unlabeled_pool), int(len(unlabeled_pool) * self.p))
        hint_pool = np.array(unlabeled_pool)[hint_pool_idx]

        weight = [1.0 for _ in range(len(labeled_pool))] +\
                 [(self.ch / self.cl) for _ in range(len(hint_pool))]
        y = list(y) + [0 for _ in range(len(hint_pool))]
        X = [x for x in labeled_pool] +\
            [x for x in hint_pool]

        p_val = hintsvm_query(
            np.array(X, dtype=np.float64),
            np.array(y, dtype=np.float64),
            np.array(weight, dtype=np.float64),
            np.array(unlabeled_pool, dtype=np.float64),
            self.svm_params)

        p_val = [abs(float(val[0])) for val in p_val]
        idx = int(np.argmax(p_val))
        return unlabeled_entry_ids[idx]
