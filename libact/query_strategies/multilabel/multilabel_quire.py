"""Multi-label Active Learning with Auxiliary Learner
"""
import copy

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel,\
    rbf_kernel

from libact.base.dataset import Dataset
from libact.base.interfaces import QueryStrategy, ContinuousModel
from libact.utils import inherit_docstring_from, seed_random_state, zip
from libact.models import LogisticRegression, SVM
from libact.models.multilabel import BinaryRelevance, DummyClf


class MultilabelQUIRE(QueryStrategy):
    r"""Multi-label Querying Informative and Representative Examples

    Parameters
    ----------
    lamba : float, optional default: 1.
        Regularization term.

    kernel : {'linear', 'poly', 'rbf', callable}, optional (default='rbf')
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf', or a callable.
        If a callable is given it is used to pre-compute the kernel matrix
        from data matrices; that matrix should be an array of shape
        ``(n_samples, n_samples)``.

    degree : int, optional (default=3)
        Degree of the polynomial kernel function ('poly').
        Ignored by all other kernels.

    gamma : float, optional (default=1.)
        Kernel coefficient for 'rbf', 'poly'.

    coef0 : float, optional (default=1.)
        Independent term in kernel function.
        It is only significant in 'poly'.

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

    Attributes
    ----------

    Examples
    --------
    Here is an example of declaring a multilabel with auxiliary learner
    query_strategy object:

    .. code-block:: python

       from libact.query_strategies.multilabel import MultilabelWithAuxiliaryLearner
       from libact.models.multilabel import BinaryRelevance
       from libact.models import LogisticRegression, SVM

       qs = MultilabelWithAuxiliaryLearner(
                dataset,
                major_learner=BinaryRelevance(LogisticRegression())
                auxiliary_learner=BinaryRelevance(SVM())
            )

    References
    ----------
    .. [1] Huang, S. J., R. Jin, and Z. H. Zhou. "Active Learning by Querying
           Informative and Representative Examples." IEEE transactions on
           pattern analysis and machine intelligence 36.10 (2014): 1936.
    """

    def __init__(self, dataset, lamba=1.0, kernel='rbf', gamma=1., coef0=1.,
            degree=3, random_state=None):
        super(MultilabelQUIRE, self).__init__(dataset)

        self.lamba = lamba

        X, _ = zip(*dataset.get_entries())
        self.kernel = kernel
        if self.kernel == 'rbf':
            self.K = rbf_kernel(X=X, Y=X, gamma=kwargs.pop('gamma', 1.))
        elif self.kernel == 'poly':
            self.K = polynomial_kernel(X=X,
                                       Y=X,
                                       coef0=kwargs.pop('coef0', 1),
                                       degree=kwargs.pop('degree', 3),
                                       gamma=kwargs.pop('gamma', 1.))
        elif self.kernel == 'linear':
            self.K = linear_kernel(X=X, Y=X)
        elif hasattr(self.kernel, '__call__'):
            self.K = self.kernel(X=np.array(X), Y=np.array(X))
        else:
            raise NotImplementedError

        self.random_state_ = seed_random_state(random_state)

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        X, Y = zip(*dataset.get_entries())
        _, lbled_Y = zip(*dataset.get_labeled_entries())

        X = np.array(X)
        K = self.K
        n = len(X)
        m = np.shape(lbled_Y)[1]
        lamba = self.lamba

        # index for labeled and unlabeled instance
        l = np.array([i for i in range(len(Y)) if Y[i] is not None])
        l = np.tile(l, m)
        u = np.array([i for i in range(len(Y)) if Y[i] is None])
        u = np.tile(u, m)

        # label correlation matrix
        R = np.corrcoef(np.array(lbled_Y).T)
        R = np.nan_to_num(R)

        L = lamba * (np.linalg.pinv(np.kron(R, K) + lamba * np.eye(n*m)))
        inv_L = np.linalg.pinv(L)

        vecY = np.reshape(np.array([y for y in Y if y is not None]), (-1, 1))
        invLuu = np.linalg.pinv(L[np.ix_(u, u)])

        score = np.zeros((n, m))
        for a in range(n):
            for b in range(m):
                s = b*n + a
                U = np.dot(L[np.ix_(u, l)], vecY) + L[np.ix_(u, [s])]
                temp1 = 2 * np.dot(L[[s], l], vecY) \
                        - np.dot(np.dot(U.T, invLuu), U)
                U = np.dot(L[np.ix_(u, l)], vecY)
                temp0 = -(np.dot(np.dot(U.T, invLuu), U))
                score[a, b] = L[s, s] \
                              + np.dot(np.dot(vecY.T, L[np.ix_(l, l)]),
                                       vecY)[0, 0]\
                              + np.max((temp1[0, 0], temp0[0, 0]))

        score = np.sum(score, axis=1)

        ask_id = self.random_state_.choice(np.where(score == np.min(score))[0])

        return ask_id
