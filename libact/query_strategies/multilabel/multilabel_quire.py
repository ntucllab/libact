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
            self.K = rbf_kernel(X=X, Y=X, gamma=gamma)
        elif self.kernel == 'poly':
            self.K = polynomial_kernel(X=X, Y=X, coef0=coef0, degree=degree,
                    gamma=gamma)
        elif self.kernel == 'linear':
            self.K = linear_kernel(X=X, Y=X)
        elif hasattr(self.kernel, '__call__'):
            self.K = self.kernel(X=np.array(X), Y=np.array(X))
        else:
            raise NotImplementedError

        self.random_state_ = seed_random_state(random_state)


        _, lbled_Y = zip(*dataset.get_labeled_entries())
        self.n_labels = np.shape(lbled_Y)[1]
        n = len(X)
        m = self.n_labels
        # label correlation matrix
        R = np.corrcoef(np.array(lbled_Y).T)
        R = np.nan_to_num(R)
        self.RK = np.kron(R, self.K)

        self.L = lamba * (np.linalg.pinv(self.RK + lamba * np.eye(n*m)))

    def _get_index(self):
        _, Y = zip(*self.dataset.get_entries())
        n_instance = len(Y)
        m = self.n_labels
        # index for labeled and unlabeled instance
        l_id = []
        a_id = []
        for i in range(n_instance * m):
            if Y[i // m] is None:
                a_id.append(i)
            else:
                l_id.append(i)
        return a_id, l_id

    #def update(self, entry_id, label):
    #    # calculate invLaa
    #    invLaa = self.invLaa
    #    # idx before update
    #    a_id, l_id = self.idxs
    #    m = len(label)
    #    # assert len(np.where(np.array(a_id) == entry_id*m)[0]) == 1
    #    idx = np.where(np.array(a_id) == entry_id*m)[0][0]
    #    for i in range(m):
    #        D = np.delete(np.delete(invLaa, idx, axis=0), idx, axis=1)
    #        b = np.delete(invLaa, idx, axis=0)[:, idx]
    #        # invLuu
    #        invLaa = D - 1./invLaa[idx, idx] * np.dot(b, b.T)
    #    self.invLaa = invLaa

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        X, Y = zip(*dataset.get_entries())
        X = np.array(X)

        n_instance = len(X)
        m = self.n_labels
        RK = self.RK
        lamba = self.lamba
        L = self.L

        a_id, l_id = self._get_index()
        # invLaa = np.linalg.pinv(L[np.ix_(a_id, a_id)])
        invLaa = ((lamba * np.eye(len(a_id)) + RK[np.ix_(a_id, a_id)]) \
                 - np.dot(np.dot(RK[np.ix_(a_id, l_id)],
                                 np.linalg.pinv(lamba * np.eye(len(l_id)) \
                                                + RK[np.ix_(l_id, l_id)])),
                          RK[np.ix_(l_id, a_id)])) / lamba

        vecY = np.reshape(np.array([y for y in Y if y is not None]).T, (-1, 1))
        detLaa = np.linalg.det(L[np.ix_(a_id, a_id)])

        score = np.zeros(len(a_id))
        b = np.zeros((len(a_id)-1))
        D = np.zeros((len(a_id)-1, len(a_id)-1))
        D[...] = invLaa[1:, 1:]
        for i, s in enumerate(a_id):
            # L -> s, Laa -> i
            u_id = a_id[:i] + a_id[i+1:]
            #D = np.delete(np.delete(invLaa, i, axis=0), i, axis=1)
            if i > 0:
                D[(i-1), :i] = invLaa[(i-1), :i]
                D[(i-1), i:] = invLaa[(i-1), (i+1):]
                D[:i, (i-1)] = invLaa[:i, (i-1)]
                D[i:, (i-1)] = invLaa[(i+1):, (i-1)]
            #D[:i, :i] = invLaa[:i, :i]
            #D[i:, i:] = invLaa[i+1:, i+1:]
            #D[:i, i:] = invLaa[:i, i+1:]
            #D[i:, :i] = invLaa[i+1:, :i]

            #b = np.delete(invLaa, i, axis=0)[:, i]
            b[:i] = invLaa[:i, i]
            b[i:] = invLaa[i+1:, i]
            invLuu = D - 1./invLaa[i, i] * np.dot(b, b.T)

            score[i] = L[s, s] - detLaa / L[s, s] \
                       + 2 * np.abs(np.dot(L[s, l_id] \
                                    - np.dot(np.dot(L[s, u_id], invLuu),
                                      L[np.ix_(u_id, l_id)]), vecY))

        score = np.sum(score.reshape(m, -1).T, axis=1)

        ask_idx = self.random_state_.choice(np.where(score == np.min(score))[0])

        return a_id[ask_idx] // m
