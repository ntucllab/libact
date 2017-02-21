"""Adaptive active learning
"""
import copy

import numpy as np
from sklearn.svm import SVC

from libact.base.dataset import Dataset
from libact.base.interfaces import QueryStrategy, ContinuousModel
from libact.utils import inherit_docstring_from, seed_random_state, zip
from libact.models.multilabel import BinaryRelevance


class AdaptiveActiveLearning(QueryStrategy):
    r"""Adaptive Active Learning

    This approach combines Max Margin Uncertainty Sampling and Label
    Cardinality Inconsistency.

    Parameters
    ----------
    base_clf : ContinuousModel object instance
        The base learner for binary relavance.

    betas : list of float, 0 <= beta <= 1, default: [0., 0.1, ..., 0.9, 1.]
        List of trade-off parameter that balances the relative importance
        degrees of the two measures.

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

    Attributes
    ----------

    Examples
    --------
    Here is an example of declaring a MMC query_strategy object:

    .. code-block:: python

       from libact.query_strategies.multilabel import AdaptiveActiveLearning
       from sklearn.linear_model import LogisticRegression

       qs = AdaptiveActiveLearning(
           dataset, # Dataset object
           base_clf=LogisticRegression()
       )

    References
    ----------
    .. [1] Li, Xin, and Yuhong Guo. "Active Learning with Multi-Label SVM
           Classification." IJCAI. 2013.
    """

    def __init__(self, dataset, base_clf, betas=None, random_state=None):
        super(BinaryMinimization, self).__init__(dataset)

        self.n_labels = len(self.dataset.data[0][1])

        self.base_clf = copy.deepcopy(base_clf)

        # TODO check beta value
        self.betas = betas
        if self.betas is None:
            self.betas = [i/10. for i in range(0, 11)]

        self.random_state_ = seed_random_state(random_state)

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        X, Y = zip(*dataset.get_labeled_entries())
        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())
        Y = np.array(Y)
        X_pool = np.array(X_pool)

        clf = BinaryRelevance(self.base_clf)
        clf.train(dataset)
        real = clf.predict_real(X_pool)
        pred = clf.predict(X_pool)

        # Separation Margin
        separation_margin = -real.min(axis=1) + real.max(axis=0)
        uncertainty = 1. / separation_margin

        # Label Cardinality Inconsistency
        average_pos_lbl = Y.mean(axis=0)
        label_cardinality = pred - np.tile(average_pos_lbl, (len(pred), 1))
        label_cardinality = np.sqrt((label_cardinality**2).sum(axis=1))

        candidate_idx_set = set()
        for b in self.betas:
            # score shape = (len(X_pool), )
            score = separation_margin**b * label_cardinality**(1.-b)
            for idx in np.where(score == np.max(score))[0]:
                candidate_idx_set.add(idx)

        approx_err = []
        candidates = list(candidate_idx_set)
        for idx in candidates:
            ds = Dataset(np.append(X, X_pool[idx]), np.append(Y, pred[idx]))
            br = BinaryRelevance(self.base_clf)
            br.train(dataset)
            br_real = br.predict_real(X_pool)

            br_real[br_real>0] = (1. - br_real[br_real>0])
            br_real[br_real>0] = np.maximum(0, br_real[br_real>0])

            br_real[br_real<0] = (1. + br_real[br_real<0])
            br_real[br_real<0] = np.maximum(0, br_real[br_real<0])

            approx_err.append(np.sum(br_real))

        choices = np.where(np.array(approx_err) == np.min(approx_err))[0]
        ask_idx = candidates[self.random_state_.choice(choices)]

        return unlabeled_entry_ids[ask_idx]
