"""ImportanceWeighted
"""
from __future__ import division
import copy
import logging

import numpy as np
from keras.models import Model
from keras.optimizers import Nadam
from keras.regularizers import l2
from keras.layers import Input, Dense
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder

from libact.base.interfaces import QueryStrategy, ProbabilisticModel
from libact.utils import inherit_docstring_from, seed_random_state, zip
from libact.base.dataset import Dataset

LOGGER = logging.getLogger(__name__)


class ImportanceWeighted(QueryStrategy):
    """Importance Weighted Active Learning Algorithm

    Support binary class and logistic regression only.

    Parameters
    ----------

    C0 : float, optional (default=8.)
        mellowness

    c1 : float, optional (default=1.)

    c2 : float, optional (default=1.)

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

    Attributes
    ----------

    Examples
    --------
    Here is an example of how to declare a ImportanceWeighted query_strategy object:

    .. code-block:: python

       from libact.query_strategies import ImportanceWeighted

       qs = ImportanceWeighted(dataset)

    References
    ----------
    .. [1] Beygelzimer, Alina, et al. "Agnostic active learning without
           constraints." Advances in Neural Information Processing Systems.
           2010.

    .. [2] https://github.com/yyysbysb/al_log_icml18

    .. [3] https://github.com/VowpalWabbit/vowpal_wabbit/blob/579c34d2d2fd151b419bea54d9921fc7f3f55bbc/vowpalwabbit/active.cc
    """

    def __init__(self, dataset, C0=8., c1=1., c2=1., random_state=None):
        super(ImportanceWeighted, self).__init__(dataset=dataset)

        X, y = dataset.format_sklearn()
        lbl_enc = OneHotEncoder(sparse=False)
        lbl_enc.fit(y.reshape(-1, 1))
        self.model = LogisticRegressionKeras(
            lbl_enc=lbl_enc, n_features=X.shape[1:], n_classes=len(np.unique(y)))
        self.model.fit(X, y)

        self.classes = np.unique(
            [d[1] for d in dataset.data if d[1] is not None]).tolist()

        self.C0 = C0
        self.c1 = c1
        self.c2 = c2
        self.random_state_ = seed_random_state(random_state)
        self.cnt = len(X)

        self.history = []
        X, y = zip(*self.dataset.get_labeled_entries())
        for i in range(len(X)):
            self.history.append((X[i], y[i], 1., ))

        self._gen = self._query_generator()
        self.prev_query = self._sample_a_query()

    def _sample_a_query(self):
        """
        Loop until getting a point sampled. Each time in the loop, get the
        probability of querying a point and then flip a coin to decide
        whether to query it.

        Return
        ------
        ask_id: int
            The data point to query.

        p_k: float, 0 <= p_k <= 1
            The probability of querying this point.
        """
        p_k = 0
        rng = self.random_state_.rand()
        while p_k < rng:
            try:
                ask_id, p_k = next(self._gen)
            except StopIteration:
                self._gen = self._query_generator()
                continue

            rng = self.random_state_.rand
            if rng <= p_k:
                return ask_id, p_k

    @inherit_docstring_from(QueryStrategy)
    def update(self, entry_id, label):
        if self.prev_query[0] != entry_id:
            raise ValueError(
                "The updated sample should be the same as the queried one."
                "Queried has id %d but get %d", self.prev_query[0], entry_id
            )
        self.cnt += 1
        updated = self.dataset.data[entry_id]
        self.history.append(
            (updated[0], updated[1], self.prev_query[1])
        )
        self.prev_query = self._sample_a_query()

        # retrain model
        X, y = self.dataset.format_sklearn()
        self.model.fit(X, y)

    @inherit_docstring_from(QueryStrategy)
    def _get_scores(self):
        pass

    def _importance_weighted_empirical_error(self, model):
        X, y, p = zip(*self.history)
        X, y, p = np.asarray(X), np.asarray(y), np.asarray(p)
        pred = model.predict_proba(X)[:, y]
        #return (1. / p * (pred == y)).sum() / len(y)
        return (1. / p * np.abs(pred - y)).sum() / len(y)

    def _query_generator(self):
        """Generator for the next point to query.

        Parameters
        ----------

        Yields
        ------
        ask_id: int
            The data point to query.

        p_k: float, 0 <= p_k <= 1
            The probability of querying this point.
        """
        dataset = self.dataset
        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())

        for k, x in enumerate(X_pool):
            X, y, p = zip(*self.history)
            X, y, p = np.asarray(X), np.asarray(y), np.asarray(p)
            #model = self.model.clone()
            #modelp = self.model.clone()
            #model.train(Dataset(X=X, y=y), sample_weight=1./p)
            #predy = model.predict(x.reshape(1, -1))
            #neg_predy = self.classes[int(not self.classes.index(predy[0]))]
            #modelp.train(
            #    Dataset(X=np.append(X, [x], axis=0), y=np.append(y, neg_predy)),
            #    sample_weight=1./np.append(p, 1e-8)
            #)
            #err_h = self._importance_weighted_empirical_error(model)
            #err_hp = self._importance_weighted_empirical_error(modelp)
            #gk = abs(err_hp - err_h)
            #del model
            #del modelp

            gk = self.model.calc_gap(x, self.cnt)
            if k == 0:
                _temp = np.inf
            else:
                _temp = self.C0 * np.log(k+1) / k
            thresh = np.sqrt(_temp) + _temp
            #print(gk, thresh, gk >= thresh)
            if gk <= thresh:
                yield unlabeled_entry_ids[k], 1.
            else:
                c1 = self.c1
                c2 = self.c2
                a = _temp + np.sqrt(_temp) - gk - np.sqrt(_temp) * c1 - _temp * c2
                b = np.sqrt(_temp) * c1
                c = _temp * c2
                s = max(
                    (-b - np.sqrt(b**2 - 4 * a * c)) / (2*a),
                    (-b + np.sqrt(b**2 - 4 * a * c)) / (2*a)
                )
                s = float(np.clip(s**2, 0, 1))
                yield unlabeled_entry_ids[k], s

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        return self.prev_query[0]


def logistic_regression_arch(input_shape, n_classes, l2_weight=0.0):
    inputs = Input(shape=input_shape)
    x = Dense(n_classes, activation='softmax')(inputs)
    return Model(inputs=[inputs], outputs=[x])

class LogisticRegressionKeras():

    def __init__(self, lbl_enc, n_features, n_classes, batch_size=128,
            epochs=100, optimizer='nadam', l2_weight=1e-6, random_state=None):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lbl_enc = lbl_enc
        self.optimizer = Nadam()
        self.l2_weight = l2_weight
        self.loss = 'categorical_crossentropy'
        self.random_state = random_state

        input_shape = tuple(n_features)
        model = logistic_regression_arch(input_shape, n_classes, l2_weight)
        model.compile(loss=self.loss,
                        optimizer=self.optimizer,
                        metrics=[])
        self.model = model

    def fit(self, X, y, sample_weight=None):
        Y = self.lbl_enc.transform(y.reshape(-1, 1))
        self.model.fit(X, Y, batch_size=self.batch_size, verbose=0,
                       epochs=self.epochs, sample_weight=sample_weight)

    def predict(self, X):
        pass
    
    def predict_proba(self, X):
        X = np.asarray(X)
        pred = self.model.predict(X)
        return np.hstack((1-pred, pred))

    def calc_gap(self, x, cnt):
        def stepsize(idx, stepsize_para0 = 0.05):
            return np.sqrt(stepsize_para0 / (stepsize_para0+idx))
        w = self.model.get_weights()[0]
        #return np.abs(2*np.inner(w, x) / (stepsize(cnt, self.l2_weight)*np.inner(x, x)))
        return np.abs((np.inner(w[:, 0], x) - np.inner(w[:, 1], x)) \
                    / (stepsize(cnt, self.l2_weight)*np.inner(x, x)))