"""Hinted Support Vector Machine

This module contains a class that implements Hinted Support Vector Machine, an
active learning algorithm.

To use this module, it is required to install the following package:

https://github.com/yangarbiter/hintsvm
"""

import hintsvmutil  # from hinsvm package
import numpy as np

from libact.base.interfaces import QueryStrategy


class HintSVM(QueryStrategy):

    """Hinted Support Vector Machine

    Hinted Support Vector Machine is an active learning algorithm within the
    hined sampling framework with an extended support vector machine.

    Parameters
    ----------
    Cl : float, >0
        The weight of the classification error on labeled pool.

    Ch : float, >0
        The weight of the hint error on hint pool.

    p : float, >0 and <=1
        The probability to select an instance from unlabeld pool to hint pool.

    Attributes
    ----------


    References
    ----------
    Li, Chun-Liang, Chun-Sung Ferng, and Hsuan-Tien Lin. "Active Learning with
    Hinted Support Vector Machine." ACML. 2012.

    Chun-Liang Li, Chun-Sung Ferng, and Hsuan-Tien Lin. Active learning using
    hint information. Neural Computation, 27(8):1738--1765, August 2015.
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

    def update(self, entry_id, label):
        pass

    def make_query(self):
        dataset = self.dataset
        unlabeled_entry_ids, unlabeled_pool = zip(
            *dataset.get_unlabeled_entries())
        labeled_pool, y = zip(*dataset.get_labeled_entries())

        cl = self.cl
        ch = self.ch
        p = self.p
        hint_pool_idx = np.random.choice(
            len(unlabeled_pool), int(
                len(unlabeled_pool)*p))
        hint_pool = np.array(unlabeled_pool)[hint_pool_idx]

        weight = [1.0 for _ in range(len(labeled_pool))] +\
                 [(ch/cl) for i in range(len(hint_pool))]
        y = list(y) + [0 for i in range(len(hint_pool))]
        X = [x.tolist() for x in labeled_pool] +\
            [x.tolist() for x in hint_pool]

        prob = hintsvmutil.svm_problem(weight, y, X)
        param = hintsvmutil.svm_parameter('-s 5 -t 0 -b 0 -c %f -q' % cl)
        m = hintsvmutil.svm_train(prob, param)

        # TODO need only p_val
        y = np.zeros((len(unlabeled_pool), ))
        p_label, p_acc, p_val = hintsvmutil.svm_predict(
            y, [x.tolist() for x in unlabeled_pool], m)

        p_val = [abs(val[0]) for val in p_val]
        idx = np.argmax(p_val)
        return unlabeled_entry_ids[idx]
