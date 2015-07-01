from libact.base.interfaces import QueryStrategy
import libact.models
import numpy as np
from functools import cmp_to_key
import math
import svmutil
import ctypes

print(svmutil.__file__)

class HintSVM(QueryStrategy):

    def __init__(self, *args, **kwargs):
        """
        model: a list of initialized libact Model instances, or class names of
               libact Model classes for prediction.
        """
        super(HintSVM, self).__init__(*args, **kwargs)
        self.cl = 0.1 # Weight on labeled data's classification error
        self.ch = 0.1 # Weight on hinted data's classification error
        self.p = 0.5  # Prabability of sampling a data from unlabeled pool to
                      # hinted pool

    def update(self, entry_id, label):
        # TODO
        pass

    def make_query(self):
        dataset = self.dataset
        unlabeled_entry_ids, unlabeled_pool = zip(*dataset.get_unlabeled_entries())
        labeled_pool, y = zip(*dataset.get_labeled_entries())

        cl = self.cl
        ch = self.ch
        p = self.p
        hint_pool_idx = np.random.choice(len(unlabeled_pool), int(len(unlabeled_pool)*p))
        hint_pool = np.array(unlabeled_pool)[hint_pool_idx]

        weight = [1.0 for i in range(len(labeled_pool))] +\
                 [(ch/cl) for i in range(len(hint_pool))]
        y = list(y) + [0 for i in range(len(hint_pool))]
        X = [x.tolist() for x in labeled_pool] +\
                [x.tolist() for x in hint_pool]

        prob  = svmutil.svm_problem(weight, y, X)
        param = svmutil.svm_parameter('-s 5 -t 0 -b 0 -c %f -q' % cl)
        m = svmutil.svm_train(prob, param)

        #TODO need only p_val
        y = np.zeros((len(unlabeled_pool), ))
        p_label, p_acc, p_val = svmutil.svm_predict(y, [x.tolist()\
                for x in unlabeled_pool], m)

        #TODO not sure to have abs or not
        p_val = [abs(val[0]) for val in p_val]
        idx = np.argmax(p_val)
        return unlabeled_entry_ids[idx]
