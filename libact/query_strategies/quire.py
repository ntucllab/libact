""" ===== Reference =====
[1] S.-J. Huang, R. Jin, and Z.-H. Zhou. Active learning by querying informative and representative examples.

"""

import bisect

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from libact.base.interfaces import QueryStrategy


class QUIRE(QueryStrategy):

    def __init__(self, *args, **kwargs):
        kwargs['update_callback'] = True
        super(QUIRE, self).__init__(*args, **kwargs)
        self.Uindex = [
            idx for idx, feature in self.dataset.get_unlabeled_entries()
            ]
        self.Lindex = [
            idx for idx in range(len(self.dataset)) if idx not in self.Uindex
            ]
        self.lmbda = kwargs.pop('lmbda', 1)
        self.gamma = kwargs.pop('gamma', 1)
        X, self.y = zip(*self.dataset.get_entries())
        self.y = list(self.y)
        K = rbf_kernel(X=X, Y=X, gamma=self.gamma)
        self.L = np.linalg.inv(K + self.lmbda * np.eye(len(X)))

    def update(self, entry_id, label):
        bisect.insort(a=self.Lindex, x=entry_id)
        self.Uindex.remove(entry_id)
        self.y[entry_id] = label

    def make_query(self):
        L = self.L
        Lindex = self.Lindex
        Uindex = self.Uindex
        query_index = -1
        min_eva = np.inf
        y_labeled = np.array([label for label in self.y if label is not None])
        Laa = (((L[Uindex]).T)[Uindex]).T
        det_Laa = np.linalg.det(Laa)
        for each_index in Uindex:
            """go through all unlabeled instances and compute their evaluation
            values one by one
            """
            Lss = L[each_index][each_index]
            Lsl = L[each_index][Lindex]
            Uindex_r = Uindex[:]
            Uindex_r.remove(each_index)
            Lsu = L[each_index][Uindex_r]
            Lul = (((L[Uindex_r]).T)[Lindex]).T
            Luu = (((L[Uindex_r]).T)[Uindex_r]).T

            tmp = np.dot(
                Lsl - np.dot(np.dot(Lsu, np.linalg.inv(Luu)), Lul),
                y_labeled,
                )
            eva = Lss - det_Laa / Lss + 2 * np.abs(tmp)

            if eva < min_eva:
                query_index = each_index
                min_eva = eva

        return query_index
