""" ===== Reference =====
[1] S.-J. Huang, R. Jin, and Z.-H. Zhou. Active learning by querying informative and representative examples.

"""

import bisect
import time
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from libact.base.interfaces import QueryStrategy


class QUIRE(QueryStrategy):

    def __init__(self, *args, **kwargs):
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
        self.K = K
        self.L = np.linalg.inv(K + self.lmbda * np.eye(len(X)))

    def update(self, entry_id, label):
        bisect.insort(a=self.Lindex, x=entry_id)
        self.Uindex.remove(entry_id)
        self.y[entry_id] = label

    def make_query(self):
        L = self.L
        K = self.K
        Lindex = self.Lindex
        len_Lindex = len(Lindex)
        Uindex = self.Uindex
        len_Uindex = len(Uindex)
        query_index = -1
        min_eva = np.inf
        y_labeled = np.array([label for label in self.y if label is not None])
        Laa = L[Uindex, :][:, Uindex]
        det_Laa = np.linalg.det(Laa)
        """efficient computation of inv(Laa)
        """
        M3 =  np.dot( self.K[Uindex, :][:, Lindex],  np.linalg.inv( self.lmbda * np.eye(len_Lindex) ) )
        M2 =  np.dot( M3, self.K[Lindex, :][:, Uindex] )
        M1 = self.lmbda * np.eye(len_Uindex) + self.K[Uindex, :][:, Uindex]
        inv_Laa = M1 - M2
        iList = list( range(len_Uindex) )
        for i, each_index in enumerate(Uindex):
            """go through all unlabeled instances and compute their evaluation
            values one by one
            """
            Lss = L[each_index][each_index]
            Lsl = L[each_index][Lindex]
            Uindex_r = Uindex[:]
            Uindex_r.remove(each_index)
            Lsu = L[each_index][Uindex_r]
            Lul = L[Uindex_r, :][:, Lindex]
            Luu = L[Uindex_r, :][:, Uindex_r]
            """efficient computation of inv(Luu)
            """
            iList_r = iList[:]
            iList_r.remove(i)
            a = inv_Laa[i, i]
            b = -inv_Laa[iList_r, i]
            D = inv_Laa[iList_r, :][:, iList_r]
            inv_Luu = D - 1/a * np.dot( b,  b.T )
            tmp = np.dot(
                Lsl - np.dot(np.dot(Lsu, inv_Luu), Lul),
                y_labeled,
                )
            """
            tmp = np.dot(
                Lsl - np.dot(np.dot(Lsu, np.linalg.inv(Luu)), Lul),
                y_labeled,
                )
            """
            eva = Lss - det_Laa / Lss + 2 * np.abs(tmp)

            if eva < min_eva:
                query_index = each_index
                min_eva = eva
        return query_index
