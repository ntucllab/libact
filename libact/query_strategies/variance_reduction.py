from libact.base.interfaces import QueryStrategy
from libact.base.dataset import Dataset
import libact.models
import copy
import numpy as np
from joblib import Parallel, delayed

class VarianceReduction(QueryStrategy):

    def __init__(self, model, sigma=100000000.0, optimality='trace'):
        """
        model: a list of initialized libact Model object for prediction.
        optimality: string, 'trace', 'determinant' or 'eigenvalue'
        (default='trace')
        """
        if type(model) is str:
            self.model = getattr(libact.models, model)()
        else:
            self.model = model
        self.optimality = optimality
        self.sigma = sigma

    def A(self, pi, c, x, label_count, feature_count):
        _pi = -1 * np.array(pi)
        _pi[c] += 1
        grad = pi[c] * np.tile(np.array([x]).T, (1, label_count)) *\
                        np.tile(np.array([_pi]), (feature_count, 1))
        grad = grad.reshape((feature_count*label_count))

        return np.dot(grad.T, grad)

    def Fisher(self, pi, x, label_count, feature_count):
        sigma = self.sigma
        _pi_l2 = np.ones((1, feature_count*label_count))
        for i in range(label_count):
            _pi_l2[0, i*feature_count:(i+1)*feature_count] = pi[i]

        _pi = np.tile(_pi_l2, (label_count*feature_count, 1))
        for i in range(label_count):
            _pi[i*feature_count:(i+1)*feature_count,
                i*feature_count:(i+1)*feature_count] = 1 - pi[i] 
        fisher =\
            np.tile(np.array([x]).T, (label_count, label_count*feature_count)) *\
            np.tile(np.array([x]), (label_count*feature_count, label_count)) *\
            np.tile(_pi_l2.T, (1, label_count*feature_count)) *\
            _pi
        fisher += (1.0/sigma) * np.eye(feature_count*label_count)
        return np.linalg.pinv(np.array(fisher))

    def Phi(self, pi, x, label_count, feature_count):
        ret = 0.0
        for i in range(label_count):
            A = self.A(pi, i, x, label_count, feature_count)
            F = self.Fisher(pi, x, label_count, feature_count)
            ret += np.trace( np.dot(A, F) )
        return ret


    def E(self, X, y, qx, clf, label_count):
        query_point = clf.predict_real([qx])
        feature_count = len(X[0])
        ret = 0.0
        for i in range(label_count):
            clf = copy.copy(self.model)
            clf.train(Dataset(X+[qx], y+[i]))
            pi = clf.predict_real([qx])
            ret += query_point[-1][i] * self.Phi(pi[-1], qx, label_count,
                    feature_count)
        return ret

    def make_query(self, dataset, n_queries=1):
        labeled_entry_ids = range(len(dataset.labeled))
        unlabeled_entries = dataset.get_unlabeled_entries()
        unlabeled_entry_ids = [i[0] for i in unlabeled_entries]
        Xlabeled = np.array([dataset.labeled[i][0] for i in labeled_entry_ids])
        y = [dataset.labeled[i][1] for i in labeled_entry_ids]
        X_pool = [dataset.unlabeled[i][0] for i in unlabeled_entry_ids]
        label_count = dataset.get_num_of_labels()

        clf = copy.copy(self.model)
        clf.train(Dataset(Xlabeled, y))

        errors = Parallel(n_jobs=20)(delayed(self.E)(Xlabeled, y, x, clf,
            label_count) for x in X_pool)

        return unlabeled_entry_ids[errors.index(min(errors))]
