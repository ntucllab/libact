from libact.base.interfaces import QueryStrategy
from libact.base.dataset import Dataset
import copy
import numpy as np
import time
from joblib import Parallel, delayed

class VarianceReduction(QueryStrategy):

    def __init__(self, model, sigma=100000000.0, optimality='trace'):
        """
        model: a list of initialized libact Model object for prediction.
        optimality: string, 'trace', 'determinant' or 'eigenvalue'
        (default='trace')
        """
        self.model = model
        self.optimality = optimality
        self.sigma = sigma

    def A(self, pi, c, x, label_count, feature_count):
        """
        import time
        s = time.time()
        grad = pi[c] *  np.ones((feature_count*label_count))
        for i in range(feature_count):
            for j in range(label_count):
                if c == j:
                    grad[i*label_count+j] *= (1-pi[c]) * x[i]
                    #grad.append(pi[c] * (1-pi[c]) * x[i])
                else:
                    grad[i*label_count+j] *= (-pi[j]) * x[i]
                    #grad.append(-pi[c] * pi[j] * x[i])
        print(s-time.time())
        print(np.shape(grad))
        """
        _pi = -1 * np.array(pi)
        _pi[c] += 1
        grad = pi[c] * np.tile(np.array([x]).T, (1, label_count)) *\
                        np.tile(np.array([_pi]), (feature_count, 1))
        grad = grad.reshape((feature_count*label_count))

        return np.dot(grad.T, grad)

    def Fisher(self, pi, x, label_count, feature_count):
        sigma = self.sigma
        """
        import time
        s = time.time()
        fisher = np.ones((label_count*feature_count, label_count*feature_count))
        for l1 in range(label_count):
            for f1 in range(feature_count):
                col = l1 * feature_count + f1
                #fisher.append([])
                for l2 in range(label_count):
                    for f2 in range(feature_count):
                        if l1 == l2 and f1 == f2:
                            fisher[col][l2*feature_count+f2] =\
                                x[f1] * x[f2] * pi[l1] * (1-pi[l1]) + 1/sigma
                            #fisher[col].append(
                            #    x[f1] * x[f2] * pi[l1] * (1-pi[l2])
                            #)
                        elif l1 == l2 and f1 != f2:
                            fisher[col][l2*feature_count+f2] =\
                                x[f1] * x[f2] * pi[l1] * (1-pi[l1])
                            #fisher[col].append(
                            #    x[f1] * x[f2] * pi[l1] * (1-pi[l2])
                            #)
                        else:
                            fisher[col][l2*feature_count+f2] =\
                                x[f1] * x[f2] * pi[l1] * pi[l2]
                            #fisher[col].append(
                            #    x[f1] * x[f2] * pi[l1] * pi[l2]
                            #)
        #print(np.shape(np.array(fisher)))
        """
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
        fisher += (1/sigma) * np.eye(feature_count*label_count)
        return np.linalg.pinv(np.array(fisher))

    def Phi(self, pi, x, label_count, feature_count):
        ret = 0.0
        for i in range(label_count):
            A = self.A(pi, i, x, label_count, feature_count)
            F = self.Fisher(pi, x, label_count, feature_count)
            ret += np.trace( A*F )
        return ret


    def E(self, X, y, qx, clf, label_count):
        query_point = clf.predict_proba([qx])
        feature_count = len(X[0])
        ret = 0.0
        for i in range(label_count):
            clf = copy.copy(self.model)
            clf.train(Dataset(X+[qx], y+[i]))
            pi = clf.predict_proba([qx])
            ret += query_point[-1][i] * self.Phi(pi[-1], qx, label_count,
                    feature_count)
        return ret

    def make_query(self, dataset, n_queries=1):
        labeled_entry_ids = dataset.get_labeled()
        unlabeled_entry_ids = dataset.get_unlabeled()
        Xlabeled = np.array([dataset[i][0] for i in labeled_entry_ids])
        y = [dataset[i][1] for i in labeled_entry_ids]
        X_pool = [dataset[i][0] for i in unlabeled_entry_ids]
        label_count = dataset.get_num_of_labels()

        clf = copy.copy(self.model)
        clf.train(Dataset(Xlabeled, y))

        s = time.time()
        errors = Parallel(n_jobs=20)(delayed(self.E)(Xlabeled, y, x, clf,
            label_count) for x in X_pool)
        print(time.time()-s)
        #errors = []
        #for x in X_pool:
        #    errors.append(self.E(Xlabeled, y, x, clf, label_count))
        #    print(errors[-1])
        #print(errors.index(min(errors)))

        return [unlabeled_entry_ids[errors.index(min(errors))]]
