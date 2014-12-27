import numpy as np
from sklearn.linear_model import LogisticRegression

class ExpectedErrorReduction(QueryStrategy):

    def __init__(self):
        pass

    def make_query(self, dataset):
        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())
        X_own, y_own = zip(*dataset.get_labeled_entries())
        model = LogisticRegression()
        model.train(X_own, y_own)

        prob = model.predict_real(X_pool)
        ask_id, min_E = -1, np.inf
        model_t = LogisticRegression()
        for u in range(len(X_pool)):
            cur_E = 0
            for j in range(len(prob[u])):
                model_t.train(np.append(X_own, X_pool[u].reshape(1, -1), axis = 0), np.append(y_own, prob[u][j]))
                cur_E = cur_E + prob[u][j] * sum(1 - np.max(model_t.predict_real(np.delete(X_pool, u, 0)), 1))
            if cur_E < min_E :
                min_E = cur_E
                ask_id = u

        return ask_id
