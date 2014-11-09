from libact.base.interfaces import QueryStrategy
from libact.models import LogisticRegression
import numpy as np


class UncertaintySampling(QueryStrategy):

    def __init__(self):
        """Currently only LogisticRegression is supported."""
        self.model = LogisticRegression()

    def make_query(self, dataset, n_queries=1, method='le'):
        """
        Three choices for method (default 'le'):
        'lc' (Least Confident), 'sm' (Smallest Margin), 'le' (Label Entropy)
        """
        self.model.fit(dataset)

        unlabeled_entry_ids = dataset.get_unlabeled()
        X_pool = [dataset[i][0] for i in unlabeled_entry_ids]

        if method == 'lc':  # least confident
            ask_id = np.argmax(1 - np.max(self.model.predict_proba(X_pool), 1))

        elif method == 'sm':  # smallest margin
            prob = self.model.predict_proba(X_pool)
            min_margin = np.inf
            for j in range(len(prob)) :
                m1_id = np.argmax(prob[j])
                m2_id = np.argmax(np.delete(prob[j], m1_id))
                margin = prob[j][m1_id] - prob[j][m2_id]
                if margin < min_margin :
                    min_margin = margin
                    ask_id = j

        elif method == 'le':  # default : label entropy (most commonly used)
            ask_id = np.argmax(-np.sum(self.model.predict_proba(X_pool)
                * self.model.predict_log_proba(X_pool), 1))

        return unlabeled_entry_ids[ask_id]

    def get_model(self):
        """Returns the model used by the last query"""
        return self.model
