from libact.base.interfaces import QueryStrategy
from libact.models import LogisticRegression
import numpy as np


class UncertaintySampling(QueryStrategy):

    def __init__(self, method='le'):
        """Currently only LogisticRegression is supported."""
        self.model = LogisticRegression()
        self.method = method

    def make_query(self, dataset):
        """
        Three choices for method (default 'le'):
        'lc' (Least Confident), 'sm' (Smallest Margin), 'le' (Label Entropy)
        """
        self.model.train(dataset)

        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())

        if self.method == 'lc':  # least confident
            # time complexity analysis:
            # self.model.predict_real(X_pool) -> O(NK)
            # np.max(..., axis=1) -> O(NK)
            # 1 - np.max(..., axis=1) -> O(NK)
            # np.argmax(...) -> O(N)
            # therefore, total time complexity is O(NK) + O(NK) + O(NK) + O(N) = O(NK)
            ask_id = np.argmax(1 - np.max(self.model.predict_real(X_pool), 1))

        elif self.method == 'sm':  # smallest margin
            # time complexity analysis:
            # O(NK) + O(N)
            prob = self.model.predict_real(X_pool)
            min_margin = np.inf
            for j in range(len(prob)) :
                m1_id = np.argmax(prob[j])
                m2_id = np.argmax(np.delete(prob[j], m1_id))
                margin = prob[j][m1_id] - prob[j][m2_id]
                if margin < min_margin :
                    min_margin = margin
                    ask_id = j

        elif self.method == 'le':  # default : label entropy (most commonly used)
            # XXX divide by zero?
            # time complexity analysis:
            # prob = self.model.predict_real(X_pool) -> O(NK)
            # np.log(prob) -> O(NK)
            # prob * np.log(prob) -> O(NK)
            # -np.sum(..., axis=1) -> O(NK)
            # np.argmax(...) -> O(N)
            # therefore, total time complexity = O(NK)
            prob = self.model.predict_real(X_pool)
            ask_id = np.argmax(-np.sum(prob * np.log(prob), 1))
            # ask_id = np.argmax(-np.sum(self.model.predict_real(X_pool)
            #     * np.log(self.model.predict_real(X_pool)), 1))

        else:
            raise ValueError(
                "Invalid method '%s' (available choices: ('lc', 'sm', 'le')"
                % self.method
                )

        return unlabeled_entry_ids[ask_id]

    def get_model(self):
        """Returns the model used by the last query"""
        return self.model
