"""Uncertainty Sampling


"""
from libact.base.interfaces import QueryStrategy
import numpy as np


class UncertaintySampling(QueryStrategy):
    """Uncertainty Sampling

    This class implements Uncertainty Sampling active learning algorithm [1]_.

    Parameters
    ----------
    model: libact.model.* object instance
        The base model used for trainin, this model should support predict_real.

    method: {'lc', 'sm'}, optional (default='lc')
        least confidence (lc), it queries the instance whose posterior
        probability of being positive is nearest 0.5 (for binary classification);
        smallest margin (sm), it queries the instance whose posterior
        probability gap between the most and the second probable labels is minimal;

    Attributes
    ----------


    References
    ----------

    .. [1] Settles, Burr. "Active learning literature survey." University of
           Wisconsin, Madison 52.55-66 (2010): 11.
    """

    def __init__(self, *args, **kwargs):
        """Currently only LogisticRegression is supported."""
        super(UncertaintySampling, self).__init__(*args, **kwargs)

        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
                )
        self.model.train(self.dataset)

        self.method = kwargs.pop('method', 'lc')
        if self.method not in ['lc', 'sm']:
            raise TypeError(
                "supported methods are ['lc', 'sm'], the given one is: " + \
                self.method
                )

    def update(self, entry_id, label):
        self.model.train(self.dataset)

    def make_query(self):
        """
        Choices for method (default 'lc'):
        'lc' (Least Confident), 'sm' (Smallest Margin)
        """
        unlabeled_entry_ids, X_pool = zip(*self.dataset.get_unlabeled_entries())

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
            for j in range(len(prob)):
                m1_id = np.argmax(prob[j])
                m2_id = np.argmax(np.delete(prob[j], m1_id))
                margin = prob[j][m1_id] - prob[j][m2_id]
                if margin < min_margin:
                    min_margin = margin
                    ask_id = j

        else:
            raise ValueError(
                "Invalid method '%s' (available choices: ('lc', 'sm', 'le')"
                % self.method
                )

        return unlabeled_entry_ids[ask_id]

    def get_model(self):
        """Returns the model used by the last query"""
        return self.model
