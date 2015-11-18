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

    method: {'lc', 'sm', 'le'}, optional (default='le')
        lc stands for least confidence, it queries the instance whose posterior
        probability of being positive is nearest 0.5 (for binary classification);
        sm stands for smallest margin, it queries the instance whose posterior
        probability gap between the most and the second probable labels is minimal;
        le stands for the common entropy approach.

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
        self.method = kwargs.pop('method', 'le')

    def make_query(self):
        """
        Three choices for method (default 'le'):
        'lc' (Least Confident), 'sm' (Smallest Margin), 'le' (Label Entropy)
        """
        dataset = self.dataset
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
            for j in range(len(prob)):
                m1_id = np.argmax(prob[j])
                m2_id = np.argmax(np.delete(prob[j], m1_id))
                margin = prob[j][m1_id] - prob[j][m2_id]
                if margin < min_margin:
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
