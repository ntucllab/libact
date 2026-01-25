"""BALD (Bayesian Active Learning by Disagreement)

This module implements BALD using an ensemble of models to approximate
Bayesian uncertainty estimation via mutual information.
"""
import logging

import numpy as np

from libact.base.dataset import Dataset
from libact.base.interfaces import QueryStrategy, ProbabilisticModel
from libact.utils import inherit_docstring_from, seed_random_state

LOGGER = logging.getLogger(__name__)


class BALD(QueryStrategy):
    """BALD (Bayesian Active Learning by Disagreement) Query Strategy

    This strategy implements Bayesian Active Learning by Disagreement (BALD)
    using an ensemble of models. BALD selects samples that maximize mutual
    information between predictions and model parameters, approximated here
    using ensemble disagreement.

    BALD score: I[y; w | x, D] = H[y | x, D] - E_w[H[y | x, w]]
                = H[mean(proba)] - mean(H[proba])

    Where H is entropy, computed as -sum(p * log(p)).

    Parameters
    ----------
    dataset : Dataset object
        The dataset to query from.

    models : list of ProbabilisticModel instances, optional
        Pre-initialized models to use as the ensemble. Each model must
        implement predict_proba(). If provided, these models are used directly
        (e.g., models with different hyperparameters).

    base_model : ProbabilisticModel instance, optional
        A base model to clone for creating the ensemble via bootstrap bagging.
        Required if `models` is not provided. Must have a `clone()` method.

    n_models : int, optional (default=10)
        Number of models to create when using `base_model` with bagging.
        Ignored if `models` is provided.

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        Random state for reproducibility.

    Attributes
    ----------
    models : list of ProbabilisticModel
        The ensemble of models.

    random_state_ : np.random.RandomState instance
        The random number generator.

    Examples
    --------
    Using pre-initialized models with different hyperparameters:

    .. code-block:: python

       from libact.query_strategies import BALD
       from libact.models import LogisticRegression

       qs = BALD(
           dataset,
           models=[
               LogisticRegression(C=0.1),
               LogisticRegression(C=1.0),
               LogisticRegression(C=10.0),
           ]
       )

    Using bootstrap bagging with a base model:

    .. code-block:: python

       from libact.query_strategies import BALD
       from libact.models import SklearnProbaAdapter
       from sklearn.ensemble import RandomForestClassifier

       base = SklearnProbaAdapter(RandomForestClassifier(n_estimators=10))
       qs = BALD(dataset, base_model=base, n_models=10)

    References
    ----------
    .. [1] Houlsby, Neil, et al. "Bayesian active learning for classification
           and preference learning." arXiv preprint arXiv:1112.5745 (2011).

    .. [2] Gal, Yarin, Riashat Islam, and Zoubin Ghahramani. "Deep bayesian
           active learning with image data." ICML 2017.
    """

    def __init__(self, dataset, **kwargs):
        super(BALD, self).__init__(dataset, **kwargs)

        models = kwargs.pop('models', None)
        base_model = kwargs.pop('base_model', None)
        self.n_models = kwargs.pop('n_models', 10)

        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)

        # Initialize ensemble
        if models is not None:
            # Use provided models directly
            if not models:
                raise ValueError("models list is empty")
            for model in models:
                if not isinstance(model, ProbabilisticModel):
                    raise TypeError(
                        "All models must be ProbabilisticModel instances"
                    )
            self.models = list(models)
            self._base_model = None
        elif base_model is not None:
            # Create ensemble via cloning
            if not isinstance(base_model, ProbabilisticModel):
                raise TypeError("base_model must be a ProbabilisticModel")
            if not hasattr(base_model, 'clone'):
                raise TypeError("base_model must have a 'clone()' method")
            self._base_model = base_model
            self.models = [base_model.clone() for _ in range(self.n_models)]
        else:
            raise TypeError(
                "__init__() requires either 'models' or 'base_model' argument"
            )

        # Train the ensemble
        self._train_ensemble()

    def _entropy(self, proba):
        """Calculate entropy of probability distributions.

        Parameters
        ----------
        proba : array-like, shape (n_samples, n_classes)
            Probability distributions.

        Returns
        -------
        entropy : ndarray, shape (n_samples,)
            Entropy for each sample.
        """
        # Clip to avoid log(0)
        proba = np.clip(proba, 1e-10, 1.0)
        return -np.sum(proba * np.log(proba), axis=1)

    def _labeled_uniform_sample(self, sample_size):
        """Sample labeled entries uniformly for bootstrap bagging."""
        X, y = self.dataset.get_labeled_entries()
        indices = self.random_state_.randint(0, len(X), size=sample_size)
        return Dataset(X[indices], np.array(y)[indices])

    def _train_ensemble(self):
        """Train the ensemble using bootstrap bagging."""
        dataset = self.dataset
        n_labeled = dataset.len_labeled()

        if n_labeled == 0:
            LOGGER.warning("No labeled samples available for training")
            return

        for model in self.models:
            # Create bootstrap sample
            bag = self._labeled_uniform_sample(int(n_labeled))
            # Ensure all classes are represented
            max_attempts = 10
            attempts = 0
            while bag.get_num_of_labels() != dataset.get_num_of_labels():
                bag = self._labeled_uniform_sample(int(n_labeled))
                attempts += 1
                if attempts >= max_attempts:
                    LOGGER.warning(
                        "Could not create balanced bootstrap sample after "
                        f"{max_attempts} attempts, using current bag"
                    )
                    break
            model.train(bag)

    @inherit_docstring_from(QueryStrategy)
    def update(self, entry_id, label):
        # Retrain ensemble with the new labeled data
        self._train_ensemble()

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        X_pool = np.asarray(X_pool)

        if len(unlabeled_entry_ids) == 0:
            raise ValueError("No unlabeled samples available")

        # Get predictions from all models
        all_proba = []
        for model in self.models:
            proba = model.predict_proba(X_pool)
            all_proba.append(np.asarray(proba))

        all_proba = np.array(all_proba)  # shape: (n_models, n_samples, n_classes)

        # Calculate BALD score: H[mean(P)] - mean(H[P])
        # Mean probability across ensemble
        mean_proba = np.mean(all_proba, axis=0)  # shape: (n_samples, n_classes)

        # Entropy of mean predictions (total uncertainty)
        entropy_mean = self._entropy(mean_proba)  # shape: (n_samples,)

        # Mean entropy across models (expected data uncertainty)
        entropies = np.array([self._entropy(p) for p in all_proba])  # shape: (n_models, n_samples)
        mean_entropy = np.mean(entropies, axis=0)  # shape: (n_samples,)

        # BALD score = mutual information
        bald_scores = entropy_mean - mean_entropy  # shape: (n_samples,)

        # Select sample with highest BALD score (break ties randomly)
        max_score = np.max(bald_scores)
        candidates = np.where(np.isclose(bald_scores, max_score))[0]
        selected_idx = self.random_state_.choice(candidates)

        return unlabeled_entry_ids[selected_idx]

    def _get_scores(self):
        """Return BALD scores for all unlabeled samples."""
        dataset = self.dataset
        unlabeled_entry_ids, X_pool = dataset.get_unlabeled_entries()
        X_pool = np.asarray(X_pool)

        if len(unlabeled_entry_ids) == 0:
            return []

        # Get predictions from all models
        all_proba = np.array([
            np.asarray(model.predict_proba(X_pool))
            for model in self.models
        ])

        mean_proba = np.mean(all_proba, axis=0)
        entropy_mean = self._entropy(mean_proba)
        entropies = np.array([self._entropy(p) for p in all_proba])
        mean_entropy = np.mean(entropies, axis=0)
        bald_scores = entropy_mean - mean_entropy

        return list(zip(unlabeled_entry_ids, bald_scores))
