"""Query by committee

This module contains a class that implements Query by committee active learning
algorithm.
"""
from __future__ import division

import logging
import math

import numpy as np

from libact.base.dataset import Dataset
from libact.base.interfaces import QueryStrategy
import libact.models
from libact.utils import inherit_docstring_from, seed_random_state, zip

LOGGER = logging.getLogger(__name__)


class QueryByCommittee(QueryStrategy):

    r"""Query by committee

    Parameters
    ----------
    models : list of :py:mod:`libact.models` instances or str
        This parameter accepts a list of initialized libact Model instances,
        or class names of libact Model classes to determine the models to be
        included in the committee to vote for each unlabeled instance.

    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

    Attributes
    ----------
    students : list, shape = (len(models))
        A list of the model instances used in this algorithm.

    random_states\_ : np.random.RandomState instance
        The random number generator using.

    Examples
    --------
    Here is an example of declaring a QueryByCommittee query_strategy object:

    .. code-block:: python

       from libact.query_strategies import QueryByCommittee
       from libact.models import LogisticRegression

       qs = QueryByCommittee(
                dataset, # Dataset object
                models=[
                    LogisticRegression(C=1.0),
                    LogisticRegression(C=0.1),
                ],
            )


    References
    ----------
    .. [1] Seung, H. Sebastian, Manfred Opper, and Haim Sompolinsky. "Query by
           committee." Proceedings of the fifth annual workshop on
           Computational learning theory. ACM, 1992.
    """

    def __init__(self, *args, **kwargs):
        super(QueryByCommittee, self).__init__(*args, **kwargs)

        models = kwargs.pop('models', None)
        if models is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'models'"
            )
        elif not models:
            raise ValueError("models list is empty")

        random_state = kwargs.pop('random_state', None)
        self.random_state_ = seed_random_state(random_state)

        self.students = list()
        for model in models:
            if isinstance(model, str):
                self.students.append(getattr(libact.models, model)())
            else:
                self.students.append(model)
        self.n_students = len(self.students)
        self.teach_students()

    def disagreement(self, votes):
        """
        Return the disagreement measurement of the given number of votes.
        It uses the vote entropy to measure the disagreement.

        Parameters
        ----------
        votes : list of int, shape==(n_samples, n_students)
            The predictions that each student gives to each sample.

        Returns
        -------
        disagreement : list of float, shape=(n_samples)
            The vote entropy of the given votes.
        """
        ret = []
        for candidate in votes:
            ret.append(0.0)
            lab_count = {}
            for lab in candidate:
                lab_count[lab] = lab_count.setdefault(lab, 0) + 1

            # Using vote entropy to measure disagreement
            for lab in lab_count.keys():
                ret[-1] -= lab_count[lab] / self.n_students * \
                    math.log(float(lab_count[lab]) / self.n_students)

        return ret

    def _labeled_uniform_sample(self, sample_size):
        """sample labeled entries uniformly"""
        labeled_entries = self.dataset.get_labeled_entries()
        samples = [labeled_entries[
            self.random_state_.randint(0, len(labeled_entries))
        ]for _ in range(sample_size)]
        return Dataset(*zip(*samples))

    def teach_students(self):
        """
        Train each model (student) with the labeled data using bootstrap
        aggregating (bagging).
        """
        dataset = self.dataset
        for student in self.students:
            bag = self._labeled_uniform_sample(int(dataset.len_labeled()))
            while bag.get_num_of_labels() != dataset.get_num_of_labels():
                bag = self._labeled_uniform_sample(int(dataset.len_labeled()))
                LOGGER.warning('There is student receiving only one label,'
                               're-sample the bag.')
            student.train(bag)

    @inherit_docstring_from(QueryStrategy)
    def update(self, entry_id, label):
        # Train each model with newly updated label.
        self.teach_students()

    @inherit_docstring_from(QueryStrategy)
    def make_query(self):
        dataset = self.dataset
        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())

        # Let the trained students vote for unlabeled data
        votes = np.zeros((len(X_pool), len(self.students)))
        for i, student in enumerate(self.students):
            votes[:, i] = student.predict(X_pool)

        id_disagreement = [(i, dis) for i, dis in
                           zip(unlabeled_entry_ids, self.disagreement(votes))]

        disagreement = sorted(id_disagreement, key=lambda id_dis: id_dis[1],
                              reverse=True)
        ask_id = self.random_state_.choice(
            [e[0] for e in disagreement if e[1] == disagreement[0][1]])

        return ask_id
