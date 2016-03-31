"""Query by committee

This module contains a class that implements Query by committee active learning
algorithm.
"""
from functools import cmp_to_key
import logging
import math

import numpy as np

from libact.base.interfaces import QueryStrategy
import libact.models

logger = logging.getLogger(__name__)


class QueryByCommittee(QueryStrategy):
    """Query by committee

    Parameters
    ----------
    models : list of :py:mod:`libact.models` instances or str
        This parameter accepts a list of initialized libact Model instances,
        or class names of libact Model classes to determine the models to be
        included in the committee to vote for each unlabeled instance.


    Attributes
    ----------
    students : list, shape = (len(models))
        A list of the model instances used in this algorithm.


    Examples
    --------
    Here is an example of declaring a QueryByCommittee query_strategy object:

    .. code-block:: python

       from libact.query_strategies import QueryByCommittee
       from libact.models import LogisticRegression

       qs = QueryStrategy(
                dataset, # Dataset object
                models=[
                    LogisticRegression(C=1.0),
                    LogisticRegression(C=0.1),
                ],
            )


    References
    ----------
    .. [1] Seung, H. Sebastian, Manfred Opper, and Haim Sompolinsky. "Query by
           committee." Proceedings of the fifth annual workshop on Computational
           learning theory. ACM, 1992.
    """

    def __init__(self, *args, **kwargs):
        super(QueryByCommittee, self).__init__(*args, **kwargs)
        self.students = list()
        models = kwargs.pop('models', None)
        if models is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'models'"
                )
        elif not models:
            raise ValueError("models list is empty")
        for model in models:
            if type(model) is str:
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
                ret[-1] -= lab_count[lab]/self.n_students * \
                            math.log(float(lab_count[lab])/self.n_students)

        return ret

    def teach_students(self):
        """
        Train each model (student) with the labeled data using bootstrap
        aggregating (bagging).
        """
        dataset = self.dataset
        for student in self.students:
            bag = dataset.labeled_uniform_sample(int(dataset.len_labeled()))
            while bag.get_num_of_labels() != dataset.get_num_of_labels():
                bag = dataset.labeled_uniform_sample(int(dataset.len_labeled()))
                logger.warning('There is student receiving only one label,'
                               're-sample the bag.')
            student.train(bag)

    def update(self, entry_id, label):
        self.teach_students()

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
        ask_id = np.random.choice(
            [e[0] for e in disagreement if e[1] == disagreement[0][1] ])

        return ask_id
