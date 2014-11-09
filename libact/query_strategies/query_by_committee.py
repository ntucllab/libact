from libact.base.interfaces import QueryStrategy
import numpy as np
from functools import cmp_to_key
import math


class QueryByCommittee(QueryStrategy):

    def __init__(self, models):
        """
        model: list trained libact Model object for prediction
               Currently only LogisticRegression is supported.
        """
        self.students = models
        self.n_students = len(self.students)

    def disagreement(self, votes):
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

    def make_query(self, dataset, n_queries=1):
        unlabeled_entry_ids = dataset.get_unlabeled()
        X_pool = [dataset[i][0] for i in unlabeled_entry_ids]
        votes = []

        # Training models with labeled data using bootstrap aggregating
        # (bagging)
        for student in self.students:
            student.fit(dataset.labeled_uniform_sample(int(dataset.len_labeled()/2), 100))

        # Let the trained students vote for unlabeled data
        for X in X_pool:
            vote = []
            for student in self.students:
                vote.append(student.predict(X)[0])
            votes.append(vote)

        id_disagreement = [(i, dis) for i, dis in
                enumerate(self.disagreement(votes))]

        disagreement = sorted(id_disagreement, key=lambda id_dis: id_dis[1],
                reverse=True)
        ret = [i[0] for i in disagreement[:n_queries]]

        return ret
