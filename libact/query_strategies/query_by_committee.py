from libact.base.interfaces import QueryStrategy
import libact.models
import numpy as np
from functools import cmp_to_key
import math


class QueryByCommittee(QueryStrategy):

    def __init__(self, models):
        """
        model: a list of initialized libact Model instances, or class names of
               libact Model classes for prediction.
        """
        self.students = list()
        for model in models:
            if type(model) is str:
                self.students.append(getattr(libact.models, model)())
            else:
                self.students.append(model)
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

    def make_query(self, dataset):
        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())
        votes = []

        # Training models with labeled data using bootstrap aggregating
        # (bagging)
        # TODO exception on only one label is sampled.
        for student in self.students:
            bag = dataset.labeled_uniform_sample(int(dataset.len_labeled()))
            while bag.get_num_of_labels() != dataset.get_num_of_labels():
                bag = dataset.labeled_uniform_sample(int(dataset.len_labeled()))
            student.train(bag)

        # Let the trained students vote for unlabeled data
        for X in X_pool:
            vote = []
            for student in self.students:
                vote.append(student.predict(X)[0])
            votes.append(vote)

        id_disagreement = [(i, dis) for i, dis in
                zip(unlabeled_entry_ids, self.disagreement(votes))]

        disagreement = sorted(id_disagreement, key=lambda id_dis: id_dis[1],
                reverse=True)
        ret = disagreement[0][0]

        return ret
