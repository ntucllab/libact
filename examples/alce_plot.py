#!/usr/bin/env python3
"""
Cost-Senstive Multi-Class Active Learning
"""

import copy
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn.datasets
from sklearn.svm import SVR

# libact classes
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import SVM, LogisticRegression
from libact.query_strategies.multiclass import ActiveLearningWithCostEmbedding as ALCE
from libact.query_strategies import UncertaintySampling, RandomSampling
from libact.labelers import IdealLabeler
from libact.utils import calc_cost


def run(trn_ds, tst_ds, lbr, model, qs, quota, cost_matrix):
    C_in, C_out = [], []

    for _ in range(quota):
        # Standard usage of libact objects
        ask_id = qs.make_query()
        X, _ = zip(*trn_ds.data)
        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lb)

        model.train(trn_ds)
        trn_X, trn_y = zip(*trn_ds.get_labeled_entries())
        tst_X, tst_y = zip(*tst_ds.get_labeled_entries())
        C_in = np.append(C_in, calc_cost(trn_y, model.predict(trn_X), cost_matrix))
        C_out = np.append(C_out, calc_cost(tst_y, model.predict(tst_X), cost_matrix))

    return C_in, C_out


def split_train_test(test_size, n_labeled):
    data = sklearn.datasets.fetch_mldata('segment')
    X = data['data']
    target = np.unique(data['target'])
    # mapping the targets to 0 to n_classes-1
    y = np.array([np.where(target == i)[0][0] for i in data['target']])

    sss = StratifiedShuffleSplit(1, test_size=test_size, random_state=1126)
    for trn_idx, tst_idx in sss.split(X, y):
        X_trn, X_tst = X[trn_idx], X[tst_idx]
        y_trn, y_tst = y[trn_idx], y[tst_idx]
    trn_ds = Dataset(X_trn, np.concatenate(
        [y_trn[:n_labeled], [None] * (len(y_trn) - n_labeled)]))
    tst_ds = Dataset(X_tst, y_tst)
    fully_labeled_trn_ds = Dataset(X_trn, y_trn)

    count = np.bincount(y)
    cost_matrix = np.zeros((len(target), len(target)), dtype='float')
    random_state = np.random.RandomState(1126)
    for i in range(len(target)):
        for j in range(len(target)):
            if i == j:
                cost_matrix[i][j] = 0.
            else:
                cost_matrix[i][j] = 2000 * random_state.rand() * float(count[i]) / count[j]

    return trn_ds, tst_ds, y_trn, fully_labeled_trn_ds, cost_matrix


def main():
    # Specifiy the parameters here:
    test_size = 0.33    # the percentage of samples in the dataset that will be
    # randomly selected and assigned to the test set
    n_labeled = 10      # number of samples that are initially labeled

    # Load dataset
    trn_ds, tst_ds, y_train, fully_labeled_trn_ds, cost_matrix = \
        split_train_test(test_size, n_labeled)
    trn_ds2 = copy.deepcopy(trn_ds)
    trn_ds3 = copy.deepcopy(trn_ds)
    lbr = IdealLabeler(fully_labeled_trn_ds)
    n_classes = len(np.unique(y_train)) # = 7

    #cost_matrix = np.random.RandomState(1126).rand(n_classes, n_classes)

    quota = 300    # number of samples to query

    # Comparing UncertaintySampling strategy with RandomSampling.
    # model is the base learner, e.g. LogisticRegression, SVM ... etc.
    qs = UncertaintySampling(trn_ds, method='lc',
                             model=SVM(decision_function_shape='ovr'))
    model = SVM(decision_function_shape='ovr')
    E_in_1, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs, quota, cost_matrix)

    qs2 = RandomSampling(trn_ds2)
    model = SVM(decision_function_shape='ovr')
    E_in_2, E_out_2 = run(trn_ds2, tst_ds, lbr, model, qs2, quota, cost_matrix)

    qs3 = ALCE(trn_ds3, cost_matrix, SVR())
    model = SVM(decision_function_shape='ovr')
    E_in_3, E_out_3 = run(trn_ds3, tst_ds, lbr, model, qs3, quota, cost_matrix)

    print("Uncertainty: ", E_out_1[::20].tolist())
    print("Random: ", E_out_2[::20].tolist())
    print("ALCE: ", E_out_3[::20].tolist())

    query_num = np.arange(1, quota + 1)
    plt.plot(query_num, E_out_1, 'g', label='Uncertainty sampling')
    plt.plot(query_num, E_out_2, 'k', label='Random')
    plt.plot(query_num, E_out_3, 'r', label='ALCE')
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.5),
               fancybox=True, shadow=True, ncol=5)
    plt.show()


if __name__ == '__main__':
    main()
