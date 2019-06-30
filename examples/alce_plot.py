#!/usr/bin/env python3
"""
Cost-Senstive Multi-Class Active Learning
"""

import copy
import os

import numpy as np
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
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

    for i in range(quota+1):
        # Standard usage of libact objects
        if i > 0:
            ask_id = qs.make_query()
            lb = lbr.label(trn_ds.data[ask_id][0])
            trn_ds.update(ask_id, lb)

        model.train(trn_ds)
        trn_X, trn_y = zip(*trn_ds.get_labeled_entries())
        tst_X, tst_y = zip(*tst_ds.get_labeled_entries())
        C_in = np.append(C_in,
                         calc_cost(trn_y, model.predict(trn_X), cost_matrix))
        C_out = np.append(C_out,
                          calc_cost(tst_y, model.predict(tst_X), cost_matrix))

    return C_in, C_out


def split_train_test(test_size):
    # choose a dataset with unbalanced class instances
    #data = sklearn.datasets.fetch_mldata('segment')
    data = sklearn.datasets.fetch_mldata('vehicle')

    X = StandardScaler().fit_transform(data['data'])
    target = np.unique(data['target'])
    # mapping the targets to 0 to n_classes-1
    y = np.array([np.where(target == i)[0][0] for i in data['target']])

    X_trn, X_tst, y_trn, y_tst = \
        train_test_split(X, y, test_size=test_size, stratify=y)

    # making sure each class appears ones initially
    init_y_ind = np.array(
        [np.where(y_trn == i)[0][0] for i in range(len(target))])
    y_ind = np.array([i for i in range(len(X_trn)) if i not in init_y_ind])
    trn_ds = Dataset(
        np.vstack((X_trn[init_y_ind], X_trn[y_ind])),
        np.concatenate((y_trn[init_y_ind], [None] * (len(y_ind)))))

    tst_ds = Dataset(X_tst, y_tst)

    fully_labeled_trn_ds = Dataset(
        np.vstack((X_trn[init_y_ind], X_trn[y_ind])),
        np.concatenate((y_trn[init_y_ind], y_trn[y_ind])))

    cost_matrix = 2000. * np.random.rand(len(target), len(target))
    np.fill_diagonal(cost_matrix, 0)

    return trn_ds, tst_ds, fully_labeled_trn_ds, cost_matrix


def main():
    test_size = 0.25  # the percentage of samples in the dataset that will be
    # randomly selected and assigned to the test set

    result = {'E1':[], 'E2':[], 'E3':[]}
    for i in range(20):
        trn_ds, tst_ds, fully_labeled_trn_ds, cost_matrix = \
            split_train_test(test_size)
        trn_ds2 = copy.deepcopy(trn_ds)
        trn_ds3 = copy.deepcopy(trn_ds)
        lbr = IdealLabeler(fully_labeled_trn_ds)
        model = SVM(kernel='rbf', decision_function_shape='ovr')

        quota = 100  # number of samples to query

        qs = UncertaintySampling(
            trn_ds, method='sm', model=SVM(decision_function_shape='ovr'))
        _, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs, quota, cost_matrix)
        result['E1'].append(E_out_1)

        qs2 = RandomSampling(trn_ds2)
        _, E_out_2 = run(trn_ds2, tst_ds, lbr, model, qs2, quota, cost_matrix)
        result['E2'].append(E_out_2)

        qs3 = ALCE(trn_ds3, cost_matrix, SVR())
        _, E_out_3 = run(trn_ds3, tst_ds, lbr, model, qs3, quota, cost_matrix)
        result['E3'].append(E_out_3)

    E_out_1 = np.mean(result['E1'], axis=0)
    E_out_2 = np.mean(result['E2'], axis=0)
    E_out_3 = np.mean(result['E3'], axis=0)

    print("Uncertainty: ", E_out_1[::5].tolist())
    print("Random: ", E_out_2[::5].tolist())
    print("ALCE: ", E_out_3[::5].tolist())

    query_num = np.arange(0, quota + 1)
    plt.figure(figsize=(10, 8))
    plt.plot(query_num, E_out_1, 'g', label='Uncertainty sampling')
    plt.plot(query_num, E_out_2, 'k', label='Random')
    plt.plot(query_num, E_out_3, 'r', label='ALCE')
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result')
    plt.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, ncol=5)
    plt.show()


if __name__ == '__main__':
    main()
