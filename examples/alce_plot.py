#!/usr/bin/env python3
"""
Cost-Senstive Multi-Class Active Learning
"""

import copy
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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


def split_train_test(test_size):
    # choose a dataset with unbalanced class instances
    data = sklearn.datasets.fetch_mldata('glass')
    X = data['data']
    X = StandardScaler().fit_transform(X)
    target = np.unique(data['target'])
    # mapping the targets to 0 to n_classes-1
    y = np.array([np.where(target == i)[0][0] for i in data['target']])

    n_labeled = len(target)      # number of samples that are initially labeled

    X_trn, X_tst, y_trn, y_tst = \
        train_test_split(X, y, test_size=test_size, stratify=y, random_state=1126)

    # making sure each class appears ones initially
    init_y_ind = np.array([np.where(y_trn == i)[0][0] for i in range(len(target))])
    y_ind = np.array([i for i in range(len(X_trn)) if i not in init_y_ind])

    trn_ds = Dataset(
            np.vstack((X_trn[init_y_ind], X_trn[y_ind])),
            np.concatenate([y_trn[init_y_ind], [None] * (len(y_ind))]))
    tst_ds = Dataset(X_tst, y_tst)
    fully_labeled_trn_ds = Dataset(
            np.vstack((X_trn[init_y_ind], X_trn[y_ind])),
            np.concatenate([y_trn[init_y_ind], y_trn[y_ind]]))

    count = np.bincount(y)
    print(count)
    cost_matrix = np.zeros((len(target), len(target)), dtype='float')
    for i in range(len(target)):
        for j in range(len(target)):
            if i == j:
                cost_matrix[i][j] = 0.
            else:
                cost_matrix[i][j] = 2000 * float(count[j]) / count[i]

    #random_state = np.random.RandomState(1126)
    #cost_matrix = random_state.rand(len(target), len(target))
    #np.fill_diagonal(cost_matrix, 0)

    return trn_ds, tst_ds, y_trn, fully_labeled_trn_ds, cost_matrix


def main():
    # Specifiy the parameters here:
    test_size = 0.25    # the percentage of samples in the dataset that will be
    # randomly selected and assigned to the test set

    # Load dataset
    trn_ds, tst_ds, y_train, fully_labeled_trn_ds, cost_matrix = \
        split_train_test(test_size)
    trn_ds2 = copy.deepcopy(trn_ds)
    trn_ds3 = copy.deepcopy(trn_ds)
    lbr = IdealLabeler(fully_labeled_trn_ds)
    n_classes = len(np.unique(y_train)) # = 7

    quota = 120    # number of samples to query

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

    print("Uncertainty: ", E_out_1[::10].tolist())
    print("Random: ", E_out_2[::10].tolist())
    print("ALCE: ", E_out_3[::10].tolist())

    query_num = np.arange(1, quota + 1)
    plt.figure(figsize=(10,8))
    plt.plot(query_num, E_out_1, 'g', label='Uncertainty sampling')
    plt.plot(query_num, E_out_2, 'k', label='Random')
    plt.plot(query_num, E_out_3, 'r', label='ALCE')
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.show()


if __name__ == '__main__':
    main()
