#!/usr/bin/env python3
"""
The script runs experiments to compare the performance of ALBL and other active
learning algorithms.
"""

import copy
import os

import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split

# libact classes
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import SVM
from libact.query_strategies import QUIRE, UncertaintySampling, RandomSampling,\
    ActiveLearningByLearning, HintSVM
from libact.labelers import IdealLabeler


def run(trn_ds, tst_ds, lbr, model, qs, quota):
    E_in, E_out = [], []

    for _ in range(quota):
        ask_id = qs.make_query()
        lb = lbr.label(trn_ds.data[ask_id][0])
        trn_ds.update(ask_id, lb)

        model.train(trn_ds)
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))

    return E_in, E_out


def split_train_test(dataset_filepath, test_size, n_labeled):
    X, y = import_libsvm_sparse(dataset_filepath).format_sklearn()

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size)

    while len(np.unique((y_train[:n_labeled]))) != 2:
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=test_size)

    trn_ds = Dataset(X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_test)
    fully_labeled_trn_ds = Dataset(X_train, y_train)

    return trn_ds, tst_ds, y_train, fully_labeled_trn_ds


def main():
    # Specifiy the parameters here:
    # path to your binary classification dataset
    ds_name = 'australian'
    dataset_filepath = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '%s.txt' % ds_name)
    test_size = 0.33    # the percentage of samples in the dataset that will be
                        # randomly selected and assigned to the test set
    n_labeled = 10      # number of samples that are initially labeled
    results = []

    for T in range(20): # repeat the experiment 20 times
        print("%dth experiment" % (T+1))

        trn_ds, tst_ds, y_train, fully_labeled_trn_ds = \
            split_train_test(dataset_filepath, test_size, n_labeled)
        trn_ds2 = copy.deepcopy(trn_ds)
        trn_ds3 = copy.deepcopy(trn_ds)
        trn_ds4 = copy.deepcopy(trn_ds)
        trn_ds5 = copy.deepcopy(trn_ds)
        lbr = IdealLabeler(fully_labeled_trn_ds)

        quota = len(y_train) - n_labeled    # number of samples to query

        # Comparing UncertaintySampling strategy with RandomSampling.
        # model is the base learner, e.g. LogisticRegression, SVM ... etc.
        qs = UncertaintySampling(trn_ds,
                                 model=SVM(decision_function_shape='ovr'))
        model = SVM(kernel='linear', decision_function_shape='ovr')
        _, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs, quota)
        results.append(E_out_1.tolist())

        qs2 = RandomSampling(trn_ds2)
        model = SVM(kernel='linear', decision_function_shape='ovr')
        _, E_out_2 = run(trn_ds2, tst_ds, lbr, model, qs2, quota)
        results.append(E_out_2.tolist())

        qs3 = QUIRE(trn_ds3)
        model = SVM(kernel='linear', decision_function_shape='ovr')
        _, E_out_3 = run(trn_ds3, tst_ds, lbr, model, qs3, quota)
        results.append(E_out_3.tolist())

        qs4 = HintSVM(trn_ds4, cl=1.0, ch=1.0)
        model = SVM(kernel='linear', decision_function_shape='ovr')
        _, E_out_4 = run(trn_ds4, tst_ds, lbr, model, qs4, quota)
        results.append(E_out_4.tolist())

        qs5 = ActiveLearningByLearning(trn_ds5,
                    query_strategies=[
                        UncertaintySampling(trn_ds5,
                                            model=SVM(kernel='linear',
                                            decision_function_shape='ovr')),
                        QUIRE(trn_ds5),
                        HintSVM(trn_ds5, cl=1.0, ch=1.0),
                    ],
                    T=quota,
                    uniform_sampler=True,
                    model=SVM(kernel='linear', decision_function_shape='ovr')
                )
        model = SVM(kernel='linear', decision_function_shape='ovr')
        _, E_out_5 = run(trn_ds5, tst_ds, lbr, model, qs5, quota)
        results.append(E_out_5.tolist())

    result = []
    for i in range(5):
        _temp = []
        for j in range(i, len(results), 5):
            _temp.append(results[j])
        result.append(np.mean(_temp, axis=0))

    # Plot the learning curve of UncertaintySampling to RandomSampling
    # The x-axis is the number of queries, and the y-axis is the corresponding
    # error rate.
    query_num = np.arange(1, quota + 1)
    plt.plot(query_num, result[0], 'g', label='uncertainty sampling')
    plt.plot(query_num, result[1], 'k', label='random')
    plt.plot(query_num, result[2], 'r', label='QUIRE')
    plt.plot(query_num, result[3], 'b', label='HintSVM')
    plt.plot(query_num, result[4], 'c', label='ALBL')
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=5)
    plt.show()


if __name__ == '__main__':
    main()
