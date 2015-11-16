#!/usr/bin/env python3
#
# The script helps guide the users to quickly understand how to use
# libact by going through a simple active learning task with clear
# descriptions.

import copy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

# libact classes
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import *
from libact.query_strategies import *


def run(trn_ds, tst_ds, y_train, model, qs, quota):
    E_in, E_out = [], []

    for i in range(quota) :
        ask_id = qs.make_query()
        trn_ds.update(ask_id, y_train[ask_id])

        model.train(trn_ds)
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))

    return E_in, E_out


def split_train_test():
    X, y = import_libsvm_sparse('./examples/diabetes').format_sklearn()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    n_labeled = 10

    trn_ds = Dataset(X_train, np.concatenate([y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_test)

    return trn_ds, tst_ds, y_train


def main():
    trn_ds, tst_ds, y_train = split_train_test()
    trn_ds2 = copy.deepcopy(trn_ds)

    quota = len(y_train) - 10

    qs = UncertaintySampling(trn_ds, method='lc')
    model = LogisticRegression()
    E_in_1, E_out_1 = run(trn_ds, tst_ds, y_train, model, qs, quota)

    qs2 = RandomSampling(trn_ds2)
    model = LogisticRegression()
    E_in_2, E_out_2 = run(trn_ds2, tst_ds, y_train, model, qs2, quota)

    query_num = np.arange(1, quota + 1)
    plt.plot(query_num, E_in_1, 'b', label='random Ein')
    plt.plot(query_num, E_in_2, 'r', label='qs Ein')
    plt.plot(query_num, E_out_1, 'g', label='random Eout')
    plt.plot(query_num, E_out_2, 'k', label='qs Eout')
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('< Experiment Result >')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.show()


if __name__ == '__main__':
    main()
