#!/usr/bin/env python3
"""
Multi-label Active Learning
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import make_multilabel_classification

# libact classes
from libact.base.dataset import Dataset
from libact.labelers import IdealLabeler
from libact.query_strategies import RandomSampling
from libact.query_strategies.multilabel import MultilabelWithAuxiliaryLearner
from libact.query_strategies.multilabel import MMC, BinaryMinimization
from libact.models.multilabel import BinaryRelevance
from libact.models import LogisticRegression, SVM

np.random.seed(1126)


def run(trn_ds, tst_ds, lbr, model, qs, quota):
    C_in, C_out = [], []

    for _ in range(quota):
        # Standard usage of libact objects
        ask_id = qs.make_query()
        X, _ = trn_ds.data[ask_id]
        lb = lbr.label(X)
        trn_ds.update(ask_id, lb)

        model.train(trn_ds)
        C_in = np.append(C_in, model.score(trn_ds, criterion='hamming'))
        C_out = np.append(C_out, model.score(tst_ds, criterion='hamming'))

    return C_in, C_out


def split_train_test(test_size):
    # choose a dataset with unbalanced class instances
    data = make_multilabel_classification(
        n_samples=300, n_classes=10, allow_unlabeled=False)
    X = StandardScaler().fit_transform(data[0])
    Y = data[1]

    X_trn, X_tst, Y_trn, Y_tst = train_test_split(X, Y, test_size=test_size)

    trn_ds = Dataset(X_trn, Y_trn[:5].tolist() + [None] * (len(Y_trn) - 5))
    tst_ds = Dataset(X_tst, Y_tst.tolist())

    fully_labeled_trn_ds = Dataset(X_trn, Y_trn)

    return trn_ds, tst_ds, fully_labeled_trn_ds


def main():
    test_size = 0.25  # the percentage of samples in the dataset that will be
    # randomly selected and assigned to the test set

    result = {'E1': [], 'E2': [], 'E3': [], 'E4': [], 'E5': [], 'E6': []}
    for i in range(10):  # repeat experiment
        trn_ds, tst_ds, fully_labeled_trn_ds = split_train_test(test_size)
        trn_ds2 = copy.deepcopy(trn_ds)
        trn_ds3 = copy.deepcopy(trn_ds)
        trn_ds4 = copy.deepcopy(trn_ds)
        trn_ds5 = copy.deepcopy(trn_ds)
        trn_ds6 = copy.deepcopy(trn_ds)
        lbr = IdealLabeler(fully_labeled_trn_ds)
        model = BinaryRelevance(LogisticRegression())

        quota = 150  # number of samples to query

        qs = MMC(trn_ds, br_base=LogisticRegression())
        _, E_out_1 = run(trn_ds, tst_ds, lbr, model, qs, quota)
        result['E1'].append(E_out_1)

        qs2 = RandomSampling(trn_ds2)
        _, E_out_2 = run(trn_ds2, tst_ds, lbr, model, qs2, quota)
        result['E2'].append(E_out_2)

        qs3 = MultilabelWithAuxiliaryLearner(
            trn_ds3,
            BinaryRelevance(LogisticRegression()),
            BinaryRelevance(SVM()),
            criterion='hlr')
        _, E_out_3 = run(trn_ds3, tst_ds, lbr, model, qs3, quota)
        result['E3'].append(E_out_3)

        qs4 = MultilabelWithAuxiliaryLearner(
            trn_ds4,
            BinaryRelevance(LogisticRegression()),
            BinaryRelevance(SVM()),
            criterion='shlr')
        _, E_out_4 = run(trn_ds4, tst_ds, lbr, model, qs4, quota)
        result['E4'].append(E_out_4)

        qs5 = MultilabelWithAuxiliaryLearner(
            trn_ds5,
            BinaryRelevance(LogisticRegression()),
            BinaryRelevance(SVM()),
            criterion='mmr')
        _, E_out_5 = run(trn_ds5, tst_ds, lbr, model, qs5, quota)
        result['E5'].append(E_out_5)

        qs6 = BinaryMinimization(trn_ds6, LogisticRegression())
        _, E_out_6 = run(trn_ds6, tst_ds, lbr, model, qs6, quota)
        result['E6'].append(E_out_6)

    E_out_1 = np.mean(result['E1'], axis=0)
    E_out_2 = np.mean(result['E2'], axis=0)
    E_out_3 = np.mean(result['E3'], axis=0)
    E_out_4 = np.mean(result['E4'], axis=0)
    E_out_5 = np.mean(result['E5'], axis=0)
    E_out_6 = np.mean(result['E6'], axis=0)

    print("MMC: ", E_out_1[::5].tolist())
    print("Random: ", E_out_2[::5].tolist())
    print("MultilabelWithAuxiliaryLearner_hlr: ", E_out_3[::5].tolist())
    print("MultilabelWithAuxiliaryLearner_shlr: ", E_out_4[::5].tolist())
    print("MultilabelWithAuxiliaryLearner_mmr: ", E_out_5[::5].tolist())
    print("BinaryMinimization: ", E_out_6[::5].tolist())

    query_num = np.arange(1, quota + 1)
    fig = plt.figure(figsize=(9, 6))
    ax = plt.subplot(111)
    ax.plot(query_num, E_out_1, 'g', label='MMC')
    ax.plot(query_num, E_out_2, 'k', label='Random')
    ax.plot(query_num, E_out_3, 'r', label='AuxiliaryLearner_hlr')
    ax.plot(query_num, E_out_4, 'b', label='AuxiliaryLearner_shlr')
    ax.plot(query_num, E_out_5, 'c', label='AuxiliaryLearner_mmr')
    ax.plot(query_num, E_out_6, 'm', label='BinaryMinimization')

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.xlabel('Number of Queries')
    plt.ylabel('Loss')
    plt.title('Experiment Result (Hamming Loss)')
    plt.show()


if __name__ == '__main__':
    main()
