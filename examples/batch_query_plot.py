#!/usr/bin/env python3
"""
Batch-mode active learning example.

Sequential active learning retrains the model after every single query
(make_query -> label -> update), which is impractical when training is
expensive. Batch querying selects batch_size points per round
(make_query_batch -> label -> update_batch), cutting the number of
training rounds by a factor of batch_size.

This script compares, on the diabetes dataset:

- sequential uncertainty sampling (one query per training round),
- plain top-k batch uncertainty sampling (one round per batch, but the
  batch may contain redundant near-duplicate points), and
- DiversityWeightedMeta over the same base strategy (one round per
  batch, with diversity-aware batches).
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
from libact.models import LogisticRegression
from libact.query_strategies import DiversityWeightedMeta, UncertaintySampling
from libact.labelers import IdealLabeler


def run_sequential(trn_ds, tst_ds, lbr, model, qs, quota):
    """One query per round: quota training rounds in total."""
    n_labeled_axis, E_out = [], []

    for _ in range(quota):
        ask_id = qs.make_query()
        lb = lbr.label(trn_ds.data[ask_id][0])
        trn_ds.update(ask_id, lb)

        model.train(trn_ds)
        n_labeled_axis.append(trn_ds.len_labeled())
        E_out.append(1 - model.score(tst_ds))

    return n_labeled_axis, E_out


def run_batch(trn_ds, tst_ds, lbr, model, qs, quota, batch_size):
    """batch_size queries per round: quota/batch_size training rounds."""
    n_labeled_axis, E_out = [], []

    for _ in range(quota // batch_size):
        ask_ids = qs.make_query_batch(batch_size)
        labels = [lbr.label(trn_ds.data[ask_id][0]) for ask_id in ask_ids]
        trn_ds.update_batch(ask_ids, labels)

        model.train(trn_ds)
        n_labeled_axis.append(trn_ds.len_labeled())
        E_out.append(1 - model.score(tst_ds))

    return n_labeled_axis, E_out


def split_train_test(dataset_filepath, test_size, n_labeled):
    X, y = import_libsvm_sparse(dataset_filepath).format_sklearn()

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size)
    trn_ds = Dataset(X_train, np.concatenate(
        [y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_test)
    fully_labeled_trn_ds = Dataset(X_train, y_train)

    return trn_ds, tst_ds, fully_labeled_trn_ds


def main():
    dataset_filepath = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'diabetes.txt')
    test_size = 0.33    # fraction of samples assigned to the test set
    n_labeled = 10      # number of samples that are initially labeled
    quota = 120         # number of samples to query in total
    batch_size = 10     # number of samples per batch query

    trn_ds, tst_ds, fully_labeled_trn_ds = \
        split_train_test(dataset_filepath, test_size, n_labeled)
    trn_ds2 = copy.deepcopy(trn_ds)
    trn_ds3 = copy.deepcopy(trn_ds)
    lbr = IdealLabeler(fully_labeled_trn_ds)

    # 1) Sequential uncertainty sampling: one training round per label.
    qs1 = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
    n1, E1 = run_sequential(
        trn_ds, tst_ds, lbr, LogisticRegression(), qs1, quota)
    print('sequential US      : %3d training rounds for %d labels'
          % (quota, quota))

    # 2) Plain top-k batch: batch_size labels per training round. The
    #    batch may contain redundant near-duplicates.
    qs2 = UncertaintySampling(trn_ds2, method='lc', model=LogisticRegression())
    n2, E2 = run_batch(
        trn_ds2, tst_ds, lbr, LogisticRegression(), qs2, quota, batch_size)
    print('top-k batch US     : %3d training rounds for %d labels'
          % (quota // batch_size, quota))

    # 3) Diversity-aware batches over the same base strategy.
    qs3 = DiversityWeightedMeta(
        trn_ds3,
        base_query_strategy=UncertaintySampling(
            trn_ds3, method='lc', model=LogisticRegression()),
        lmbda=0.5,
        random_state=1126,
    )
    n3, E3 = run_batch(
        trn_ds3, tst_ds, lbr, LogisticRegression(), qs3, quota, batch_size)
    print('diversity batch US : %3d training rounds for %d labels'
          % (quota // batch_size, quota))

    plt.plot(n1, E1, 'g', label='sequential US')
    plt.plot(n2, E2, 'b--o', label='top-k batch US')
    plt.plot(n3, E3, 'r--s', label='DiversityWeightedMeta batch')
    plt.xlabel('Number of labeled samples')
    plt.ylabel('Test error')
    plt.title('Sequential vs batch-mode active learning')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=3)
    plt.show()


if __name__ == '__main__':
    main()
