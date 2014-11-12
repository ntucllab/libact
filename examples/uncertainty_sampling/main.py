#!/usr/bin/env python3

# < Active Learning Experiment >

# Task : Multiclass Classification

# Basic Training Algorithm : Logistic Regression

# Query Strategy - Uncertainty Sampling by -
#                                            (a) Least Confidence
#					     (b) Smallest Margin
#                                            (c) Label Ertropy

import sys
import numpy as np
import matplotlib.pyplot as plt

# including libact from base dir
import os
BASE_DIR = os.path.dirname(__file__) + '/../..'
sys.path.append(BASE_DIR)

from libact.base.dataset import Dataset
from libact.models import LogisticRegression
from libact.query_strategies import UncertaintySampling


def simple_read(file_name) :
    X, y = [], []
    with open(file_name) as f :
        for line in f :
            data = line.split()
            X.append(list(map(float, data[ : -1])))
            y.append(int(data[-1]))
    return np.array(X), np.array(y)


def main():
    X, y = simple_read(BASE_DIR + '/examples/uncertainty_sampling/datasets.txt')

    # shuffle the data
    zipper = list(zip(X, y))
    np.random.shuffle(zipper)
    X, y = zip(*zipper)
    X, y = np.array(X), np.array(y)

    # X is a 2D [m x n] numpy.ndarray, where m is the number of examples and n is the number of features.
    # y is a 1D numpy.ndarray of length m.
    # print('X looks like this :\n' + str(X))
    # print('y looks like this :\n' + str(y))

    N = int(2 * len(X) / 3)  # control the number of training and testing examples

    X_train, y_train = X[ : N], y[ : N]		# training examples, which will then be splitted into two pieces
    X_test, y_test = X[N : ], y[N :]		# testing examples used for calculating E_out

    # < Note >
    # I get this dataset from UCI Machin Learning Repository : http://archive.ics.uci.edu/ml/datasets.html
    # You don't need to know the details of this dataset,
    # all you need to know is : (a) m = 2310, n = 19
    #                           (b) There are 7 (1 ~ 7) labels

    model = LogisticRegression()

    E_in_1, E_out_1 = [], []
    E_in_2, E_out_2 = [], []

    # simulate the scenario when student don't choose which question to ask
    for i in range(10, N) :
        model.train(Dataset(X_train[ : i + 1], y_train[ : i + 1]))
        E_in_1 = np.append(E_in_1, 1 - model.score(Dataset(X_train[ : i + 1], y_train[ : i + 1])))
        E_out_1 = np.append(E_out_1, 1 - model.score(Dataset(X_test, y_test)))

    print('< Scenario 1 > The student doesn\'t choose which question to ask :')
    print('After randomly asking %d questions, (E_in, E_out) = (%f, %f)' % (N -
        10, E_in_1[-1], E_out_1[-1]))

    # ==============================================================================================

    dataset = Dataset(X_train,
        np.concatenate([y_train[:10], [None] * (len(y_train) - 10)]))
    quota = N - 10  # the student can only ask [quota] questions, otherwise the teacher will get unpatient

    # now, the student start asking questions
    qs = UncertaintySampling()
    for i in range(quota) :
        # select a question
        ask_id = qs.make_query(dataset, method=sys.argv[1])

        if i != 0:
            # the student redo the exam and see the result
            E_in_2 = np.append(E_in_2, 1 - qs.get_model().score(dataset))
            E_out_2 = np.append(E_out_2, 1 - qs.get_model().score(Dataset(X_test, y_test)))

        # the student asks the teacher the most confusing question and learns it
        dataset.update(ask_id, y_train[ask_id])

    # the student redo the exam and see the result
    qs.get_model().train(dataset)
    E_in_2 = np.append(E_in_2, 1 - qs.get_model().score(dataset))
    E_out_2 = np.append(E_out_2, 1 - qs.get_model().score(Dataset(X_test, y_test)))

    print('< Scenario 2 > The student chooses which question to ask :')
    print('After wisely asking %d questions, (E_in, E_out) = (%f, %f)' % (quota,
        E_in_2[-1], E_out_2[-1]))

    # now let's plot the result
    query_num = np.arange(1, quota + 1)
    plt.plot(query_num, E_in_1, 'b')    # the blue curve
    plt.plot(query_num, E_in_2, 'r')    # the red curve
    plt.plot(query_num, E_out_1, 'g')   # the green curve
    plt.plot(query_num, E_out_2, 'k')   # the black curve
    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('< Experiment Result >')
    plt.show()


if __name__ == '__main__':
    main()
