#!/usr/bin/env python3
#
# The script helps guide the users to quickly understand how to use
# libact by going through a simple active learning task with clear
# descriptions.

import os, sys
import getopt
import json

import numpy as np
import matplotlib.pyplot as plt

# add base dir to path
BASE_DIR = os.path.dirname(__file__) + '/..'
sys.path.append(BASE_DIR)

# libact classes
from libact.base.dataset import Dataset, import_libsvm_sparse
import libact.models
import libact.query_strategies


def train_and_plot(model_class, model_params, qs_class, qs_params):
    # Before running this script, you need to make sure you have
    # your dataset under directory libact-root/examples/.
    #
    # You can run libact-root/examples/get_datasets.py to
    # download demo datasets (see get_datasets.py for more details).
    #
    # For now, we assume that you have dataset `heart` under
    # libact-root/examples/.
    X, y = import_libsvm_sparse(os.path.join(BASE_DIR, 'examples/heart.txt')).format_sklearn()

    # shuffle the data
    zipper = list(zip(X, y))
    np.random.shuffle(zipper)
    X, y = zip(*zipper)
    X, y = np.array(X), np.array(y)

    N = int(2 * len(X) / 3)  # control the number of training and testing examples
    X_train, y_train = X[ : N], y[ : N]  # training examples, which will then be splitted into two pieces
    X_test, y_test = X[N : ], y[N : ]    # testing examples used for calculating E_out

    model = model_class(**model_params)

    E_in_1, E_out_1 = [], []
    E_in_2, E_out_2 = [], []

    # simulate the scenario when the student randomly picks questions to ask
    for i in range(10, N) :
        model.train(Dataset(X_train[ : i + 1], y_train[ : i + 1]))
        E_in_1 = np.append(E_in_1, 1 - model.score(Dataset(X_train[ : i + 1], y_train[ : i + 1])))
        E_out_1 = np.append(E_out_1, 1 - model.score(Dataset(X_test, y_test)))

    print('< Scenario 1 > The student doesn\'t choose which question to ask :')
    print('After randomly asking %d questions, (E_in, E_out) = (%f, %f)' % (N - 10, E_in_1[-1], E_out_1[-1]))

    # ==========================================================================

    dataset = Dataset(X_train, np.concatenate([y_train[:10], [None] * (len(y_train) - 10)]))
    quota = N - 10  # the student can only ask [quota] questions, otherwise the teacher will get unpatient

    # now, the student start asking questions
    qs = qs_class(dataset, **qs_params)
    for i in range(quota) :
        # the student asks the teacher the most confusing question and learns it
        ask_id = qs.make_query()
        dataset.update(ask_id, y_train[ask_id])

        # the student redo the exam and see the result
        model.train(dataset)
        E_in_2 = np.append(E_in_2, 1 - model.score(dataset))
        E_out_2 = np.append(E_out_2, 1 - model.score(Dataset(X_test, y_test)))

    print('< Scenario 2 > The student wisely picks question to ask :')
    print('After wisely asking %d questions, (E_in, E_out) = (%f, %f)' % (quota, E_in_2[-1], E_out_2[-1]))

    # now let's plot the result
    # query_num = np.arange(1, quota + 1)
    # plt.plot(query_num, E_in_1, 'b')    # the blue curve
    # plt.plot(query_num, E_in_2, 'r')    # the red curve
    # plt.plot(query_num, E_out_1, 'g')   # the green curve
    # plt.plot(query_num, E_out_2, 'k')   # the black curve
    # plt.xlabel('Number of Queries')
    # plt.ylabel('Error')
    # plt.title('< Experiment Result >')
    # plt.show()

    # now let's plot the result
    query_num = np.arange(1, quota + 1)
    plt.figure(1)
    plt.subplot(211)
    plt.title('Experiment Result')
    plt.xlabel('#(queries)')
    plt.ylabel('error')
    plt.plot(query_num, E_out_1, 'r', label='random')
    plt.plot(query_num, E_out_2, 'g', lanel='yours')
    # plt.legend(loc='lower right')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

def usage():
    print("Usage: %s [options]" % sys.argv[0])
    print("""Options:
-m model              : specify the class name of model used
                        (under libact.models)
--model-params params : specify model parameters (in JSON format)
-q query_strategy     : specify the class name of query strategy used
                        (under libact.query_strategies)
--qs-params params    : specify query stratrgy parameters (in JSON format)""")


def err_exit(errstr, status=1):
    print("Error: %s" % errstr)
    usage()
    sys.exit(status)


def main():
    model_classname = None
    qs_classname = None
    model_params = {}
    qs_params = {}

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'm:q:',
            ['model-params=', 'qs-params='])
        for opt, arg in opts:
            if opt == '-m':
                model_classname = arg
            elif opt == '-q':
                qs_classname = arg
            elif opt == '--model-params':
                model_params = json.loads(arg)
            elif opt == '--qs-params':
                qs_params = json.loads(arg)
            else:
                err_exit("invalid option %s" % opt)
    except getopt.GetoptError:
        usage()
        sys.exit(1)
    except ValueError:
        err_exit("invalid model/query strategy parameters")

    if not model_classname:
        err_exit("model class not specified")
    if not qs_classname:
        err_exit("query strategy class not specified")

    try:
        model_class = getattr(libact.models, model_classname)
    except AttributeError:
        err_exit("specified model does not exist")

    try:
        qs_class = getattr(libact.query_strategies, qs_classname)
    except AttributeError:
        err_exit("specified query strategy does not exist")

    print("Start training with model %s, query strategy %s"
        % (model_classname, qs_classname))
    train_and_plot(model_class, model_params, qs_class, qs_params)


if __name__ == '__main__':
    main()
