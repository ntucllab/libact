#!/usr/bin/env python3
#
# The script helps guide the users to quickly understand how to use
# libact by going through a simple active learning task with clear
# descriptions.

import copy
import time
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
        #start_time = time.time()
        ask_id = qs.make_query()
        #print( "the %s query, %s, takes %s seconds" % (i, ask_id, time.time() - start_time) )
        trn_ds.update(ask_id, y_train[ask_id])

        model.train(trn_ds)
        E_in = np.append(E_in, 1 - model.score(trn_ds))
        E_out = np.append(E_out, 1 - model.score(tst_ds))

    return E_in, E_out


def split_train_test():
    X, y = import_libsvm_sparse('./examples/diabetes.txt').format_sklearn()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    n_labeled = 10

    trn_ds = Dataset(X_train, np.concatenate([y_train[:n_labeled], [None] * (len(y_train) - n_labeled)]))
    tst_ds = Dataset(X_test, y_test)

    return trn_ds, tst_ds, y_train


def main():
    #quota = len(y_train) - 10
    quota = 10
    Tsim = 20
    qs_list = ['quire','uncertain','random','qbc','albl','vr']
    exp_qs_idx = [0, 2]
    for i in range(len(qs_list)):
        exp_qs_list = [qs_list[i] for i in exp_qs_idx]
    E_in = np.zeros( [len(exp_qs_list), quota] )
    E_out = np.zeros( [len(exp_qs_list), quota] )
    avg_Ein = np.zeros( [len(exp_qs_list), quota] )
    avg_Eout = np.zeros( [len(exp_qs_list), quota] )
    linestyle_list = ['-', '--', ':']
    color_list = ['b', 'r', 'k']
    list_trn_ds = []
    for t in range(Tsim):
        trn_ds, tst_ds, y_train = split_train_test()
        for i in range(len(exp_qs_list)):
            list_trn_ds.append( copy.deepcopy(trn_ds) )
        i = 0
        if 'quire' in exp_qs_list:
            qs = QUIRE(list_trn_ds[i])
            model = LogisticRegression()
            e_in, e_out = run(list_trn_ds[i], tst_ds, y_train, model, qs, quota)
            E_in[i] = e_in
            E_out[i] = e_out
            i = i + 1

        if 'uncertain' in exp_qs_list:
            qs = UncertaintySampling(list_trn_ds[i], method='lc')
            model = LogisticRegression()
            e_in, e_out = run(list_trn_ds[i], tst_ds, y_train, model, qs, quota)
            E_in[i] = e_in
            E_out[i] = e_out
            i = i + 1
        
        if 'random' in exp_qs_list:
            qs = RandomSampling(list_trn_ds[i])
            model = LogisticRegression()
            e_in, e_out = run(list_trn_ds[i], tst_ds, y_train, model, qs, quota)
            E_in[i] = e_in
            E_out[i] = e_out
            i = i + 1

        avg_Ein = avg_Ein + E_in
        avg_Eout = avg_Eout + E_out

    avg_Ein = avg_Ein/Tsim
    avg_Eout = avg_Eout/Tsim
    
    query_num = np.arange(1, quota + 1)
    for i in range(len(exp_qs_list)):
        plt.plot(query_num, avg_Ein[i], color_list[0]+linestyle_list[i], lw=2.0 , label=exp_qs_list[i]+' Ein')
    for i in range(len(exp_qs_list)):
        plt.plot(query_num, avg_Eout[i], color_list[1]+linestyle_list[i], lw=2.0 , label=exp_qs_list[i]+' Eout')

    plt.xlabel('Number of Queries')
    plt.ylabel('Error')
    plt.title('Experiment Result')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    plt.show()

if __name__ == '__main__':
    main()
