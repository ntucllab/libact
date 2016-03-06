from numpy.testing import assert_array_equal
import numpy as np

from libact.base.interfaces import QueryStrategy, ContinuousModel
from libact.base.dataset import Dataset, import_libsvm_sparse
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler

X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [-2, -2], [0, -2], \
        [0, 1], [1.5, 1.5]]
Y = [-1, -1, -1, 1, 1, 1, -1, -1, 1, 1]
fully_labeled_trn_ds = Dataset(X, Y)
lbr = IdealLabeler(fully_labeled_trn_ds)
quota = 4    

def init_toyexample():
    trn_ds = Dataset(X, np.concatenate([Y[:6], [None] * 4]))
    return trn_ds

def run_qs(trn_ds, lbr, model, qs, quota):
    qseq = []
    for i in range(quota) :
        ask_id = qs.make_query()
        X, y = zip(*trn_ds.data)
        lb = lbr.label(X[ask_id])
        trn_ds.update(ask_id, lb)
        qseq.append(ask_id)

    return np.array(qseq)

    
def test_uncertainty_lc():
    trn_ds = init_toyexample()
    qs = UncertaintySampling(trn_ds, method='lc', model=LogisticRegression())
    model = LogisticRegression()
    qseq = run(trn_ds, lbr, model, qs, quota)
    assert_array_equal(qseq, np.array(Y[6:]))
    
def test_uncertainty_sm():
    trn_ds = init_toyexample()
    qs = UncertaintySampling(trn_ds, method='sm', model=LogisticRegression())
    model = LogisticRegression()
    qseq = run(trn_ds, lbr, model, qs, quota)
    assert_array_equal(qseq, np.array(Y[6:]))
