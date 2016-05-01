"""This module includes some functions to be reused in query strategy testing.
"""

import numpy as np


def run_qs(trn_ds, qs, truth, quota):
    """Run query strategy on specified dataset and return quering sequence.

    Parameters
    ----------
    trn_ds : Dataset object
        The dataset to be run on.

    qs : QueryStrategy instance
        The active learning algorith to be run.

    truth : array-like
        The true label.

    quota : int
        Number of iterations to run

    Returns
    -------
    qseq : numpy array, shape (quota,)
        The numpy array of entry_id representing querying sequence.
    """
    ret = []
    for _ in range(quota):
        ask_id = qs.make_query()
        trn_ds.update(ask_id, truth[ask_id])

        ret.append(ask_id)
    return np.array(ret)
