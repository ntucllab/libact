import unittest

import numpy as np

from libact.base.dataset import Dataset

try:
    from libact.query_strategies import HintSVM
    HAS_HINTSVM = True
except (ImportError, ModuleNotFoundError):
    HAS_HINTSVM = False


@unittest.skipUnless(HAS_HINTSVM, "HintSVM C extension not compiled")
class HintSVMTestCase(unittest.TestCase):

    def setUp(self):
        self.X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [0, 1],
                  [0, -2], [1.5, 1.5], [-2, -2]]
        self.y = [2, 3, 4, 1, 2, 4]

    def test_hintsvm_multiclass_error(self):
        dataset = Dataset(self.X, np.concatenate([self.y[:6], [None] * 4]))
        qs = HintSVM(dataset)
        with self.assertRaises(ValueError):
            qs.make_query()

    def test_scores_agree_with_make_query(self):
        """argmax(scores) must select the same entry id as make_query().

        _get_scores draws the hint pool from self.random_state_, so identical
        RNG state is required for the two calls to see the same hint pool (and,
        the solver being deterministic, produce identical scores).

        Note: HintSVM's C extension treats label ``0`` as a hint marker, so the
        real labeled classes must be non-zero (here -1 / +1).
        """
        X = [[-10, -1], [-10, 1], [10, -1], [10, 1],
             [0, 0], [-9.5, 0], [9.5, 0], [0.5, 0]]
        y = [-1, -1, 1, 1] + [None] * 4
        dataset = Dataset(X, y)
        qs = HintSVM(dataset, random_state=1126)

        qs.random_state_ = np.random.RandomState(7)
        entry_ids, scores = qs._get_scores()

        qs.random_state_ = np.random.RandomState(7)
        ask_id = qs.make_query()

        self.assertEqual(ask_id, entry_ids[int(np.argmax(scores))])

    def test_scores_direction_prefers_boundary(self):
        """Higher score = closer to the hinted boundary.

        With symmetric, linearly separable clusters, the point sitting on the
        boundary must out-score points buried deep inside either cluster, and
        make_query must return that boundary point.

        p=0 disables hint sampling so the query boundary is the plain max-margin
        separator (x = 0 here). This isolates the score *direction* -- the thing
        this fix changes -- from the orthogonal hint-pull mechanism, which with
        only a handful of unlabeled points would randomly bend the boundary.
        Real labels are -1 / +1 because the C extension reserves label 0 for
        hints.
        """
        X = [[-10, -1], [-10, 1], [10, -1], [10, 1],
             [0, 0], [-9.5, 0], [9.5, 0]]
        y = [-1, -1, 1, 1] + [None] * 3
        dataset = Dataset(X, y)
        boundary_pos = 4  # global entry id of the on-boundary point [0, 0]

        qs = HintSVM(dataset, random_state=1126, p=0.0)
        entry_ids, scores = qs._get_scores()

        entry_ids = list(entry_ids)
        boundary_idx = entry_ids.index(boundary_pos)
        far_idxs = [i for i in range(len(entry_ids)) if i != boundary_idx]

        # Scores are negated absolute decision values.
        self.assertTrue(np.all(scores <= 0))
        # On-boundary point strictly beats every deep-in-cluster point.
        for i in far_idxs:
            self.assertGreater(scores[boundary_idx], scores[i])
        # argmax lands on the on-boundary point ...
        self.assertEqual(int(np.argmax(scores)), boundary_idx)
        # ... and make_query returns its entry id.
        self.assertEqual(qs.make_query(), boundary_pos)


if __name__ == '__main__':
    unittest.main()
