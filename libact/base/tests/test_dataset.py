import unittest

import numpy as np

from libact.base.dataset import Dataset


class TestDatasetMethods(unittest.TestCase):

    initial_X = np.arange(15).reshape((5, 3))
    initial_y = np.array([1, 2, None, 1, None])

    def setUp(self):
        self.addTypeEqualityFunc(np.ndarray, self.assertNdArrayEqual)

    def assertNdArrayEqual(self, a, b, msg=None):
        return np.array_equal(a, b)

    def setup_dataset(self):
        return Dataset(self.initial_X, self.initial_y)

    def callback(self, entry_id, new_label):
        self.cb_index = entry_id
        self.cb_label = new_label

    def test_len(self):
        dataset = self.setup_dataset()
        self.assertEqual(len(dataset), 5)
        self.assertEqual(dataset.len_labeled(), 3)
        self.assertEqual(dataset.len_unlabeled(), 2)

    def test_get_num_of_labels(self):
        dataset = self.setup_dataset()
        self.assertEqual(dataset.get_num_of_labels(), 2)

    def test_append(self):
        dataset = self.setup_dataset()
        # labeled
        dataset.append(np.array([9, 8, 7]), 2)
        last_labeled_entry = [e[-1] for e in dataset.get_labeled_entries()]
        self.assertEqual(last_labeled_entry[0], np.array([9, 8, 7]))
        self.assertEqual(last_labeled_entry[1], 2)
        # unlabeled
        idx = dataset.append(np.array([8, 7, 6]))
        last_unlabeled_entry = [e[-1] for e in dataset.get_unlabeled_entries()]
        self.assertEqual(last_unlabeled_entry[0], idx)
        self.assertEqual(last_unlabeled_entry[1], np.array([8, 7, 6]))

    def test_update(self):
        dataset = self.setup_dataset()
        dataset.on_update(self.callback)
        idx = dataset.append(np.array([8, 7, 6]))
        dataset.update(idx, 2)
        self.assertEqual(self.cb_index, idx)
        self.assertEqual(self.cb_label, 2)
        last_labeled_entry = [e[-1] for e in dataset.get_labeled_entries()]
        self.assertEqual(last_labeled_entry[0], np.array([8, 7, 6]))
        self.assertEqual(last_labeled_entry[1], 2)

    def test_format_sklearn(self):
        dataset = self.setup_dataset()
        X, y = dataset.format_sklearn()
        self.assertEqual(X, self.initial_X[[0, 1, 3]])
        self.assertEqual(y, self.initial_y[[0, 1, 3]])

    def test_get_labeled_entries(self):
        dataset = self.setup_dataset()
        X, y = dataset.get_labeled_entries()
        self.assertEqual(X[0], np.array([0, 1, 2]))
        self.assertEqual(X[1], np.array([3, 4, 5]))
        self.assertEqual(X[2], np.array([9, 10, 11]))
        self.assertEqual(y[0], 1)
        self.assertEqual(y[1], 2)
        self.assertEqual(y[2], 1)

    def test_get_unlabeled_entries(self):
        dataset = self.setup_dataset()
        _, X = dataset.get_unlabeled_entries()
        self.assertTrue(np.array_equal(X[0], np.array([6, 7, 8])))
        self.assertTrue(np.array_equal(X[1], np.array([12, 13, 14])))

    def test_labeled_uniform_sample(self):
        dataset = self.setup_dataset()
        pool_X, pool_y = dataset.get_labeled_entries()
        # with replacement
        dataset_s = dataset.labeled_uniform_sample(10)
        for entry_s in zip(*dataset_s.get_labeled_entries()):
            for entry in zip(pool_X, pool_y):
                if np.array_equal(entry_s[0], entry[0]) and entry_s[1] == entry[1]:
                    break
            else:
                self.fail()
        # without replacement
        dataset_s = dataset.labeled_uniform_sample(3, replace=False)
        used_indexes = set()
        for entry_s in zip(*dataset_s.get_labeled_entries()):
            for idx, entry in enumerate(zip(pool_X, pool_y)):
                if (
                    np.array_equal(entry_s[0], entry[0]) and entry_s[1] == entry[1]
                    and idx not in used_indexes
                ):
                    used_indexes.add(idx)
                    break
            else:
                self.fail()
        with self.assertRaises(ValueError):
            dataset_s = dataset.labeled_uniform_sample(4, replace=False)


class TestUpdateBatchMethods(unittest.TestCase):

    initial_X = np.arange(15).reshape((5, 3))
    initial_y = np.array([1, 2, None, 1, None])

    def setup_dataset(self):
        return Dataset(np.copy(self.initial_X), np.copy(self.initial_y))

    def test_callback_stream_matches_sequential(self):
        # update_batch must produce exactly the same (entry_id, label)
        # notification stream and final labels as the equivalent series
        # of individual update() calls.
        ds_batch = self.setup_dataset()
        ds_seq = self.setup_dataset()
        log_batch, log_seq = [], []
        ds_batch.on_update(lambda eid, lbl: log_batch.append((int(eid), lbl)))
        ds_seq.on_update(lambda eid, lbl: log_seq.append((int(eid), lbl)))

        ds_batch.update_batch([2, 4], [3, 5])
        ds_seq.update(2, 3)
        ds_seq.update(4, 5)

        self.assertEqual(log_batch, log_seq)
        self.assertEqual(list(ds_batch.get_entries()[1]),
                         list(ds_seq.get_entries()[1]))
        self.assertTrue(ds_batch.modified)

    def test_incremental_visibility(self):
        # Each callback observes only the labels applied so far — exactly
        # like sequential update() calls (labels are NOT pre-applied
        # atomically, which would leak future labels to observers that
        # read dataset state, e.g. committee retraining).
        ds = self.setup_dataset()
        seen = []
        ds.on_update(lambda eid, lbl: seen.append(ds.len_labeled()))
        ds.update_batch([2, 4], [3, 5])
        self.assertEqual(seen, [4, 5])

    def test_length_mismatch_raises(self):
        ds = self.setup_dataset()
        with self.assertRaises(ValueError):
            ds.update_batch([2, 4], [3])

    def test_scalar_entry_ids_raise_clearly(self):
        # update(entry_id, label) takes scalars; update_batch must reject
        # the analogous misuse with a clear error, not a cryptic one.
        ds = self.setup_dataset()
        with self.assertRaises(ValueError):
            ds.update_batch(2, 3)

    def test_scalar_labels_raise_clearly(self):
        ds = self.setup_dataset()
        with self.assertRaises(ValueError):
            ds.update_batch([2, 4], None)

    def test_multidimensional_entry_ids_raise(self):
        ds = self.setup_dataset()
        with self.assertRaises(ValueError):
            ds.update_batch([[2, 4]], [3, 5])

    def test_duplicate_entry_ids_raise(self):
        ds = self.setup_dataset()
        with self.assertRaises(ValueError):
            ds.update_batch([2, 2], [3, 5])

    def test_empty_is_noop(self):
        ds = self.setup_dataset()
        fired = []
        ds.on_update(lambda eid, lbl: fired.append(eid))
        ds.update_batch([], [])
        self.assertEqual(fired, [])
        self.assertEqual(ds.len_labeled(), 3)

    def test_none_label_unlabels(self):
        ds = self.setup_dataset()
        received = []
        ds.on_update(lambda eid, lbl: received.append((int(eid), lbl)))
        ds.update_batch([0], [None])
        self.assertEqual(ds.len_labeled(), 2)
        self.assertEqual(received, [(0, None)])

    def test_numpy_array_inputs(self):
        ds = self.setup_dataset()
        ds.update_batch(np.array([2, 4]), np.array([3, 5]))
        self.assertEqual(ds.len_labeled(), 5)
        self.assertEqual(ds.get_entries()[1][2], 3)
        self.assertEqual(ds.get_entries()[1][4], 5)

    def test_batch_callback_fires_once_after_all_labels(self):
        # An on_update_batch observer is notified exactly once, with the
        # whole batch, after every label has been applied.
        ds = self.setup_dataset()
        received = []
        ds.on_update_batch(lambda ids, lbls: received.append(
            (list(ids), list(lbls), ds.len_labeled())))
        ds.update_batch([2, 4], [3, 5])
        self.assertEqual(received, [([2, 4], [3, 5], 5)])

    def test_batch_callback_not_fired_by_single_update(self):
        ds = self.setup_dataset()
        received = []
        ds.on_update_batch(lambda ids, lbls: received.append(1))
        ds.update(2, 3)
        self.assertEqual(received, [])

    def test_batch_callback_not_fired_on_empty_batch(self):
        ds = self.setup_dataset()
        received = []
        ds.on_update_batch(lambda ids, lbls: received.append(1))
        ds.update_batch([], [])
        self.assertEqual(received, [])

    def test_batch_aware_observer_skips_per_entry_stream(self):
        # An observer registered on both channels (as every QueryStrategy
        # is) gets the single batch notification instead of the
        # per-entry callbacks; an independent per-entry callback still
        # sees the full incremental stream.
        class Observer(object):
            def __init__(self):
                self.entry_calls = []
                self.batch_calls = []

            def update(self, entry_id, label):
                self.entry_calls.append((int(entry_id), label))

            def update_batch(self, entry_ids, labels):
                self.batch_calls.append(
                    (list(map(int, entry_ids)), list(labels)))

        ds = self.setup_dataset()
        observer = Observer()
        ds.on_update(observer.update)
        ds.on_update_batch(observer.update_batch)
        plain = []
        ds.on_update(lambda eid, lbl: plain.append((int(eid), lbl)))

        ds.update_batch([2, 4], [3, 5])
        self.assertEqual(observer.entry_calls, [])
        self.assertEqual(observer.batch_calls, [([2, 4], [3, 5])])
        self.assertEqual(plain, [(2, 3), (4, 5)])

        # A single update() notifies the per-entry channel as usual.
        ds.update(2, 7)
        self.assertEqual(observer.entry_calls, [(2, 7)])
        self.assertEqual(observer.batch_calls, [([2, 4], [3, 5])])

    def test_out_of_range_entry_ids_raise_before_any_update(self):
        # An out-of-range id must be rejected up front, leaving the dataset
        # untouched — not applied partially up to the offending entry.
        ds = self.setup_dataset()
        fired = []
        ds.on_update(lambda eid, lbl: fired.append(int(eid)))
        with self.assertRaises(ValueError):
            ds.update_batch([2, 4, 9999], [1, 0, 1])
        self.assertEqual(fired, [])
        self.assertEqual(ds.len_labeled(), 3)

    def test_negative_entry_ids_raise(self):
        # Negative ids must raise rather than silently wrapping around to
        # the wrong entry via numpy indexing.
        ds = self.setup_dataset()
        with self.assertRaises(ValueError):
            ds.update_batch([-1], [7])


if __name__ == '__main__':
    unittest.main()
