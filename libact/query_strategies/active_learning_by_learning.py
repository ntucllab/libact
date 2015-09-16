"""Active learning by learning
"""
from libact.base.interfaces import QueryStrategy
import numpy as np
import copy


class ActiveLearningByLearning(QueryStrategy):

    def __init__(self, *args, **kwargs):
        """ ALBL """
        super(ActiveLearningByLearning, self).__init__(*args, **kwargs)
        self.models = kwargs.pop('models', None)
        if self.models is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'models'"
                )
        elif not self.models:
            raise ValueError("models list is empty")

        # parameters for Exp4.p
        self.delta = kwargs.pop('delta', 1.)
        self.pmin = kwargs.pop('pmin', 0.05)
        # query budget
        self.T = kwargs.pop('T', 100)

        self.unlabeled_entry_ids, X_pool = zip(*self.dataset.get_unlabeled_entries())
        self.invert_id_idx = {}
        for i, entry in enumerate(self.dataset.get_unlabeled_entries()):
            self.invert_id_idx[entry[0]] = i

        self.exp4p = Exp4P(
                    experts = self.models,
                    T = self.T,
                    delta = self.delta,
                    K = self.dataset.len_unlabeled(),
                    invert_id_idx = self.invert_id_idx,
                )
        self.budget_used = 0

        # classifier instance
        self.clf = kwargs.pop('clf', None)
        if self.clf is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'models'"
                )

        self.reward = -1.
        self.W = []
        self.queried_hist = []

    def calc_reward_fn(self):
        clf = copy.copy(self.clf)
        clf.train(self.dataset)

        # reward function: Importance-Weighted-Accuracy (IW-ACC) (tau, f)
        self.reward = 0.
        for i in range(len(self.queried_hist)):
            self.reward += self.W[i] *\
                    (clf.predict(self.dataset.data[self.queried_hist[i]][0])[0] == \
                     self.dataset.data[self.queried_hist[i]][1])
        self.reward /= (self.dataset.len_labeled() + self.dataset.len_unlabeled())
        self.reward /= self.T

    def update(self, entry_id, label):
        # TODO
        # it is expected user to call update after makeing a query to update the
        # query
        self.calc_reward_fn()

    def make_query(self):
        dataset = self.dataset
        unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())

        while self.budget_used < self.T:
            # query vector on unlabeled instances
            if self.reward == -1.:
                p = self.exp4p.next(self.reward, None, None)
            else:
                p = self.exp4p.next(
                        self.reward,
                        self.queried_hist[-1],
                        self.dataset.data[self.queried_hist[-1]][1]
                        )
            ask_idx = np.random.choice(np.arange(self.exp4p.K), size=1, p=p)[0]
            ask_id = self.unlabeled_entry_ids[ask_idx]

            self.W.append(1./p[ask_idx])
            self.queried_hist.append(ask_id)
            if ask_id in unlabeled_entry_ids:
                self.budget_used += 1
                return ask_id
            else:
                self.calc_reward_fn()


class Exp4P():

    def __init__(self, *args, **kwargs):
        """ """
        # QueryStrategy class object instances
        self.experts = kwargs.pop('experts', None)
        if self.experts is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'experts'"
                )
        elif not self.experts:
            raise ValueError("experts list is empty")

        self.N = len(self.experts)
        # weight vector to each experts, shape = (N, )
        self.w = np.array([1. for i in range(self.N)])
        # max iters
        self.T = kwargs.pop('T', 100)
        # delta > 0
        self.delta = kwargs.pop('delta', 1.0)

        # n_arms --> n_unlabeled_data
        self.K = kwargs.pop('K', None)
        if not self.K:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'K'"
                )

        # p_min in [0, 1/n_arms]
        self.pmin = kwargs.pop('pmin', np.sqrt(np.log(self.N) / self.K / self.T))

        self.exp4p_gen = self.exp4p()

        self.invert_id_idx = kwargs.pop('invert_id_idx')
        if not self.invert_id_idx:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'invert_id_idx'"
                )

    # Python3 compatibility
    def __next__(self, reward, ask_id, lbl):
        """ """
        return self.next(reward ,ask_id, lbl)

    def next(self, reward, ask_id, lbl):
        # first run don't have reward, TODO exception on reward == -1 only once
        if reward == -1:
            return next(self.exp4p_gen)
        else:
        # TODO exception on reward in [0, 1]
            return self.exp4p_gen.send((reward, ask_id, lbl))

    def update_experts(self, qid, lbl):
        for expert in self.experts:
            expert.update(qid, lbl)

    def exp4p(self):
        self.t = 0

        rhat = np.zeros((self.K,))
        yhat = np.zeros((self.N,))
        vhat = np.zeros((self.N,))
        while self.t < self.T:
            advice = np.zeros((self.N, self.K))
            for i, expert in enumerate(self.experts):
            #TODO probabilistic active learning algorithm
                advice[i][self.invert_id_idx[expert.make_query()]] = 1
            W = np.sum(self.w)

            # shape = (self.K, )
            p = (1 - self.K * self.pmin) * \
                    np.sum(np.tile(self.w, (self.K, 1)).T * advice, axis=0) / W + \
                    self.pmin

            reward, ask_id, lbl = yield p
            self.update_experts(ask_id, lbl)
            ask_idx = self.invert_id_idx[ask_id]

            rhat[ask_idx] = reward / p[ask_idx]
            #for i in range(self.N):
            #    yhat[i] = np.dot(advice[i], rhat)
            #    vhat[i] = np.sum(advice[i] / p)

            #    self.w[i] = self.w[i] * np.exp(
            #            self.pmin / 2 * (yhat[i] + vhat[i]*np.sqrt(
            #                np.log(self.N/self.delta) / self.K / self.T)
            #                )
            #            )
            yhat = np.dot(advice, rhat)
            vhat = np.sum(advice / np.tile(p, (self.N, 1)), axis=1)
            self.w = self.w * np.exp(
                    self.pmin / 2 * (yhat + vhat*np.sqrt(
                        np.log(self.N/self.delta) / self.K / self.T)
                        )
                    )

            self.t += 1

        raise StopIteration


