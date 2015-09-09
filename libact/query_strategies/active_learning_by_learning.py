"""Active learning by learning
"""
from libact.base.interfaces import QueryStrategy
import numpy as np


class ActiveLearmingByLearning(QueryStrategy):

    def __init__(self, *args, **kwargs):
        """ """
        super(ActiveLearningByLearning, self).__init__(*args, **kwargs)
        self.models = kwargs.pop('models', None)
        if self.models is none:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'models'"
                )
        elif not models:
            raise ValueError("models list is empty")

        # parameters for Exp4.p
        self.delta = kwargs.pop('delta', 1.)
        self.pmin = kwargs.pop('pmin', 0.05)
        # query budget
        self.T = kwargs.pop('T', 100)

        self.exp4p = Exp4P(
                    experts = self.models,
                    T = self.T,
                    delta = self.delta,
                    K = self.dataset.len_unlabeled()
                )
        self.budget_used = 0

        # Classifier instance
        sef.clf = kwargs.pop('clf', None)
        if self.clf is none:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'models'"
                )

        self.reward = -1
        self.W = []
        self.queried_hist = []

    def calc_reward_fn(self):
        clf = copy.copy(self.clf)
        clf.train(self.dataset)

        # reward function: Importance-Weighted-Accuracy (IW-ACC) (tau, f)
        self.reward = 0.
        for i in range(self.exp4p.t):
            self.reward += self.W *\
                    (clf.predict(self.dataset.data[self.queried_hist[i]][0]) == \
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
            p = exp4p.next(self.reward)
            ask_id = np.random.choice(np.arange(self.K), size=1, p=p)

            self.W.append(1./p[ask_id])
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
        if experts is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'experts'"
                )
        elif not experts:
            raise ValueError("experts list is empty")

        self.N = len(self.experts)
        # weight vector to each experts, shape = (N, )
        self.w = np.array([1. for i in range(self.N)])
        # max iters
        self.T = kwargs.pop('T', 100)
        # delta > 0
        self.delta = kwargs.pop('delta', 1.0)

        # n_arms
        self.K = kwargs.pop('K', None)
        if not self.K:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'K'"
                )

        # p_min in [0, 1/n_arms]
        self.pmin = kwargs.pop('pmin', np.sqrt(np.log(self.N) / self.K / self.T))

        self.exp4p_gen = exp4p()

    # Python3 compatibility
    def __next__(self, reward):
        """ """
        return self.next(reward)

    def next(self, reward):
        # first run don't have reward, TODO exception on reward == -1 only once
        if reward == -1:
            return self.exp4p_gen.next()
        else:
        # TODO exception on reward in [0, 1]
            return self.exp4p_gen.send(reward)

    def update_experts(qid, lbl):
        for expert in self.experts:
            expert.update(qid, lbl)

    def exp4p(self):
        #TODO probabilistic active learning algorithm
        self.t = 0
        rhat = np.zeros((self.K,))
        yhat = np.zeros((self.N,))
        vhat = np.zeros((self.N,))
        while self.t < self.T:
            advice = np.zeros((self.N, self.K))
            for i, expert in enumerate(experts):
                advice[i, :] = expert.make_query()
            W = np.sum(self.w)

            # shape = (self.N, )
            p = (1 - self.K * self.pmin) * \
                    np.sum(np.tile(self.w, (self.K, 1)).T * advice, axis=0) + \
                    self.pmin

            reward, ask_id, lbl = yield p
            self.update_experts(ask_id, lbl)

            rhat[ask_id] = reward / p[ask_id]

            for i in range(self.N):
                yhat[i] = np.dot(advice[i], rhat)
                vhat[i] = np.sum(advice[i] / p)

                self.w[i] = self.w[i] * np.exp(
                        self.pmin / 2 * (yhat[i] + vhat[i]*np.sqrt(
                            np.log(self.N/self.delta) / self.K / self.T)
                            )
                        )

            self.t += 1

        raise StopIteration


