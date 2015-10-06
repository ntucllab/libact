"""Active learning by learning (ALBL)

This module includes two classes. ActiveLearningByLearning is the main algorithm
for ALBL and Exp4P is the multi-armed bandit algorithm which will be used in
ALBL.

"""
from libact.base.interfaces import QueryStrategy
import numpy as np
import copy


class ActiveLearningByLearning(QueryStrategy):
    """Active Learning By Learning (ALBL) query strategy.

    ALBL is an active learning algorithm that adaptively choose among existing
    query strategies to decide which data to make query. It utilizes multi-armed
    bandit algorithm Exp4.P to adaptively make such decision. More detail of
    ALBL can refer to the work listed in the reference section.

    Parameters
    ----------
    models: list of libact.query_strategies.* object instance
        The active learning algorithms used in ALBL, which will be the experts
        in the multi-armed bandit algorithm Exp4.P.

    delta: float, optional (default=1.)
        Parameter for Exp4.P.

    pmin: float, 0<pmin<1, optional (default=0.05)
        Parameter for Exp4.P. The minimal probability for random selection of
        the arms (aka the unlabeled data).

    uniform_expert: {True, False}, optional (default=Truee)
        Determining whether to include uniform random sample as one of expert.

    T: integer, optional (default=100)
        Total query budget.

    clf: libact.model.* object instance
        The learning model used for the task.


    Attributes
    ----------
    models_: list of libact.query_strategies.* object instance

    exp4p_: instance of Exp4P object

    queried_hist_: list of integer
        A list of entry_id of the dataset which is queried in the past.


    Reference
    ---------

    .. [1] Hsu, Wei-Ning, and Hsuan-Tien Lin. "Active Learning by Learning."
           Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.
    """

    def __init__(self, *args, **kwargs):
        super(ActiveLearningByLearning, self).__init__(*args, **kwargs)
        self.models_ = kwargs.pop('models', None)
        if self.models_ is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'models'"
                )
        elif not self.models_:
            raise ValueError("models list is empty")

        # parameters for Exp4.p
        self.delta = kwargs.pop('delta', 1.)
        self.pmin = kwargs.pop('pmin', None)
        if self.pmin and (self.pmin >= 1. or self.pmin <= 0):
            raise ValueError("pmin should be 0 < pmin < 1")
        # query budget
        self.T = kwargs.pop('T', 100)

        self.unlabeled_entry_ids, X_pool = zip(*self.dataset.get_unlabeled_entries())
        self.invert_id_idx = {}
        for i, entry in enumerate(self.dataset.get_unlabeled_entries()):
            self.invert_id_idx[entry[0]] = i

        self.uniform_expert = kwargs.pop('uniform_expert', True)
        self.exp4p_ = Exp4P(
                    experts = self.models_,
                    T = self.T * 3,
                    delta = self.delta,
                    invert_id_idx = self.invert_id_idx,
                    uniform_expert = self.uniform_expert
                )
        self.budget_used = 0

        # classifier instance
        self.clf = kwargs.pop('clf', None)
        if self.clf is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'models'"
                )

        self.W = []
        self.queried_hist_ = []


    def calc_reward_fn(self):
        """Calculate the reward value"""
        clf = copy.copy(self.clf)
        clf.train(self.dataset)

        # reward function: Importance-Weighted-Accuracy (IW-ACC) (tau, f)
        reward = 0.
        for i in range(len(self.queried_hist_)):
            reward += self.W[i] *\
                (clf.predict(self.dataset.data[self.queried_hist_[i]][0])[0] == \
                 self.dataset.data[self.queried_hist_[i]][1])
        reward /= (self.dataset.len_labeled() + self.dataset.len_unlabeled())
        reward /= self.T
        return reward

    def calc_query(self):
        dataset = self.dataset
        try:
            unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())
        except ValueError:
            # might be no more unlabeled data left
            return

        while self.budget_used < self.T:
            try:
                q = self.exp4p_.next(
                        self.calc_reward_fn(),
                        self.queried_hist_[-1],
                        self.dataset.data[self.queried_hist_[-1]][1]
                        )
            except StopIteration:
                # early stop, out of budget for Exp4.P
                pass
            ask_idx = np.random.choice(
                        np.arange(len(self.invert_id_idx)), size=1, p=q
                    )[0]
            ask_id = self.unlabeled_entry_ids[ask_idx]
            self.W.append(1./q[ask_idx])
            self.queried_hist_.append(ask_id)

            if ask_id in unlabeled_entry_ids:
                self.budget_used += 1
                return ask_id

    def update(self, entry_id, label):
        """Caluculate the next query after updated the question asked with
        answer."""
        self.calc_query()

    def make_query(self):
        """Except for the initial query, it return the id of the data albl wants
        to query."""
        if self.queried_hist_ == []:
            # initial query
            q = self.exp4p_.next(-1, None, None)
            ask_idx = np.random.choice(
                        np.arange(len(self.invert_id_idx)), size=1, p=q
                    )[0]
            ask_id = self.unlabeled_entry_ids[ask_idx]
            self.W.append(1./q[ask_idx])
            self.queried_hist_.append(ask_id)
        return self.queried_hist_[-1]



class Exp4P():
    """A multi-armed bandit algorithm Exp4.P.

    Parameters
    ----------
    experts: QueryStrategy instances
        The active learning algorithms wanted to use.

    invert_id_idx: dictionary
        A look up table for the correspondance of entry_id to the index of the
        unlabeled data.

    delta: float, >0, optional (default=1.)
        A parameter.

    pmin: float, 0<pmin<1, optional (default=:math:`\frac{√{log(N)}{KT}`)
        The minimal probability for random selection of the arms (aka the
        unlabeled data).

    T: integer, optional (default=100)
        The maximum number of rounds.

    K: integer
        The number of arms (number of unlabel data).

    uniform_expert: {True, False}, optional (default=Truee)
        Determining whether to include uniform random sample as one of expert.


    Attributes
    ----------
    t: integer
        The current round this instance is at.

    N: integer
        The number of experts in this exp4.p instance.


    Reference
    ---------

    .. [1] Beygelzimer, Alina, et al. "Contextual bandit algorithms with
           supervised learning guarantees." arXiv preprint arXiv:1002.4058
           (2010).
    """

    def __init__(self, *args, **kwargs):
        """ """
        # QueryStrategy class object instances
        self.experts_ = kwargs.pop('experts', None)
        if self.experts_ is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'experts'"
                )
        elif not self.experts_:
            raise ValueError("experts list is empty")

        # whether to include uniform random sample as one of expert
        self.uniform_expert = kwargs.pop('uniform_expert', True)

        # n_experts
        if self.uniform_expert:
            self.N = len(self.experts_) + 1
        else:
            self.N = len(self.experts_)

        # weight vector to each experts, shape = (N, )
        self.w = np.array([1. for i in range(self.N)])

        # max iters
        self.T = kwargs.pop('T', 100)

        # delta > 0
        self.delta = kwargs.pop('delta', 1.0)

        # n_arms --> n_experts (n_query_algorithms)
        self.K = self.N

        # p_min in [0, 1/n_arms]
        self.pmin = kwargs.pop('pmin', np.sqrt(np.log(self.N) / self.K / self.T))

        self.exp4p_gen = self.exp4p()

        self.invert_id_idx = kwargs.pop('invert_id_idx')
        if not self.invert_id_idx:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'invert_id_idx'"
                )

    def __next__(self, reward, ask_id, lbl):
        """For Python3 compatibility of generator."""
        return self.next(reward ,ask_id, lbl)

    def next(self, reward, ask_id, lbl):
        """Taking the label and the reward value of last question and returns
        the next question to ask."""
        # first run don't have reward, TODO exception on reward == -1 only once
        if reward == -1:
            return next(self.exp4p_gen)
        else:
        # TODO exception on reward in [0, 1]
            return self.exp4p_gen.send((reward, ask_id, lbl))

    def update_experts(self, qid, lbl):
        """Update expert's dataset after making a query."""
        for expert in self.experts_:
            expert.update(qid, lbl)

    def exp4p(self):
        """The generator which implements the main part of Exp4.P.

        Yield
        -------
        p: array-like, shape = [K]
            The query vector which tells ALBL what kind of distribution if
            should sample from the unlabeled pool.


        Send
        ----
        reward: float
            The reward value calculated from ALBL.

        ask_id: integer
            The entry_id of the sample point ALBL asked.

        lbl: integer
            The answer received from asking the entry_id ask_id.
        """

        self.t = 0

        while self.t < self.T:
            #TODO probabilistic active learning algorithm
            # len(self.invert_id_idx) is the number of unlabeled data
            query = np.zeros((self.N, len(self.invert_id_idx)))
            if self.uniform_expert:
                query[-1, :] = 1. / len(self.invert_id_idx)
            for i, expert in enumerate(self.experts_):
                query[i][self.invert_id_idx[expert.make_query()]] = 1

            # choice vector, shape = (self.K, )
            W = np.sum(self.w)
            p = (1 - self.K * self.pmin) * self.w / W + self.pmin

            # query vector, shape= = (self.n_unlabeled, )
            q = np.dot(p, query)

            reward, ask_id, lbl = yield q
            self.update_experts(ask_id, lbl)
            ask_idx = self.invert_id_idx[ask_id]

            rhat = reward * query[:, ask_idx] / q[ask_idx]

            # The original advice vector in Exp4.P in ALBL is a identity matrix
            #yhat = np.dot(advice, rhat)
            yhat = rhat
            #vhat = np.sum(advice / np.tile(p, (self.N, 1)), axis=1)
            vhat = 1 / p
            self.w = self.w * np.exp(
                    self.pmin / 2 * (yhat + vhat*np.sqrt(
                        np.log(self.N/self.delta) / self.K / self.T)
                        )
                    )

            self.t += 1

        raise StopIteration


