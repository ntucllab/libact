"""Active learning by learning (ALBL)

This module includes two classes. ActiveLearningByLearning is the main
algorithm for ALBL and Exp4P is the multi-armed bandit algorithm which will be
used in ALBL.

"""
from libact.base.interfaces import QueryStrategy
import numpy as np
import copy


class ActiveLearningByLearning(QueryStrategy):
    """Active Learning By Learning (ALBL) query strategy.

    ALBL is an active learning algorithm that adaptively choose among existing
    query strategies to decide which data to make query. It utilizes
    Exp4.P, a multi-armed bandit algorithm to adaptively make such decision.
    More details of ALBL can refer to the work listed in the reference section.

    Parameters
    ----------
    T: integer
        Query budget, the maximal number of queries to be made.
        
    query_strategies: list of libact.query_strategies.* object instance
        The active learning algorithms used in ALBL, which will be both the
        the arms in the multi-armed bandit algorithm Exp4.P.
        Note that these query_strategies should share the same dataset
        instance with ActiveLearningByLearning instance.

    delta: float, optional (default=0.1)
        Parameter for Exp4.P.

    uniform_sampler: {True, False}, optional (default=True)
        Determining whether to include uniform random sample as one of arms.

    pmin: float, 0<pmin<1/len(n_active_algorithm), optional (default=:math:`\frac{√{log(N)}{KT}`)
        Parameter for Exp4.P. The minimal probability for random selection of
        the arms (aka the underlying active learning algorithms). N = K =
        number of query_strategies, T is the number of query budgets.

    model: libact.model.* object instance
        The learning model used for the task.


    Attributes
    ----------
    query_strategies_: list of libact.query_strategies.* object instance

    exp4p_: instance of Exp4P object

    queried_hist_: list of integer
        A list of entry_id of the dataset which is queried in the past.


    Reference
    ---------

    .. [1] Wei-Ning Hsu, and Hsuan-Tien Lin. "Active Learning by Learning."
           Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.
    """

    def __init__(self, *args, **kwargs):
        super(ActiveLearningByLearning, self).__init__(*args, **kwargs)
        self.query_strategies_ = kwargs.pop('query_strategies', None)
        if self.query_strategies_ is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'query_strategies'"
                )
        elif not self.query_strategies_:
            raise ValueError("query_strategies list is empty")

        # check if query_strategies share the same dataset with albl
        for qs in self.query_strategies_:
            if qs.dataset != self.dataset:
                raise ValueError("query_strategies should share the same"
                                 "dataset instance with albl")

        # parameters for Exp4.p
        self.delta = kwargs.pop('delta', 0.1)

        # query budget
        self.T = kwargs.pop('T', None)
        if self.T is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'T'"
                )

        self.unlabeled_entry_ids, X_pool = \
            zip(*self.dataset.get_unlabeled_entries())
        self.unlabeled_invert_id_idx = {}
        for i, entry in enumerate(self.dataset.get_unlabeled_entries()):
            self.unlabeled_invert_id_idx[entry[0]] = i

        self.uniform_sampler = kwargs.pop('uniform_sampler', True)
        if type(self.uniform_sampler) != type(True):
            raise ValueError("'uniform_sampler' should be {True, False}")

        self.pmin = kwargs.pop('pmin', None)
        if self.pmin and (self.pmin < 1./(len(self.query_strategies_)+self.uniform_sampler) or self.pmin < 0):
            raise ValueError("'pmin' should be 0 < pmin < "
                             "1/len(n_active_algorithm)")

        self.exp4p_ = Exp4P(
            query_strategies=self.query_strategies_,
            T=self.T,
            delta=self.delta,
            pmin=self.pmin,
            unlabeled_invert_id_idx=self.unlabeled_invert_id_idx,
            uniform_sampler=self.uniform_sampler
        )
        self.budget_used = 0

        # classifier instance
        self.model = kwargs.pop('model', None)
        if self.model is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'model'"
                )

        self.query_dist = None

        self.W = []
        self.queried_hist_ = []

    def calc_reward_fn(self):
        """Calculate the reward value"""
        model = copy.copy(self.model)
        model.train(self.dataset)

        # reward function: Importance-Weighted-Accuracy (IW-ACC) (tau, f)
        reward = 0.
        for i in range(len(self.queried_hist_)):
            reward += self.W[i] *\
                (model.predict(self.dataset.data[self.queried_hist_[i]][0])[0] ==
                 self.dataset.data[self.queried_hist_[i]][1])
        reward /= (self.dataset.len_labeled() + self.dataset.len_unlabeled())
        reward /= self.T
        return reward

    def calc_query(self):
        """Calculate the sampling query distribution"""
        # initial query
        if self.query_dist == None:
            self.query_dist = self.exp4p_.next(-1, None, None)
        else:
            self.query_dist = self.exp4p_.next(
                self.calc_reward_fn(),
                self.queried_hist_[-1],
                self.dataset.data[self.queried_hist_[-1]][1]
            )
        return

    def update(self, entry_id, label):
        """Calculate the next query after updating the question asked with an
        answer."""
        ask_idx = self.unlabeled_invert_id_idx[entry_id]
        self.W.append(1./self.query_dist[ask_idx])
        self.queried_hist_.append(entry_id)

    def make_query(self):
        """Except for the initial query, it returns the id of the data albl
        wants to query."""
        dataset = self.dataset
        try:
            unlabeled_entry_ids, X_pool = zip(*dataset.get_unlabeled_entries())
        except ValueError:
            # might be no more unlabeled data left
            return

        while self.budget_used < self.T:
            self.calc_query()
            ask_idx = np.random.choice(
                        np.arange(len(self.unlabeled_invert_id_idx)),
                        size=1,
                        p=self.query_dist
                    )[0]
            ask_id = self.unlabeled_entry_ids[ask_idx]

            if ask_id in unlabeled_entry_ids:
                self.budget_used += 1
                return ask_id
            else:
                self.update(ask_id, dataset.data[ask_id][1])


        raise ValueError("Out of query budget")


class Exp4P():
    """A multi-armed bandit algorithm Exp4.P.

    For the Exp4.P used in ALBL, the number of arms (actions) and number of
    experts are equal to the number of active learning algorithms wanted to
    use. The arms (actions) are the active learning algorithms, where is
    inputed from parameter 'query_strategies'. There is no need for the input of
    experts, the advice of the kth expert are always equal e_k, where e_k is
    the kth column of the identity matrix.

    Parameters
    ----------
    query_strategies: QueryStrategy instances
        The active learning algorithms wanted to use, it is equivalent to
        actions or arms in original Exp4.P.

    unlabeled_invert_id_idx: dictionary
        A look up table for the correspondance of entry_id to the index of the
        unlabeled data.

    delta: float, >0, optional (default=0.1)
        A parameter.

    pmin: float, 0<pmin<1/len(query_strategies), optional (default=:math:`\frac{√{log(N)}{KT}`)
        The minimal probability for random selection of the arms (aka the
        unlabeled data), N = K = number of query_strategies, T is the maximum
        number of rounds.

    T: integer, optional (default=100)
        The maximum number of rounds.

    uniform_sampler: {True, False}, optional (default=Truee)
        Determining whether to include uniform random sampler as one of the
        underlying active learning algorithms.


    Attributes
    ----------
    t: integer
        The current round this instance is at.

    N: integer
        The number of arms (actions) in this exp4.p instance.

    quert_models_: list of libact.query_strategies.* object instance
        The underlying active learning algorithm instances.


    Reference
    ---------
    .. [1] Beygelzimer, Alina, et al. "Contextual bandit algorithms with
           supervised learning guarantees." In Proceedings on the International
           Conference on Artificial Intelligence and Statistics (AISTATS),
           2011u.
    """

    def __init__(self, *args, **kwargs):
        """ """
        # QueryStrategy class object instances
        self.query_strategies_ = kwargs.pop('query_strategies', None)
        if self.query_strategies_ is None:
            raise TypeError(
                "__init__() missing required keyword-only argument: 'query_strategies'"
                )
        elif not self.query_strategies_:
            raise ValueError("query_strategies list is empty")

        # whether to include uniform random sampler as one of underlying active
        # learning algorithms
        self.uniform_sampler = kwargs.pop('uniform_sampler', True)

        # n_armss
        if self.uniform_sampler:
            self.N = len(self.query_strategies_) + 1
        else:
            self.N = len(self.query_strategies_)

        # weight vector to each query_strategies, shape = (N, )
        self.w = np.array([1. for i in range(self.N)])

        # max iters
        self.T = kwargs.pop('T', 100)

        # delta > 0
        self.delta = kwargs.pop('delta', 0.1)

        # n_arms = n_models (n_query_algorithms) in ALBL
        self.K = self.N

        # p_min in [0, 1/n_arms]
        self.pmin = kwargs.pop('pmin', None)
        if self.pmin == None:
            self.pmin = np.sqrt(np.log(self.N) / self.K / self.T)

        self.exp4p_gen = self.exp4p()

        self.unlabeled_invert_id_idx = kwargs.pop('unlabeled_invert_id_idx')
        if not self.unlabeled_invert_id_idx:
            raise TypeError(
                "__init__() missing required keyword-only argument:"
                "'unlabeled_invert_id_idx'"
                )

    def __next__(self, reward, ask_id, lbl):
        """For Python3 compatibility of generator."""
        return self.next(reward, ask_id, lbl)

    def next(self, reward, ask_id, lbl):
        """Taking the label and the reward value of last question and returns
        the next question to ask."""
        # first run don't have reward, TODO exception on reward == -1 only once
        if reward == -1:
            return next(self.exp4p_gen)
        else:
        # TODO exception on reward in [0, 1]
            return self.exp4p_gen.send((reward, ask_id, lbl))


    def exp4p(self):
        """The generator which implements the main part of Exp4.P.

        Yields
        ------
        q: array-like, shape = [K]
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
        while True:
            #TODO probabilistic active learning algorithm
            # len(self.unlabeled_invert_id_idx) is the number of unlabeled data
            query = np.zeros((self.N, len(self.unlabeled_invert_id_idx)))
            if self.uniform_sampler:
                query[-1, :] = 1. / len(self.unlabeled_invert_id_idx)
            for i, model in enumerate(self.query_strategies_):
                query[i][self.unlabeled_invert_id_idx[model.make_query()]] = 1

            # choice vector, shape = (self.K, )
            W = np.sum(self.w)
            p = (1 - self.K * self.pmin) * self.w / W + self.pmin

            # query vector, shape= = (self.n_unlabeled, )
            q = np.dot(p, query)

            reward, ask_id, lbl = yield q
            ask_idx = self.unlabeled_invert_id_idx[ask_id]

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

        raise StopIteration
