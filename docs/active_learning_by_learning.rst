Active Learning By Learning
===========================
Currently, most pool-based active learning algorithms are designed based on
different human-designed philosophy, it is hard for user to decide which
algorithm to use with a given problem. `Active Learning By Learning` (ALBL)
algorithm is a meta active learn algorithm designed to solve this problem.
ALBL considers multiple existing active learning algorithms and adaptively
*learns* a querying strategy based on the performance of these algorithms.

ALBL's design is based on a well-known adaptive learning problem called
multi-armed bandit problem. In bandit problem, it is given :math:`K` bandit
machines and a budget of :math:`T` iterations. Each time a bandit machine is
pulled, the machine returns with a reward. Multi-armed bandit problem wants to
balance between exploring each bandit machine and exploit the information gotten
from previous exploration to receive good reward. It wants to maximize the total
rewards earned through a series of decisions.

In ALBL, it utilizes `Exp4.P` contextual bandit algorithm. The bandit machines
corresponds to each sub-active learning algorithm. It wants to balance between
each exploring a good active learning algorithm for this problem and exploiting
the active learning algorithm that already performs well to earn a good reward.
The reward function that ALBL adopt is the Importance-Weighted-Accuracy (IW-ACC)

.. math::

    IW-ACC(f, τ) = \frac{1}{nT} \sum^{τ}_{t=1} W_t[y_{i_t} = f(x_{i_t})]

:math:`f` is the current model learned from the labeled samples, :math:`τ` is
the number of queries ALBL have asked, :math:`n` is nmuber of samples (labeled +
unlabeled).
