Active Learning By Learning
===========================
Currently, most pool-based active learning algorithms are designed based on
different human-designed philosophy, it is hard for user to decide which
algorithm to use with a given problem. `Active Learning By Learning` (ALBL)
algorithm is a meta active learn algorithm designed to solve this problem.
ALBL considers multiple existing active learning algorithms and adaptively
*learns* a querying strategy based on the performance of these algorithms.

ALBL's design is based on a well-known adaptive learning problem called
multi-armed bandit. In the problem, :math:`K` bandit
machines and a budget of :math:`T` iterations are given.
Each time a bandit machine is pulled, the machine returns a reward that reflects
the goodness of the machine. The multi-armed bandit problem aims at balancing
between exploring each bandit machine and exploit the observed information in
order to maximize the cumulative rewards after a seris of pulling decisions.
The details can be found in the paper

Wei-Ning Hsu, and Hsuan-Tien Lin. "Active Learning by Learning." Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.

Here is an example of how to declare a ALBL query_strategy object:

.. code-block:: python
   :linenos:

   from libact.query_strategies import ActiveLearningByLearning
   from libact.query_strategies import HintSVM
   from libact.query_strategies import UncertaintySampling
   from libact.models import LogisticRegression

   qs = ActiveLearningByLearning(
            dataset, # Dataset object
            T=100, # qs.make_query() can be called for at most 100 times
            query_strategies=[
                UncertaintySampling(dataset, model=LogisticRegression(C=1.)),
                UncertaintySampling(dataset, model=LogisticRegression(C=.01)),
                HintSVM(dataset)
                ],
            model=LogisticRegression()
            )

The :code:`T` parameter provides the query budget for ALBL, which is the number
of times you may ask the query_strategy (ALBL) to make a query.
The :code:`query_strategies` parameter is a list of
:code:`libact.query_strategies` object instances where each of their associated
dataset must be the same :code:`Dataset` instance.
ALBL combines the result of these query strategies and generate its own
suggestion of which sample to query.
ALBL will adaptively *learn* from each of the decision it made, using the given
supervised learning model in :code:`model` parameter.
