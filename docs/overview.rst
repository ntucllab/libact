Overview
========

`libact` designed a framework to make active learning easy for user to apply to
real world problem. Currently `libact` supports only pool-based active learning
problems. In the start of the problem, there is a small labeled set, a larger
unlabeled set, and a training model. During the problem, 
active learning algorithm has to choose a data point from the unlabeled set
and ask the oracle for its label. The goal for active learning algorithm is to
make the trainign model performs better with less labeled data.

`libact` is consitituded by the following four sub-modules:

base
^^^^

query_strategies
^^^^^^^^^^^^^^^^
query_strategies consists different pool-based active learning algorithms.


labelers
^^^^^^^^


models
^^^^^^

