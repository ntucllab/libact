Overview
========

`libact` is a Python package designed to make `active learning
<https://en.wikipedia.org/wiki/Active_learning_(machine_learning)>`_ easier for real-world users. The package not only implements several popular active learning strategies, but also features the active-learning-by-learning meta-algorithm that assists the users to automatically select the best strategy
on the fly. Furthermore, the package provides a unified interface for implementing more strategies, models and application-specific labelers. The package is open-source along with issue trackers on github, and can be easily installed from Python Package Index repository.


Currently `libact` supports pool-based active learning problems, which consist
of a set of labeled examples, a set of unlabeled examples, a supervised learning model, and a labeling oracle. In each iteration of active learning, the algorithm (also called a query strategy) queries the oracle to label an unlabeled example. The model can then be improved by the newly-labeled example.
The goal is to use as few queries as possible for the model to achieve decent learning performance. Based on the components above,
we have designed the following four interfaces for `libact`.

Dataset
-------
A :py:class:`libact.base.dataset.Dataset` object stores the labeled set
and the unlabeled set. Each unlabeled or labeled example within a Dataset object is assigned with a unique identifier. After retrieving the label for an unlabeled example 
from the Labeler (the oracle to be discussed below), the update method is used to 
assign the label to the example, referenced by its identifier.

Internally, Dataset also maintains a callback queue. The on_update method can be
used to register callback functions, which will be called after each update to
the Dataset. The callback functions can be used for active learning algorithms that need to update their internal states after querying the oracle.

Labeler
-------
A :py:class:`libact.base.interfaces.Labeler` object plays the role of the oracle in
the given active learning problem. Its label method takes in an unlabeled example and returns the retrieved label.

QueryStrategy
-------------
A :py:class:`libact.base.interfaces.QueryStrategy` object implements an active learning algorithm.
Each QueryStrategy object is associated with a Dataset object. When a QueryStrategy object is initialized, it will automatically register its update
method as a callback function to the associated Dataset to be informed of any Dataset updates. The make_query method of a QueryStrategy object returns
the identifier of an unlabeled example that the object (active learning algorithm) wants to query.

Currently, the following active learning algorithms are supported:

- Binary Classification

  - Density Weighted Uncertainty Sampling (density_weighted_uncertainty_sampling.py)
  - Hinted Sampling with SVM (hintsvm.py)
  - Query By Committee (query_by_committee.py)
  - Querying Informative and Representative Examples (quire.py)
  - Random Sampling (random_sampling.py)
  - Uncertainty Sampling (uncertainty_sampling.py)
  - Variance Reduction (variance_reduction.py)

- Multi-class Classification

  - Active Learning with Cost Embedding (multiclass/active_learning_with_cost_embedding.py)
  - Hierarchical Sampling (multiclass/hierarchical_sampling.py)
  - Expected Error Reduction (multiclass/expected_error_reduction.py)
  - Uncertainty Sampling (uncertainty_sampling.py)

- Multi-label Classification

  - Adaptive Active Learning (multilabel/adaptive_active_learning.py)
  - Binary Minimization (multilabel/binary_minimization.py)
  - Maximal Loss Reduction with Maximal Confidence (multilabel/maximum_margin_reduction.py)
  - Multi-label Active Learning With Auxiliary Learner (multilabel/multilabel_with_auxiliary_learner.py)

Note that because of legacy reasons, Uncertainty Sampling can handle multi-class setting though it is not under the multiclass submodule.

Additionally, we supported the `Active Learning By Learning` meta-algorithm (active_learning_by_learning.py) for selecting active learning algorithms for binary classification on the fly.

Model
-----
A :py:class:`libact.base.interfaces.Model` object represents a supervised classifiation algorithm. It contains train and predict methods, just like the fit and predict methods of the classification algorithms in `scikit-learn <http://scikit-learn.org/>`_. Note that the train method of Model only takes the labeled examples within Dataset for learning.

A :py:class:`libact.base.interfaces.ContinuousModel` object represents an algorithm that supports continuous outputs during predictions, which includes an additional predict_real method.

Note that there is a :py:class:`libact.models.SklearnAdapter` which
takes a sklearn classifier instance and adaptes it to the libact Model
interface.

Example Usage
-------------
Here is an example usage of `libact`:

.. code-block:: python
   :linenos:

   # declare Dataset instance, X is the feature, y is the label (None if unlabeled)
   dataset = Dataset(X, y)
   query_strategy = QueryStrategy(dataset) # declare a QueryStrategy instance
   labler = Labeler() # declare Labeler instance
   model = Model() # declare model instance

   for _ in range(quota): # loop through the number of queries
       query_id = query_strategy.make_query() # let the specified QueryStrategy suggest a data to query
       lbl = labeler.label(dataset.data[query_id][0]) # query the label of the example at query_id
       dataset.update(query_id, lbl) # update the dataset with newly-labeled example
       model.train(dataset) # train model with newly-updated Dataset
