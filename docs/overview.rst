Overview
========

`libact` is a Python package designed to make `active learning
<https://en.wikipedia.org/wiki/Active_learning_(machine_learning)>`_ easier for real-world users. The package not only implements several popular active learning strategies, but also features the active-learning-by-learning meta-algorithm that assists the users to automatically select the best strategy
on the fly. Furthermore, the package provides a unified interface for implementing more strategies, models and application-specific labelers. The package is open-source along with issue trackers on github, and can be easily installed from Python Package Index repository.


Currently `libact` supports pool-based active learning problems, which consist
of a set of labeled examples, a set of unlabeled examples, a supervised learning model, and an labeling oracle. In each iteration of active learning, the algorithm (also called a query strategy) queries the oracle to label an unlabeled example. The model can then be improved by the newly-labeled example.
The goal is to use as few queries as possible for the model to achieve decent learning performance. Based on the components above,
we designed the following four interfaces for `libact`.

Dataset
-------
A :py:class:`libact.base.dataset.Dataset` object stores the labeled set
and unlabeled set. Each unlabeled or labeled example within a Dataset object is assigned with a unique identifier. After retrieving the label for an unlabeled example 
from the Labeler (the oracle to be discussed below), the update method is used to 
assign the label to the example, referenced by its identifier.

Internally, Dataset also maintains a callback queue. The method on_update can be
used to register callback functions, which will be called after each update to
the Dataset. The callback functions can be used for active learning algorithms that need to update their internal states after querying the oracle.

Labeler
-------
A :py:class:`libact.base.interfaces.Labeler` object plays the role of the oracle in
the given active learning problem. Its label method takes in an unlabeled example and returns the retrieved label.

QueryStrategy
-------------
A :py:class:`libact.base.interfaces.QueryStrategy` object represents an implementation active learning algorithm.
Each QueryStrategy object is associated with a Dataset object. When a QueryStrategy object is initialized, it will automatically register its update
method as a callback function to the associated Dataset to be informed of any Dataset updates. The make_query method of a QueryStrategy object returns
the identifier of an unlabeled example that the object (active learning algorithm) wants to query.

Currently supported algorithms includes the following:

+--------------------------------------------------+---------------------------------------+---------------------------------------------------+
| Binary-class                                     | Multi-class                           | Multi-label                                       |
+==================================================+=======================================+===================================================+
| Density Weighted Uncertainty Sampling            | Uncertainty Sampling (entropy)        | Binary Minimization                               |
+--------------------------------------------------+---------------------------------------+---------------------------------------------------+
| Query By Committee (vote entropy)                | Uncertainty Sampling (largest margin) | Maximal Loss Reduction with Maximal Confidence    |
+--------------------------------------------------+---------------------------------------+---------------------------------------------------+
| Query By Committee (KL-divergence)               | Uncertainty Sampling (least confident)| Multi-label Active Learning With Auxiliary Learner|
+--------------------------------------------------+---------------------------------------+---------------------------------------------------+
| Variance Reduction                               | Active Learning With Cost Embedding   | Adaptive Active Learning                          |
+--------------------------------------------------+---------------------------------------+---------------------------------------------------+
| HintSVM                                          |                                       |                                                   |
+--------------------------------------------------+---------------------------------------+---------------------------------------------------+
| Querying Informative and Representative Examples |                                       |                                                   |
+--------------------------------------------------+---------------------------------------+---------------------------------------------------+
| HintSVM                                          |                                       |                                                   |
+--------------------------------------------------+---------------------------------------+---------------------------------------------------+
| Random Sampling                                  |                                       |                                                   |
+--------------------------------------------------+---------------------------------------+---------------------------------------------------+

Note that Uncertainty Sampling can handle multi-class setting though it is not
under the multiclass submodule.

Additionally, we supported the `Active Learning By Learning` meta algorithm for
binary class active learning algorithm selection.

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
       model.train(dataset) #train model with newly-updated Dataset
