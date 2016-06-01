Overview
========

`libact` is a Python package designed to make `active learning
<https://en.wikipedia.org/wiki/Active_learning_(machine_learning)>`_ easier for real-world users. The package not only implements several popular active learning strategies, but also features the active-learning-by-learning meta-algorithm that assists the users to automatically select the best strategy
on the fly. Furthermore, the package provides a unified interface for implementing more strategies, models and application-specific labelers. The package is open-source along with issue trackers on github, and can be easily installed from Python Package Index repository.


Currently `libact` supports pool-based active learning problems, which consists
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
:py:class:`libact.base.interfaces.Labeler` object plays the role as an oracle in
the given problem. After retrieveing the sample to be queried, pass the samepl
(feature) to the label method, it will return the label from oracle.

QueryStrategy
-------------
:py:class:`libact.base.interfaces.QueryStrategy` objects are the
implementation of active learning algorithms.  Each QueryStrategy object is
associated with a Dataset object, when Dataset object gets its update with a
label for unlabeled data, it will trigger the update method under QueryStrategy
object. The update method can be used to update the internal states in
QueryStrategy. There is also another method under QueryStrategy called
make_query. This method returns the unlabeled sample's index with this active
learning algorithm wants to query.

Model
-----
:py:class:`libact.base.interfaces.Model` objects are the implementation of
classification algorithms. It has method train and predict just like the
classification algorithms in `scikit-learn <http://scikit-learn.org/>`_ has fit
and predict. The only difference is that the train method takes in an Dataset
instance, which will train the model with only the labeled samples.

:py:class:`libact.base.interfaces.ContinuousModel` are the classification
algorithms that supports continuous predictions, which has the predict_real
method.

Example Usage
-------------
Here is an example usage of `libact`:

.. code-block:: python
   :linenos:

   # declare Dataset instance, X is the feature, y is the label (None if unlabeled)
   ds = Dataset(X, y)
   qs = QueryStrategy(trn_ds) # declare a QueryStrategy instance
   lbr = Labeler() # declare Labeler instance
   model = Model() # declare model instance

   for i in range(quota): # loop through the number of chances to ask oracle for label
       ask_id = qs.make_query() # let the specified QueryStrategy suggest a data to query
       X, _ = zip(*ds.data) # retrieve feature from Dataset
       lb = lbr.label(X[ask_id]) # query the label of unlabeled sample from labeler instance
       ds.update(ask_id, lb) # update the dataset with newly queried sample
       model.train(ds) # train model with newly updated Dataset

