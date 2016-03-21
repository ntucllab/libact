Overview
========

`libact` is a framework to make `active learning
<https://en.wikipedia.org/wiki/Active_learning_(machine_learning)>`_ easy for
user to apply to real world problem. Currently `libact` supports only pool-based
active learning problems. For a active learning problem, there is a labeled set,
unlabeled set, oracle and a supervised learning model. During the querying
state, active learning algorithm will choose a data point from the unlabeled set
and ask the oracle for its label. The goal for active learning algorithm is to
make the given supervied learning model performs better with less labeled data.

`libact` is consitituded of the following parts:

Dataset
-------
:py:class:`libact.base.dataset.Dataset` object stores the labeled set
and unlabeled set. When Dataset is created it assigned a index to each sample.
Both unlabeled samples and labeled samples shares a same set of indexs. When a
label is retrieved from Labeler (oracle), it uses the update method to assign
the label to a unlabeled sample. The sample index will be used to identify which
sample to update.

Internally, Dataset also maintains a callback queue. The method on_update can be
used to register callback functions, which will be called after each update to
the Dataset.

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

