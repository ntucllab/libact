Develop with Libact
===================

To develope active learning usage under `libact` framwork, you may implement
your own oracle, active learning algorithm and machine learning algorithms.

Write your own models
---------------------
To implement your own models, your model class should inherent from either
:py:class:`libact.base.interfaces.Model` or
:py:class:`libact.base.interfaces.ContinuousModel`. For regular model, there are
three methods to be implmented: :py:func:`train`, :py:func:`predict`, and
:py:func:`score`. For learning models that supports continuous output, method
:py:func:`predict_real` should be implemented for :code:`ContinuousModel`.

train
^^^^^
Method :code:`train` takes in a :code:`Dataset` object, which may include both
labeled and unlabeled data. With supervised learning models, labeled data can be
retrieved like this:

.. code-block:: python

   X, y = zip(*Dataset.get_labeled_entries())

:code:`X`, :code:`y` is the samples (shape=(n_samples, n_feature)) and labels
(shape=(n_samples)).

You should train your model in this method like the :code:`fit` method in
`scikit-learn` model.

predict
^^^^^^^
This method should work like the :code:`predict` method in `scikit-learn` model.
Takes in the feature of each sample and output the label of the prediction for
these samples.

score
^^^^^
This method should calculate the accuracy on a given dataset's labeled data.

predict_real
^^^^^^^^^^^^
For models that can generate continuous predictions (for example, the distance
to boundary).

Examples
^^^^^^^^
Take a look at :py:class:`libact.models.svm.SVM`, it serves as an interface of
scikit-learn's SVC model. The train method is connected to scikit-learn's fit
method and predict is connected to scikit-learn's predict. For the predict_real
method, it represens the decision value to each label.

.. literalinclude:: ../libact/models/svm.py
   :language: python
   :pyobject: SVM


Implement your active learning algorithm
----------------------------------------
You may implement your own active learning algorithm under QueryStrategy
classes. QueryStrategy class should inherent from
:py:class:`libact.base.interfaces.QueryStrategy` and add the following into your
__init__ method.

.. code-block:: python

   super(YourClassName, self).__init__(*args, **kwargs)

This would associate the given dataset with your query strategy and registers
the update method under the associated dataset as a callback function.

The :py:func:`update` method should be used if the active learning algorithm
wants to change its internal state after the dataset is updated with newly
retrieved label. Take ALBL's :py:func:`update` method as example:

.. literalinclude:: ../libact/query_strategies/active_learning_by_learning.py
   :language: python
   :pyobject: ActiveLearningByLearning.update

:py:func:`make_query` is another method need to be implmented. It calculates
which sample to query and outputs the entry id of that sample. Take the
uncertainty sampling algorithm as example:

.. literalinclude:: ../libact/query_strategies/uncertainty_sampling.py
   :language: python
   :pyobject: UncertaintySampling.make_query

In uncertainty sampling, it asks the sample with the lowest decision value (the
output from :py:func:`predict_real` of a :py:class:`ContinuousModel`).


Write your Oracle
-----------------
Different usage requires different ways of retrieving the label for an unlabeled
sameple, therefore you may want to implement your own oracle for different
condition To implement Labeler class you should inherent from
:py:class:`libact.base.interfaces.Labeler` and implment the :py:func:`label`
function with how to retrieve the label of a given sample (feature).

Examples
^^^^^^^^
We have provided two example labelers:
:py:class:`libact.labelers.IdealLabeler` and
:py:class:`libact.labelers.InteractiveLabeler`.

:py:class:`IdealLabeler` is usually used for testing the performance of a active
learning algorithm. You give it a fully-labeled dataset, simulating a oracle
that know the true label of all samples. Its :py:func:`label` is simple
searching through the given feature in the fully-labeled dataset and return the
corresponding label.

.. literalinclude:: ../libact/labelers/ideal_labeler.py
   :language: python
   :pyobject: IdealLabeler

:py:class:`InteractiveLabeler` can be used in the situation where you want to
show your feature through image, let a human be the oracle and label the image
interactively. To implement its :py:func:`label` method, it may include showing
the feature through image using :py:func:`matplotlib.pyplot.imshow` and receive
input through command line interface:

.. literalinclude:: ../libact/labelers/interactive_labeler.py
   :language: python
   :pyobject: InteractiveLabeler
