plot.py
=======

.. role:: python_code(code)
   :language: python

This example shows the basic way to compare two active learning algorithm.
The script is located in :code:`/examples/plot.py`. Before running the script, you
need to download sample dataset by running :code:`/examples/get_dataset.py` and choose
the one you want in variable :python_code:`dataset_filepath`.

.. literalinclude:: ../../examples/plot.py
   :language: python
   :lines: 38

First, the data are splitted into training and testing set:

.. literalinclude:: ../../examples/plot.py
   :language: python
   :pyobject: split_train_test
   :linenos:

The main part that uses `libact` is in the :python_code:`run` function: 

.. literalinclude:: ../../examples/plot.py
   :language: python
   :pyobject: run
   :linenos:


Full source code:

.. literalinclude:: ../../examples/plot.py
   :language: python
   :linenos:
