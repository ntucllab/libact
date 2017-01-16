.. libact documentation master file, created by
   sphinx-quickstart on Sun Nov  1 23:21:58 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

libact: Pool-based Active Learning in Python
============================================

`libact` is a python package designed to make active learning easier for
real-world users. The package not only implements several popular active
learning strategies, but also features the `active learning by
learning <http://www.csie.ntu.edu.tw/~htlin/paper/doc/aaai15albl.pdf>`_
meta-strategy that allows the machine to automatically *learn* the best strategy
on the fly. The package is designed for easy extension in terms of strategies,
models and labelers. In particular, `libact` models can be easily obtained by
interfacing with the models in `scikit-learn <http://scikit-learn.org/>`_.

-----------------
Table of Contents
-----------------

.. toctree::
  :maxdepth: 2

  overview.rst
  examples/examples.rst
  active_learning_by_learning.rst
  cost_sensitive_multiclass.rst
  dev_with_libact.rst
  api_reference.rst

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
