# libact: Pool-based Active Learning in Python

authors: Yu-An Chung, Shao-Chuan Lee, Yao-Yuan Yang, Tung-En Wu, [Hsuan-Tien Lin](http://www.csie.ntu.edu.tw/~htlin)

[![Build Status](https://travis-ci.org/ntucllab/libact.svg)](https://travis-ci.org/ntucllab/libact)
[![Documentation Status](https://readthedocs.org/projects/libact/badge/?version=latest)](http://libact.readthedocs.org/en/latest/?badge=latest)

# Introduction

`libact` is a python package designed to make active learning easier for real-world users. The package not only implements several popular active learning strategies, but also features the [active learning by learning](http://www.csie.ntu.edu.tw/~htlin/paper/doc/aaai15albl.pdf) meta-strategy that allows the machine to automatically *learn* the best strategy on the fly. The package is designed for easy extension in terms of strategies, models and labelers. In particular, `libact` models can be easily obtained by interfacing with the models in [`scikit-learn`](http://scikit-learn.org/).

Comments on the package is welcomed at (temporarily) `htlin@csie.ntu.edu.tw`. If you find this package useful, please cite the original works (see Reference of each strategy) as well as (temporarily)

```
@Misc{libact,
  author =   {Yu-An Chung and Shao-Chuan Lee and Yao-Yuan Yang and Tung-En Wu and Hsuan-Tien Lin},
  title =    {Pool-based Active Learning in Python},
  howpublished = {\url{https://github.com/ntucllab/libact}},
  year = {2015}
}
```

# Basic Dependencies

Python3 dependencies
```
pip3 install -r requirements.txt
```

Debian (>= 7) / Ubuntu (>= 14.04)
```
sudo apt-get install build-essential gfortran libatlas-base-dev liblapacke-dev python3-dev
```

MacOS
```
brew tap homebrew/science
brew install openblas
```

# Installation

One of the query strategies (`hintsvm`) depends on the HintSVM package that requires special installation. If you are not using the strategy, please simply follow the section on general installation. Otherwise please follow the section on HintSVM.

## General Installation
After resolving the dependencies, it should be fairly simple to install the package in your home directory:

```
python setup.py install --user
```

To install for all users on Unix/Linux:
```
python setup.py build
sudo python setup.py install
```

## Special Installation for HintSVM

For HintSVM, you would have to install the [hintsvm package](https://github.com/ntucllab/hintsvm) first.

Before running, you need to make sure the path to the library and
python code of `hintsvm` are set in the environment variables:

    export LD_LIBRARY_PATH=/path/to/hintsvm:$LD_LIBRARY_PATH
    export PYTHONPATH=/path/to/hintsvm/python:$PYTHONPATH

# Usage

The main usage of `libact` is as follows:

```python
qs = UncertaintySampling(trn_ds, method='lc') # query strategy instance

ask_id = qs.make_query() # let the specified query strategy suggest a data to query
X, y = zip(*trn_ds.data)
lb = lbr.label(X[ask_id]) # query the label of unlabeled data from labeler instance
trn_ds.update(ask_id, lb) # update the dataset with newly queried data
```

Some examples are available under the `examples` directory. Before running, use
`examples/get_dataset.py` to retrieve the dataset used by the examples.

Available examples:

  - `examples/plot.py`: This example performs basic usage of libact. It splits
    a fully-labeled dataset and remove some label from dataset to simulate
    the pool-based active learning scenario. Each query of an unlabeled dataset is then equivalent to revealing one labeled example in the original data set.

# Acknowledgments

The authors thank Chih-Wei Chang and other members of the [Computational Learning Lab](https://learner.csie.ntu.edu.tw/) at National Taiwan University for valuable discussions and various contributions to making this package better.
