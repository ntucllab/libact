# libact

libact: Pool-based active learning in Python

[![Build Status](https://travis-ci.org/ntucllab/libact.svg)](https://travis-ci.org/ntucllab/libact)

# Introduction

`libact` is a package designed to make active learning easier for real-world users. The package not only implements several popular active learning strategies, but also features the [active learning by learning](www.csie.ntu.edu.tw/~htlin/paper/doc/aaai15albl.pdf) meta-strategy that allows the machine to automatically *learn* the best strategy on the fly. The package is designed for easy extension in terms of strategies, models and labelers. In particular, `libact` models can be easily wrapped from models in [`scikit-learn`](http://scikit-learn.org/).

# Dependencies

Python3 dependencies
```
pip3 install -r requirements.txt
```

Debian (>= 7) / Ubuntu (>= 14.04)
```
sudo apt-get install build-essential gfortran libatlas-base-dev liblapacke-dev
```

MacOS
```
brew tap homebrew/science
brew install openblas
```

# Installation

To install in your home directory:

```
python setup.py install --user
```

To install for all users on Unix/Linux:
```
python setup.py build
sudo python setup.py install
```

# Examples

Some examples are available under the `examples` directory. Before running, use
`examples/get_dataset.py` to retrieve the dataset used by the examples.

Available examples:

  - `examples/plot.py`: This example performs basic usage of libact. It splits
    an supervised learning dataset and remove some label from dataset to simulate
    an active learning scenario. Each query of an unlabeled dataset is simply putting
    the label back to dataset.

    The main libact usage part is below:
    ```python
    qs = UncertaintySampling(trn_ds, method='lc') # query strategy instance
    
    ask_id = qs.make_query() # let the specified query strategy suggest a data to query
    X, y = zip(*trn_ds.data)
    lb = lbr.label(X[ask_id]) # query the label of unlabeled data from labeler instance
    trn_ds.update(ask_id, lb) # update the dataset with newly queried data
    ```
  
  


## HintSVM

For HintSVM, you would have to install package from https://github.com/ntucllab/hintsvm

Before running, you need to make sure the path to hintsvm's library and
python code are set. Set them up by setting environment variables:

    export LD_LIBRARY_PATH=/path/to/hintsvm:$LD_LIBRARY_PATH
    export PYTHONPATH=/path/to/hintsvm/python:$PYTHONPATH
