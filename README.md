# libact

libact - a pool-based active learning package

# Dependencies

Python3 dependencies
```
pip3 install -r requirements.txt
```

Debian/Ubuntu
```
sudo apt-get install liblapacke-dev
```

MacOS
```
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
    trn_ds.update(ask_id, y_train[ask_id]) # update the dataset with newly queried data
    ```
  
  


## HintSVM

For HintSVM, you would have to install package from https://github.com/yangarbiter/hintsvm

Before running, you need to make sure the path to hintsvm's library and
python code are set. Set them up by setting environment variables:

    export LD_LIBRARY_PATH=/path/to/hintsvm:$LD_LIBRARY_PATH
    export PYTHONPATH=/path/to/hintsvm/python:$PYTHONPATH
