# libact

libact - a pool-based active learning package

[![build status](https://gitlab.lv6.tw/ci/projects/1/status.png?ref=master)](https://gitlab.lv6.tw/ci/projects/1?ref=master)

# Dependencies

Python3 dependencies
```
pip3 install numpy scikit-learn
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

  - `examples/plot.py`: see example usage below
  
```
$ examples/plot.py -m LogisticRegression -q UncertaintySampling --qs-params='{"method": "lc"}'
$ examples/plot.py -m LogisticRegression -q QueryByCommittee --qs-params='{"models": ["Perceptron", "Perceptron", "Perceptron"]}'
```

## HintSVM

For HintSVM, you would have to install package from https://github.com/yangarbiter/hintsvm

Before running, you need to make sure the path to hintsvm's library and
python code are set. (set them up by setting the environment variable
LD_LIBRARY_PATH=PATH_TO_HINTSVM PYTHONPATH=PATH_TO_HINTSVM/python/)
