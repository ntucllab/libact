# libact

libact - a pool-based active learning package

# Examples

Some examples are available under the `examples` directory. Before running, use
`examples/get_dataset.py` to retrieve the dataset used by the examples.

Available examples:

  - `examples/plot.py`: see example usage below
  
```
$ examples/plot.py -m LogisticRegression -q UncertaintySampling --qs-params='{"method": "lc"}'
$ examples/plot.py -m LogisticRegression -q QueryByCommittee --qs-params='{"models": ["Perceptron", "Perceptron", "Perceptron"]}'
```
