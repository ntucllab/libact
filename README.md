# libact: Pool-based Active Learning in Python

authors: [Yao-Yuan Yang](http://yyyang.me), Shao-Chuan Lee, Yu-An Chung, Tung-En Wu, Si-An Chen, [Hsuan-Tien Lin](http://www.csie.ntu.edu.tw/~htlin)

[![Build Status](https://github.com/ntucllab/libact/actions/workflows/tests.yml/badge.svg)](https://github.com/ntucllab/libact/actions/workflows/tests.yml)
[![Documentation Status](https://readthedocs.org/projects/libact/badge/?version=latest)](http://libact.readthedocs.org/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/libact.svg)](https://badge.fury.io/py/libact)
[![codecov.io](https://codecov.io/github/ntucllab/libact/coverage.svg?branch=master)](https://codecov.io/github/ntucllab/libact?branch=master)

## Introduction

`libact` is a Python package designed to make active learning easier for
real-world users. The package not only implements several popular active learning strategies, but also features the [active-learning-by-learning](http://www.csie.ntu.edu.tw/~htlin/paper/doc/aaai15albl.pdf)
meta-algorithm that assists the users to automatically select the best strategy
on the fly. Furthermore, the package provides a unified interface for implementing more strategies, models and application-specific labelers. The package is open-source along with issue trackers on github, and can be easily installed from Python Package Index repository.

## Documentation

The technical report associated with the package is on [arXiv](https://arxiv.org/abs/1710.00379), and the documentation for the latest release is available on [readthedocs](http://libact.readthedocs.org/en/latest/).
Comments and questions on the package is welcomed at `libact-users@googlegroups.com`. All contributions to the documentation are greatly appreciated!

## Basic Dependencies

- Python 3.9, 3.10, 3.11, 3.12

- Python dependencies (automatically installed with pip):
  - numpy >= 2
  - scipy >= 1.13
  - scikit-learn >= 1.6
  - matplotlib >= 3.8
  - joblib

### BLAS/LAPACKE Dependencies

- Debian (>= 7) / Ubuntu (>= 14.04)

```
sudo apt-get install build-essential gfortran libatlas-base-dev liblapacke-dev python3-dev
```

- Arch Linux

```
sudo pacman -S lapacke
```

- macOS
```
brew install openblas
```

- Others: refer to the BLAS/LAPACKE installation guides.

## Installation

- Install the official release (from PyPI):

```shell
pip install libact
```

> **Note:** For Windows users, it is recommended to use **Windows Subsystem for Linux (WSL)** as the primary environment for installing and running `libact`.

- Install the latest development version

```shell
pip install git+https://github.com/ntucllab/libact.git
```

## Build Options

This package supports the following build options:

- `blas`: BLAS library to use (default='auto'). Options: `auto`, `openblas`, `Accelerate`, `mkl`, `lapack`, `blis`.
- `lapack`: LAPACK library to use (default=`auto`). Options: `auto`, `openblas`, `Accelerate`, `mkl`, `lapack`, `blis`.
- `variance_reduction`: Build variance reduction module (default: true)
- `hintsvm`: Build hintsvm module (default: true)

### Examples

To install `libact` with the default configuration, run:

```shell
pip install libact
```

Install without optional modules:

```shell
pip install libact --config-settings=setup-args="-Dvariance_reduction=false" \
                    --config-settings=setup-args="-Dhintsvm=false"
```

## Build from Source

### Overview

This project utilizes `meson` and `meson-python` as the build backend. To build from source, ensure you have the aforementioned dependencies installed on your system. The building procedure additionally requires the following dependencies:

- `meson-python`
- `ninja`
- `cython`
- `numpy`

### The Recommended Approach (Using Bootstrapped Environment Config)

To simplify the environment setup, we provide a pre-configured `environment.yml` located at the root directory of the project. Install with `conda/mamba` to get a head start.

```shell
# Clone the repository
git clone https://github.com/ntucllab/libact.git
cd libact

# Create and activate conda environment
conda env create -f environment.yml
conda activate libact

# Install in development mode
pip install --no-build-isolation -e .

# Or build and install
pip install --no-build-isolation .
```

### Regular Install (Recommended for Users)

For regular usage (not development), simply install from PyPI or from a local clone:

```shell
# From PyPI
pip install libact

# From local clone
pip install .
```

Regular installs do **not** require build tools at runtime and will work without any additional dependencies.

### Editable/Development Install (Recommended Method)

Editable installs with meson-python automatically rebuild compiled components when you import the package. To ensure build tools are available, use `--no-build-isolation`:

```shell
# First install build dependencies in your environment
pip install meson-python meson ninja cython numpy

# Then install in editable mode without build isolation
pip install --no-build-isolation -e .
```

This ensures that `ninja`, `meson`, and other build tools remain available in your environment for rebuilds.

**Troubleshooting:** If you get errors about missing `ninja` or build tools when importing libact:
- You may have installed in editable mode with build isolation (which is not recommended)
- Solution: Reinstall using the method above, OR use a regular install: `pip install .`

## Usage

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

  - [plot](examples/plot.py) : This example performs basic usage of libact. It splits
    a fully-labeled dataset and remove some label from dataset to simulate
    the pool-based active learning scenario. Each query of an unlabeled dataset is then equivalent to revealing one labeled example in the original data set.
  - [label_digits](examples/label_digits.py) : This example shows how to use libact in the case
    that you want a human to label the selected sample for your algorithm.
  - [albl_plot](examples/albl_plot.py): This example compares the performance of ALBL
    with other active learning algorithms.
  - [multilabel_plot](examples/multilabel_plot.py): This example compares the performance of
    algorithms under multilabel setting.
  - [alce_plot](examples/alce_plot.py): This example compares the performance of
    algorithms under cost-sensitive multi-class setting.

## Running tests

To run the test suite:

```
python -m unittest -v
```

To run pylint, install pylint through ```pip install pylint``` and run the following command in root directory:

```
pylint libact
```

To measure the test code coverage, install coverage through ```pip install coverage``` and run the following commands in root directory:

```
python -m coverage run --source libact --omit */tests/* -m unittest
python -m coverage report
```

## Citing

If you find this package useful, please cite the original works (see Reference of each strategy) as well as the following

```
@techreport{YY2017,
  author = {Yao-Yuan Yang and Shao-Chuan Lee and Yu-An Chung and Tung-En Wu and Si-An Chen and Hsuan-Tien Lin},
  title = {libact: Pool-based Active Learning in Python},
  institution = {National Taiwan University},
  url = {https://github.com/ntucllab/libact},
  note = {available as arXiv preprint \url{https://arxiv.org/abs/1710.00379}},
  month = oct,
  year = 2017
}
```

## Acknowledgments

The authors thank Chih-Wei Chang and other members of the [Computational Learning Lab](https://learner.csie.ntu.edu.tw/) at National Taiwan University for valuable discussions and various contributions to making this package better.


