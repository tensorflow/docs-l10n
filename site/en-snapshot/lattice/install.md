# Install TensorFlow Lattice

There are several ways to set up your environment to use TensorFlow Lattice
(TFL).

*   The easiest way to learn and use TFL requires no installation: run the any
    of the tutorials (e.g.
    [canned estimators tutorial](tutorials/canned_estimators.ipynb)).
*   To use TFL on a local machine, install the `tensorflow-lattice` pip package.
*   If you have a unique machine configuration, you can build the package from
    source.

## Install TensorFlow Lattice using pip

Install using pip.

```shell
pip install --upgrade tensorflow-lattice
```

## Build from source

Clone the github repo:

```shell
git clone https://github.com/tensorflow/lattice.git
```

Build pip package from source:

```shell
python setup.py sdist bdist_wheel --universal --release
```

Install the package:

```shell
pip install --user --upgrade /path/to/pkg.whl
```
