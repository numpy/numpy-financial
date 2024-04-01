# Building with poetry

## Installing poetry
numpy-financial uses [spin](https://github.com/scientific-python/spin) and conda 
to manage dependencies, build wheels and sdists, and publish to PyPI this page 
documents how to work with spin and conda.

To install poetry follow their [official guide](https://docs.anaconda.com/free/miniconda/miniconda-install/)
it is recommended to use the official installer for local development.

To check your installation try to check the version of miniconda:

```shell
conda -V
```

## Setting up a virtual environment using poetry

Once conda is installed it is time to set up the virtual environment. To do
this run:

```shell
conda env create -f environment.yml
```

This command looks for dependencies in the ``environment.yml`` file,
resolves them to the most recent version and installs the dependencies
in a virtual environment. It is now possible to launch an interactive REPL
by running the following command:

```shell
conda activate numpy-financial-dev
```

## Building NumPy-Financial

NumPy-Financial is built using a combination of Python and Cython. We therefore
require a build step. This can be run using

```shell
spin build
```

## Running the test suite

NumPy-Financial has an extensive test suite, which can be run with the
following command:

```shell
spin test
```

## Building distributions

It is possible to manually build distributions for numpy-financial using
poetry. This is possible via the `build` command:

```shell
spin build
```

The `build` command creates a `dist` directory containing a wheel and sdist
file.
