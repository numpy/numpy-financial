====================
Building with poetry
====================

Installing poetry
=================

numpy-financial uses `poetry <https://python-poetry.org/>`__ to manage
dependencies, build wheels and sdists, and publish to PyPI this page documents
how to work with poetry.

To install poetry follow their `official guide <https://python-poetry.org/docs/#installing-with-the-official-installer>`__.
It is recommended to use the official installer for local development.

To check your installation try to check the version of poetry::

    poetry -V


Setting up a virtual environment using poetry
=============================================

Once poetry is installed it is time to set up the virtual environment. To do
this, run::

    poetry install


``poetry install`` looks for dependencies in the ``pyproject.toml`` file,
resolves them to the most recent version and installs the dependencies
in a virtual environment. It is now possible to launch an interactive REPL
by running the following command::

    poetry run python

Running the test suite
======================

``numpy-financial``` has an extensive test suite, which can be run with the
following command::

    poetry run pytest

Building distributions
======================

It is possible to manually build distributions for ``numpy-financial`` using
poetry. This is possible via the ``build`` command::

    poetry build

The ``build`` command creates a ``dist`` directory containing a wheel and sdist
file.

Publishing to PyPI
==================

poetry provides support to publish packages to PyPI. This is possible using
the ``publish`` command::

    poetry publish --build --username <your username> --password <your password>

Note that this builds the package before publishing it. You will need to 
provide the correct username and password for PyPI.
