=====================
How to run benchmarks
=====================

NumPy-Financial implements a benchmarking suite using
`air speed velocity (asv) <https://asv.readthedocs.io/en/latest/>`_. This document
provides a summary on how to perform some common tasks with asv.


Getting asv
===========

asv is installed by using the ``lint`` dependency group of poetry. Make sure that you
install the ``lint`` group before following the remaining steps.

.. code-block::
  :caption: Installing the lint dependency group

      poetry install --with lint


Running asv
###########

Since we are using poetry we need to preface the usual ``asv``  commands with
``poetry run``. For example, to run the bench we need to use to following invocation:

.. code-block::
    :caption: Running the benchmarks with poetry and asv

        poetry run asv run

This should run the benchmarking suite for NumPy-Financial. Note that the second
``run`` maybe be replaced by any of the usual asv
`commands <https://asv.readthedocs.io/en/latest/commands.html>`_.

Some useful asv commands
########################

run
***

asv `run <https://asv.readthedocs.io/en/latest/commands.html#asv-run>`_ allows the
user to run benchmarks defined in the benchmarks directory.

publish
*******

Collate all results into a website. This website will be written to the
``html_dir`` given in the ``asv.conf.json`` file, and may be served using any
static web server.

preview
*******

Preview the results using a local web server


check
*****

This imports and checks basic validity of the benchmark suite, but does not
run the benchmark target code
