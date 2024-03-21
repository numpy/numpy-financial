
numpy-financial |version|
=========================

The `numpy-financial` package is a collection of elementary financial
functions.

The `financial functions in NumPy <https://numpy.org/doc/1.17/reference/routines.financial.html>`_
are deprecated and eventually will be removed from NumPy; see
`NEP-32 <https://numpy.org/neps/nep-0032-remove-financial-functions.html>`_
for more information.  This package is the replacement for the deprecated
NumPy financial functions.

The source code for this package is available at https://github.com/numpy/numpy-financial.

The importable name of the package is `numpy_financial`.  The recommended
alias is `npf`.  For example,

>>> import numpy_financial as npf
>>> npf.irr([-250000, 100000, 150000, 200000, 250000, 300000])
0.5672303344358536


Functions
---------

.. currentmodule:: numpy_financial

.. toctree::
   :maxdepth: 4

.. autosummary::

   fv
   ipmt
   irr
   mirr
   nper
   npv
   pmt
   ppmt
   pv
   rate

.. The following "hidden" toctree is a hack to prevent Sphinx warnings
   about "document isn't included in any toctree"

.. toctree::
   :hidden:

   fv
   ipmt
   irr
   mirr
   nper
   npv
   pmt
   ppmt
   pv
   rate

Development
===========

.. toctree::
   :maxdepth: 1

   dev/getting_the_code.md
   dev/building_with_poetry.md
   dev/running_the_benchmarks.md

.. include:: release-notes.rst


Index and Search
================

* :ref:`genindex`
* :ref:`search`
