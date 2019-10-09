.. numpy-financial documentation master file, created by
   sphinx-quickstart on Tue Oct  1 23:15:00 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

numpy-financial |version|
=========================

The `numpy-financial` package contains a collection of elementary financial
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
   :caption: Contents:

.. autosummary::
   :toctree: .

   numpy_financial.fv
   numpy_financial.ipmt
   numpy_financial.irr
   numpy_financial.mirr
   numpy_financial.nper
   numpy_financial.npv
   numpy_financial.pmt
   numpy_financial.ppmt
   numpy_financial.pv
   numpy_financial.rate


Index and Search
================

* :ref:`genindex`
* :ref:`search`
