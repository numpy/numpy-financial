
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

.. toctree::
   :hidden:
   :maxdepth: 4

   api
   dev/index


Index and Search
----------------

* :ref:`genindex`
* :ref:`search`
