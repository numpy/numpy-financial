
# This script was used to generate the rst function stubs
# numpy_financial.fv.rst, numpy_financial.ipmt.rst, etc.
# It requires that numpy_financial is already installed,
# although it could just as easily use a hard-coded list
# of function names.

import numpy_financial as nf


def generate_rst_function_files():
    names = [name for name in dir(nf) if not name.startswith('_')]
    for name in names:
        with open('numpy_financial.' + name + '.rst', 'w') as f:
            f.write('''{name}
{underline}

.. currentmodule:: numpy_financial

.. toctree::

.. autofunction:: {name}
'''.format(name=name, underline='='*len(name)))


if __name__ == "__main__":
    generate_rst_function_files()
