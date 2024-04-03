project(
    'numpy-financial',
    'cython',
    version: '2.0.0.dev',
    license: 'BSD-3',
    meson_version: '>=1.4.0',
)

cy = meson.get_compiler('cython')

if not cy.version().version_compare('>=3.0.9')
  error('NumPy-Financial requires Cython >= 3.0.9')
endif

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

subdir('numpy_financial')
