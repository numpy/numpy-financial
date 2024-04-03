project(
    'numpy-financial',
    'cython', 'cpp',
    version: '2.0.0.dev',
    license: 'BSD-3',
    meson_version: '>=1.4.0',
)

cpp = meson.get_compiler('cpp')
cy = meson.get_compiler('cython')
# generator() doesn't accept compilers, only found programs - cast it.
cython = find_program(cy.cmd_array()[0])

version_link_args = ['-Wl']

if not cy.version().version_compare('>=3.0.9')
  error('NumPy-Financial requires Cython >= 3.0.9')
endif

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

subdir('numpy_financial')
