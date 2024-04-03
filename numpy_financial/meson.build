cython_cpp_args = ['-DCYTHON_CCOMPLEX=0'] # see scipy-18975 for why we need this

_solve_lib = static_library(
    '_solve',
    ['src/_solve.h', 'src/_solve.cpp'],
)

_solve_dep = declare_dependency(
    link_with: _solve_lib
)

_inner_loop_lib = static_library(
    '_inner_loop',
    ['src/_inner_loop.h', 'src/_inner_loop.cpp'],
)

_inner_loop_dep = declare_dependency(
    link_with: _inner_loop_lib
)

cython_args = ['-3', '--fast-fail', '--output-file', '@OUTPUT@', '--include-dir', '@BUILD_ROOT@', '@INPUT@']
cython_cplus_args = ['--cplus'] + cython_args

cython_gen_cpp = generator(
  cython,
  arguments : cython_cplus_args,
  output : '@BASENAME@.cpp',
)


py.extension_module(
    '_cfinancial',
    cython_gen_cpp.process('_cfinancial.pyx'),
    cpp_args: cython_cpp_args,
    install: true,
    subdir: 'numpy_financial',
    dependencies: [_solve_dep, _inner_loop_dep],
    include_directories: 'src/',
)


python_sources = [
    '__init__.py',
    '_financial.py',
]

py.install_sources(
    python_sources,
    subdir: 'numpy_financial',
)

install_subdir('tests', install_dir: py.get_install_dir() / 'numpy_financial')
