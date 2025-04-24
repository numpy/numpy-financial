py.extension_module(
    '_cfinancial',
    '_cfinancial.pyx',
    install: true,
    subdir: 'numpy_financial',
)

python_sources = [
    '__init__.py',
    '_financial.py',
    '_cfinancial.pyi',
    'py.typed',
]

py.install_sources(
    python_sources,
    subdir: 'numpy_financial',
)

install_subdir('tests', install_dir: py.get_install_dir() / 'numpy_financial')
