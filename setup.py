#!/usr/bin/env python

from os import path


def get_version():
    """
    Find the value assigned to __version__ in numpy_financial/__init__.py.

    This function assumes that there is a line of the form

        __version__ = "version-string"

    in that file.  It returns the string version-string, or None if such a
    line is not found.
    """
    with open("numpy_financial/__init__.py", "r") as f:
        for line in f:
            s = [w.strip() for w in line.split("=", 1)]
            if len(s) == 2 and s[0] == "__version__":
                return s[1][1:-1]


CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Developers
Intended Audience :: Financial and Insurance Industry
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3 :: Only
Topic :: Software Development
Topic :: Office/Business :: Financial :: Accounting
Topic :: Office/Business :: Financial :: Investment
Topic :: Office/Business :: Financial :: Spreadsheet
"""

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), 'r') as f:
    long_description = f.read()

if __name__ == "__main__":
    from setuptools import setup, find_packages

    setup(name='numpy-financial',
          version=get_version(),
          description='Simple financial functions',
          long_description=long_description,
          long_description_content_type='text/markdown',
          packages=find_packages(exclude=['doc']),
          author='Travis E. Oliphant et al.',
          license='BSD 3-Clause',
          maintainer='Numpy Financial Developers',
          maintainer_email='numpy-discussion@python.org',
          install_requires=['numpy>=1.15'],
          python_requires='>=3.5',
          classifiers=CLASSIFIERS.splitlines(),
          url='https://numpy.org/numpy-financial/',
          download_url='https://pypi.org/project/numpy-financial/',
          project_urls={
              "Bug Tracker": "https://github.com/numpy/numpy-financial/issues",
              "Documentation": "https://numpy.org/numpy-financial/",
              "Source Code": "https://github.com/numpy/numpy-financial",
          })
