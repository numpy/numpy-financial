# Building the docs

This guide goes through how to build the NumPy-Financial documentation with spin and conda

## Assumptions

This guide assumes that you have set up poetry and a virtual environment. If you have 
not done this please read [building_with_spin](building_with_spin).

You can check that conda and spin are installed by running:

```shell
conda -V
```

```shell
spin -V
```

## Building the documentation

spin handles building the documentation for us. All we have to do is invoke the built-in command.

```shell
spin docs -j 1
```

This will create the docs as a html document in the ``doc/build`` directory. Note that there are several options 
available, however, only the html documentation is built officially.
  