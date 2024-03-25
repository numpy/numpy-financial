# Editing another person's pull request

## Respect

Please be respectful of other's work.

## Expected setup

This guide expects that you have set up your git environment as is outlined in [getting_the_code](getting_the_code.md). 
In particular, it assumes that you have:

1. a remote called ``origin`` for your fork of NumPy-Financial
2. a remote called ``upstream`` for the original fork of NumPy-Financial

You can check this by running: 

```shell
git remote -v
```

Which should output lines similar to the below:

```
origin	https://github.com/<your_username>/numpy-financial.git (fetch)
origin	https://github.com/<your_username>/numpy-financial.git (push)
upstream	https://github.com/numpy/numpy-financial.git (fetch)
upstream	https://github.com/numpy/numpy-financial.git (push)
```

## Accessing the pull request

You will need to find the pull request ID from the pull request you are looking at. Then you can fetch the pull request and create a branch in the process by:

```shell
git fetch upstream pull/<ID>/head:<BRANCH_NAME>
```

Where:

* ``<ID>`` is the id that you found from the pull request
* ``<BRANCH_NAME>`` is the name that you would like to give to the branch once it is created.

Note that the branch name can be anything you want, however it has to be unique.

## Switching to the new branch

```shell
git switch <BRANCH_NAME>
```
