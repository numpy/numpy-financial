# Getting the code

This document explains how to get the source code for NumPy-Financial using Git.

## Code Location

NumPy-Financial is hosted on GitHub. The repository URL is https://github.com/numpy/numpy-financial.

## Creating a fork

To create a fork click the green "fork" button at the top right of the page. This creates a new repository on your
GitHub profile. This will have the URL: https://github.com/<your_username>/numpy-financial.

## Cloning the repository

Now that you have forked the repository you will need to clone it. This copies the repository from GitHub to the local
machine. To clone a repository enter the following commands in the terminal.

```shell
git clone https://github.com/<your_username>/numpy-financial.git
```

Hooray! You now have a working copy of NumPy-Financial.


## Updating the code with other's changes

From time to time you may want to pull down the latest code. Do this with:

```shell
git fetch
```

The `git fetch` command downloads commits, files and refs from a remote repo into your local repo.

Then run:

```shell
git merge --ff-only
```

The `git merge` command is Git's way of putting independent branches back together. The `--ff-only` flag tells git to
use `fast-forward` merges. This is preferable as fast-forward merge avoids creating merge commits.