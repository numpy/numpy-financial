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

## Adding the upstream repo

Now that your fork of NumPy-Financial is available locally, it is worth adding the upstream repository as a remote. 

You can view the current remotes by running: 

```shell
git remote -v
```

This should produce some output similar to:

```shell
origin	https://github.com/<your_username>/numpy-financial.git (fetch)
origin	https://github.com/<your_username>/numpy-financial.git (push)
```

Now tell git that there is a remote repository that we will call ``upstream`` pointing to the numpy-financial repository: 

```shell
git remote add upstream https://github.com/numpy/numpy-financial.git
```

We can now check the remotes again:

```shell
git remote -v
```

which gives two additional lines as output:

```shell
origin	https://github.com/<your_username>/numpy-financial.git (fetch)
origin	https://github.com/<your_username>/numpy-financial.git (push)
upstream	https://github.com/numpy/numpy-financial.git (fetch)
upstream	https://github.com/numpy/numpy-financial.git (push)
```


## Pulling from upstream by default

We want to be able to get the changes from the upstream repo by default. This way you pull the most recent changes into your repo.

To set up your repository to read from the remote that we called `upstream`:

```shell
git config branch.main.remote upstream
git config branch.main.merge refs/heads/main
```

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