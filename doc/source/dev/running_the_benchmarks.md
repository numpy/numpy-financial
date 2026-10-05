# Running the benchmarks

This document outlines how to setup and run the benchmarks using [asv](https://asv.readthedocs.io/en/v0.6.1/).

## Running the benchmarks

With the development environment activated, register your machine once:

```shell
asv machine --yes
```

Then run:

```shell
spin bench
```

This builds the local checkout and runs the benchmarks against that build,
including uncommitted changes. It uses the current Python environment and
does not save results. Because it is a dry run, the timings are only a rough guide and are not meant to be compared across sessions or machines. To select benchmarks by name or regular expression:

```shell
spin bench -t Npv2D.time_broadcast
```

To benchmark committed revisions and save results for publishing, use ASV:

```shell
asv run
```

## Viewing the results

There are two steps to viewing the results locally. The results need to be published and then launched in a local web browser.

To publish the results use:

```shell
asv publish
```

And then to view the results:

```shell
asv preview
```

This will launch a local web browser from which you can view the results
