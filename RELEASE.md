# Release process for `numpy-financial`

## Introduction

Example `version number`

- 1.8.dev0 # development version of 1.8 (release candidate 1)
- 1.8rc1 # 1.8 release candidate 1
- 1.8rc2.dev0 # development version of 1.8 release candidate 2
- 1.8 # 1.8 release
- 1.9.dev0 # development version of 1.9 (release candidate 1)

Releases are published to PyPI by the `Build Wheel and Release` workflow
(`.github/workflows/release.yml`), which starts when you push a `v*` tag.
It builds the wheels and the sdist, attests them, and uploads them with
PyPI trusted publishing. Do not build or upload artifacts by hand.

The workflow only runs in the `numpy` organization, and the publish job
uses the `release` GitHub environment. If that environment has required
reviewers, the upload waits for an approval.

## Process

- Set release variables:

      export VERSION=<version number>
      export PREVIOUS=<previous version number>
      export ORG="numpy"
      export REPO="numpy-financial"
      export NOTES="doc/source/_includes/release-notes.rst"

- Make sure the test suite is green on `main`, for all operating systems
  and Python versions in `.github/workflows/pythonpackage.yml`.

- Write the release notes for ${VERSION} at the top of the version list in
  ${NOTES}. To get a list of the merged pull requests since the last
  release:

      changelist ${ORG}/${REPO} v${PREVIOUS} main --version ${VERSION} --config pyproject.toml

- Set the release version. `__version__` in `numpy_financial/__init__.py`
  is the only place that holds it; `meson.build` reads it from there at
  build time, and the package metadata follows.

- Commit changes:

      git add numpy_financial/__init__.py ${NOTES}
      git commit -m "Designate ${VERSION} release"

- Tag the release in git:

      git tag -s v${VERSION} -m "signed ${VERSION} tag"

  If you do not have a gpg key, use -u instead; it is important for
  Debian packaging that the tags are annotated

- Push the new meta-data to github:

      git push --tags origin main

  where `origin` is the name of the `github.com:numpy/numpy-financial`
  repository

- Wait for the release workflow to complete, then check that the new
  version is on https://pypi.org/project/numpy-financial/

      - approve the deployment to the `release` environment, if it waits
        for a review
      - the workflow builds CPython wheels for the versions in
        `[tool.cibuildwheel]` in `pyproject.toml`; add the new
        interpreter there before you release for it

- Create release from tag

      - go to https://github.com/numpy/numpy-financial/releases/new?tag=v${VERSION}
      - add v${VERSION} for the `Release title`
      - paste the ${VERSION} section of ${NOTES} in the `Describe this release section`
      - if pre-release check the box labelled `Set as a pre-release`

- Update https://github.com/numpy/numpy-financial/milestones:

      - close old milestone
      - ensure new milestone exists (perhaps setting due date)

- Set the development version for the next cycle in
  `numpy_financial/__init__.py`.

- Commit changes:

      git add numpy_financial/__init__.py
      git commit -m 'Bump version'
      git push origin main
