# Release process for `numpy-financial`

## Introduction

Example `version number`, set in `__init__.py`:

- 1.8.dev0 # development version of 1.8 (release candidate 1)
- 1.8rc1 # 1.8 release candidate 1
- 1.8rc2.dev0 # development version of 1.8 release candidate 2
- 1.8 # 1.8 release
- 1.9.dev0 # development version of 1.9 (release candidate 1)

Releases are published to PyPI by the `Build Wheel and Release` workflow
(`.github/workflows/release.yml`), which starts when you push a `v*` tag.
It builds the wheels and the sdist, attests them, and uploads them with
PyPI trusted publishing. Do not build or upload artifacts by hand.

## Process

- Set release variables:

      export VERSION=<version number>
      export PREVIOUS=<previous version number>
      export ORG="numpy"
      export REPO="numpy-financial"
      export NOTES="CHANGELOG.md"
      export CHANGES="${VERSION}.md"

- Make sure the test suite is green on `main`, for all operating systems
  and Python versions in `.github/workflows/pythonpackage.yml`.

- Generate the list of merged pull requests since the last release:

      changelist ${ORG}/${REPO} v${PREVIOUS} main --version ${VERSION} --config pyproject.toml --out ${CHANGES}

  ${CHANGES} is a scratch file. Keep it until the GitHub release is
  made, then delete it. Do not commit it.

- Put the generated notes at the top of ${NOTES}:

      cat ${CHANGES} | cat - ${NOTES} > temp && mv temp ${NOTES}

  Then edit the new section: keep what is useful to a reader, and remove
  the rest.

- Set the release version. `__version__` in `numpy_financial/__init__.py`.

- Ensure that the Python versions you want to build for are updated in
  `[tool.cibuildwheel]` in `pyproject.toml`.

- Commit changes:

      git add numpy_financial/__init__.py pyproject.toml ${NOTES}
      git commit -m "Designate ${VERSION} release"

- Tag the release in git:

      git tag -s v${VERSION} -m "signed ${VERSION} tag"

  If you do not have a gpg key, use -u instead; it is important for
  Debian packaging that the tags are annotated.

- Push the new meta-data to github:

      git push --tags origin main

  where `origin` is the name of the `github.com:numpy/numpy-financial`
  repository

- Wait for the release workflow to complete, then check that the new
  version is on https://pypi.org/project/numpy-financial/

      - Approve the deployment to the `release` environment, if it waits
        for a review.

- Create release from tag

      - go to https://github.com/numpy/numpy-financial/releases/new?tag=v${VERSION}
      - add v${VERSION} for the `Release title`
      - paste the contents of ${CHANGES} in the `Describe this release section`
      - if pre-release check the box labelled `Set as a pre-release`

- Delete the scratch file:

      rm ${CHANGES}

- Publish the documentation for the release. A GH workflow builds
  `dev/` from `main`; the released versions have to be published by
  hand. Skip this step for a pre-release.

      - wait for the workflow to rebuild `dev/` from the release
        commit, and check that https://numpy.org/numpy-financial/dev/
        shows ${VERSION}. Do this before the version bump below, which
        returns `dev/` to a development version.

      - in a clone of the `gh-pages` branch, copy those docs to their
        permanent location and point `latest` at them:

            cp -r dev version/${VERSION}
            rm -rf latest
            ln -s version/${VERSION} latest

      - commit and push `gh-pages`

- Update https://github.com/numpy/numpy-financial/milestones:

      - close old milestone
      - ensure new milestone exists (perhaps setting due date)

- Set the development version for the next cycle in
  `numpy_financial/__init__.py`.

- Commit changes:

      git add numpy_financial/__init__.py
      git commit -m 'Bump version'
      git push origin main
