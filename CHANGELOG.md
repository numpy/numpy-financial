# 1.1.0

We're happy to announce the release of numpy-financial 1.1.0!

## Enhancements

- ENH: Use Newton's method to calculate IRR ([#47](https://github.com/numpy/numpy-financial/pull/47)).
- ENH: Allow npv calculation to be broadcastable ([#54](https://github.com/numpy/numpy-financial/pull/54)).
- Avoid redundant computations in IRR calculation ([#60](https://github.com/numpy/numpy-financial/pull/60)).
- ENH: NPV: Support calculation for vectors of rates and cashflows ([#96](https://github.com/numpy/numpy-financial/pull/96)).
- REL: Update minimum required versions ([#102](https://github.com/numpy/numpy-financial/pull/102)).
- ENH: npv: Rework npv function to mimic broadcasting ([#108](https://github.com/numpy/numpy-financial/pull/108)).
- Updated irr function [issue 98] ([#99](https://github.com/numpy/numpy-financial/pull/99)).
- ENH: nper: broadcast rework with Cython ([#118](https://github.com/numpy/numpy-financial/pull/118)).
- ENH: nper: Perform C division where appropriate ([#120](https://github.com/numpy/numpy-financial/pull/120)).
- ENH: mirr: Mimic broadcasting ([#125](https://github.com/numpy/numpy-financial/pull/125)).

## Documentation

- DOC: First draft to help getting the code ([#71](https://github.com/numpy/numpy-financial/pull/71)).
- Updating mirr docstring [issue 76] ([#86](https://github.com/numpy/numpy-financial/pull/86)).
- DOC: Add developer documentation for benchmarking ([#103](https://github.com/numpy/numpy-financial/pull/103)).
- DOC: build dev documentation ([#107](https://github.com/numpy/numpy-financial/pull/107)).
- DOC: Add documentation for editing another persons PR ([#111](https://github.com/numpy/numpy-financial/pull/111)).
- DOC: docs: Document how to build the documentation ([#106](https://github.com/numpy/numpy-financial/pull/106)).
- Doc/building with spin ([#115](https://github.com/numpy/numpy-financial/pull/115)).
- MAINT: Config benchmarks to work with asv again ([#124](https://github.com/numpy/numpy-financial/pull/124)).
- ENH: mirr: Mimic broadcasting ([#125](https://github.com/numpy/numpy-financial/pull/125)).

## Maintenance

- MAINT: Tidy up environment.yml ([#127](https://github.com/numpy/numpy-financial/pull/127)).
- CI: Fix CI errors ([#135](https://github.com/numpy/numpy-financial/pull/135)).
- MAINT: use a PEP 639 ``project.license`` SPDX expression ([#134](https://github.com/numpy/numpy-financial/pull/134)).

## Other

- BUG: fix some issues with `nper` ([#21](https://github.com/numpy/numpy-financial/pull/21)).
- BUG: make `ipmt` return `nan` for `per < 1` ([#22](https://github.com/numpy/numpy-financial/pull/22)).
- MAINT: clean up the implementation of `fv` ([#24](https://github.com/numpy/numpy-financial/pull/24)).
- BUG: prevent underflow/overflow when finding roots in IRR ([#25](https://github.com/numpy/numpy-financial/pull/25)).
- MAINT: small cleanups to `ipmt` ([#26](https://github.com/numpy/numpy-financial/pull/26)).
- TST: clean up `ppmt` tests ([#27](https://github.com/numpy/numpy-financial/pull/27)).
- Move tests into own classes ([#41](https://github.com/numpy/numpy-financial/pull/41)).
- DOC: Correctly link to NumPy's Code ([#42](https://github.com/numpy/numpy-financial/pull/42)).
- ENH: IRR: Select lowest positive rate ([#43](https://github.com/numpy/numpy-financial/pull/43)).
- TEST: Add regression test for gh-44 ([#45](https://github.com/numpy/numpy-financial/pull/45)).
- TEST: Drop python 3.5 tests ([#51](https://github.com/numpy/numpy-financial/pull/51)).
- ENH: Return close results in rate calculation ([#49](https://github.com/numpy/numpy-financial/pull/49)).
- TST/MAINT: Update CI scirpts ([#59](https://github.com/numpy/numpy-financial/pull/59)).
- Small addition to IRR function ([#62](https://github.com/numpy/numpy-financial/pull/62)).
- MAINT: Migrate to poetry ([#63](https://github.com/numpy/numpy-financial/pull/63)).
- DOC: Add GitHub issue templates ([#67](https://github.com/numpy/numpy-financial/pull/67)).
- TST: Make linting own step in CI pipeline ([#68](https://github.com/numpy/numpy-financial/pull/68)).
- MAINT: Test against Python 3.12 ([#69](https://github.com/numpy/numpy-financial/pull/69)).
- parametrize tests ([#75](https://github.com/numpy/numpy-financial/pull/75)).
- MAINT: Make `guess` parameter keyword only ([#77](https://github.com/numpy/numpy-financial/pull/77)).
- Initial change to pydata_sphinx_theme ([#65](https://github.com/numpy/numpy-financial/pull/65)).
- CI/STY: Move to Ruff ([#79](https://github.com/numpy/numpy-financial/pull/79)).
- MAINT: Run ruff on test file ([#80](https://github.com/numpy/numpy-financial/pull/80)).
- BENCH: Initial asv setup ([#81](https://github.com/numpy/numpy-financial/pull/81)).
- CI/BENCH: Add style checks for benchmarks ([#82](https://github.com/numpy/numpy-financial/pull/82)).
- Set up development docs as rst files ([#88](https://github.com/numpy/numpy-financial/pull/88)).
- Add automated documentation testing when building ([#94](https://github.com/numpy/numpy-financial/pull/94)).
- Removed all instances of assert_almost_equal ([#97](https://github.com/numpy/numpy-financial/pull/97)).
- REV: Remove numba ([#101](https://github.com/numpy/numpy-financial/pull/101)).
- DOC: Docs fixup ([#109](https://github.com/numpy/numpy-financial/pull/109)).
- BLD: Attempt to build using spin ([#114](https://github.com/numpy/numpy-financial/pull/114)).
- DOC: Move documentation to correct folder ([#113](https://github.com/numpy/numpy-financial/pull/113)).
- MAINT: Tidy up imports ([#117](https://github.com/numpy/numpy-financial/pull/117)).
- Altered IRR function to accept 2D-array ([#122](https://github.com/numpy/numpy-financial/pull/122)).
- DOC: Fixed a typo - `principle` to `principal` ([#129](https://github.com/numpy/numpy-financial/pull/129)).
- TYP: Inline typing annotations ([#136](https://github.com/numpy/numpy-financial/pull/136)).
- Add spin lint ([#142](https://github.com/numpy/numpy-financial/pull/142)).
- Test on newer Python versions ([#143](https://github.com/numpy/numpy-financial/pull/143)).
- Add trusted publishing workflows ([#144](https://github.com/numpy/numpy-financial/pull/144)).
- Derive version from numpy_financial/__init__.py ([#145](https://github.com/numpy/numpy-financial/pull/145)).
- Release docs ([#146](https://github.com/numpy/numpy-financial/pull/146)).
- Move release notes to CHANGELOG.md ([#147](https://github.com/numpy/numpy-financial/pull/147)).
- Move release notes to CHANGELOG.md ([#148](https://github.com/numpy/numpy-financial/pull/148)).
- Fix docs workflow ([#149](https://github.com/numpy/numpy-financial/pull/149)).
- Fix the documentation publishing workflow ([#150](https://github.com/numpy/numpy-financial/pull/150)).
- docs: do not fail if the dev directory exists ([#151](https://github.com/numpy/numpy-financial/pull/151)).
- Document how to publish the release documentation ([#152](https://github.com/numpy/numpy-financial/pull/152)).

## Contributors

17 authors added to this release (alphabetically):

- [@AlexMGTNO](https://github.com/AlexMGTNO)
- [@Eugenia-Mazur](https://github.com/Eugenia-Mazur)
- [@jlopezpena](https://github.com/jlopezpena)
- Daniel McCloy ([@drammock](https://github.com/drammock))
- Jamie Cook ([@jamiecook](https://github.com/jamiecook))
- Jasper Lee ([@yatshunlee](https://github.com/yatshunlee))
- Joren Hammudoglu ([@jorenham](https://github.com/jorenham))
- Josh Wilson ([@person142](https://github.com/person142))
- Kai Striega ([@Kai-Striega](https://github.com/Kai-Striega))
- Maharshi Basu ([@MashyBasker](https://github.com/MashyBasker))
- Mandeep Singh ([@mandeep-singh-sndk](https://github.com/mandeep-singh-sndk))
- Melissa Weber Mendonça ([@melissawm](https://github.com/melissawm))
- Mika ([@tal66](https://github.com/tal66))
- SeanZ ([@seanzian2093](https://github.com/seanzian2093))
- Sebastian Berg ([@seberg](https://github.com/seberg))
- Stefan van der Walt ([@stefanv](https://github.com/stefanv))
- Warren Weckesser ([@WarrenWeckesser](https://github.com/WarrenWeckesser))

17 reviewers added to this release (alphabetically):

- [@101AlexMartin](https://github.com/101AlexMartin)
- [@AlexMGTNO](https://github.com/AlexMGTNO)
- [@Eugenia-Mazur](https://github.com/Eugenia-Mazur)
- Daniel McCloy ([@drammock](https://github.com/drammock))
- Inessa Pawson ([@InessaPawson](https://github.com/InessaPawson))
- Jamie Cook ([@jamiecook](https://github.com/jamiecook))
- Jasper Lee ([@yatshunlee](https://github.com/yatshunlee))
- Joren Hammudoglu ([@jorenham](https://github.com/jorenham))
- Josh Wilson ([@person142](https://github.com/person142))
- Kai Striega ([@Kai-Striega](https://github.com/Kai-Striega))
- Maharshi Basu ([@MashyBasker](https://github.com/MashyBasker))
- Melissa Weber Mendonça ([@melissawm](https://github.com/melissawm))
- Mika ([@tal66](https://github.com/tal66))
- Ralf Gommers ([@rgommers](https://github.com/rgommers))
- SeanZ ([@seanzian2093](https://github.com/seanzian2093))
- Stefan van der Walt ([@stefanv](https://github.com/stefanv))
- Warren Weckesser ([@WarrenWeckesser](https://github.com/WarrenWeckesser))

# 1.0.0

* The transition of the source code from NumPy to this package is complete.

# 0.2.0

* Removed the use of `numpy.core.overrides.array_function_dispatch` to create
  wrappers of the financial functions.
* Support NumPy versions back to 1.15.

# 0.1.0

* This is the initial release of numpy-financial.  The functions were
  copied from NumPy 1.17.
