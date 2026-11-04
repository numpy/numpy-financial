import numpy as np
import pytest
from numpy.testing import assert_allclose

import numpy_financial as npf


@pytest.mark.parametrize("rate", [0.0, 0.05, -0.05, -1.0, -1.5, 1e-12,
                                  np.inf, np.nan])
@pytest.mark.parametrize("nper", [0, -3, 0.5, 120, 10000])
@pytest.mark.parametrize("when", ["begin", "end"])
def test_scalar_matches_array_path(rate, nper, when):
    # A zero-dimensional rate array exercises the general implementation.
    with np.errstate(all="ignore"):
        expected = npf.pmt(np.array(rate), nper, 1000, -100, when)
        actual = npf.pmt(rate, nper, 1000, -100, when)
    assert type(actual) is type(expected)
    assert_allclose(actual, expected, rtol=1e-13)


@pytest.mark.parametrize("args", [(0.0, 0, 100), (-1.5, 0.5, 100),
                                  (10.0, 10000, 100)])
def test_scalar_respects_seterr(args):
    with np.errstate(all="raise"):
        with pytest.raises(FloatingPointError):
            npf.pmt(*args)


@pytest.mark.parametrize("value", [np.float32(100), np.int64(100), 2**64])
def test_scalar_other_types(value):
    expected = npf.pmt(np.array(0.05), 10, value)
    actual = npf.pmt(0.05, 10, value)
    assert type(actual) is type(expected)
    assert_allclose(actual, expected)
