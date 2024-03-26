from typing import Literal

from hypothesis import strategies as st
from hypothesis.extra import numpy as npst


# numba only supports 32-bit or 64-bit little endian values
NUMBA_ALLOWED_SIZES: list[Literal[32, 64]] = [32, 64]
NUMBA_ALLOWED_ENDIANNESS = "<"


def float_dtype():
    return npst.floating_dtypes(
        sizes=NUMBA_ALLOWED_SIZES,
        endianness=NUMBA_ALLOWED_ENDIANNESS,
    )


def int_dtype():
    return npst.integer_dtypes(
        sizes=NUMBA_ALLOWED_SIZES,
        endianness=NUMBA_ALLOWED_ENDIANNESS,
    )


def uint_dtype():
    return npst.unsigned_integer_dtypes(
        sizes=NUMBA_ALLOWED_SIZES,
        endianness=NUMBA_ALLOWED_ENDIANNESS,
    )


real_scalar_dtypes = st.one_of(float_dtype(), int_dtype(), uint_dtype())
cashflow_array_strategy = npst.arrays(
    dtype=real_scalar_dtypes,
    shape=npst.array_shapes(min_dims=1, max_dims=2, min_side=0, max_side=25),
)
cashflow_list_strategy = cashflow_array_strategy.map(lambda x: x.tolist())
cashflow_array_like_strategy = st.one_of(
    cashflow_array_strategy,
    cashflow_list_strategy,
)
short_scalar_array = npst.arrays(
    dtype=real_scalar_dtypes,
    shape=npst.array_shapes(min_dims=0, max_dims=1, min_side=0, max_side=5),
)
when_strategy = st.sampled_from(
    ['end', 'begin', 'e', 'b', 0, 1, 'beginning', 'start', 'finish']
)
