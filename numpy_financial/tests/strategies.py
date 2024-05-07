import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

real_scalar_dtypes = st.one_of(
    npst.floating_dtypes(),
    npst.integer_dtypes(),
    npst.unsigned_integer_dtypes()
)
nicely_behaved_doubles = npst.from_dtype(
    np.dtype("f8"),
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
)
cashflow_array_strategy = npst.arrays(
    dtype=npst.floating_dtypes(sizes=64),
    shape=npst.array_shapes(min_dims=1, max_dims=2, min_side=0, max_side=25),
    elements=nicely_behaved_doubles,
)
cashflow_list_strategy = cashflow_array_strategy.map(lambda x: x.tolist())
cashflow_array_like_strategy = st.one_of(
    cashflow_array_strategy,
    cashflow_list_strategy,
)
short_nicely_behaved_doubles = npst.arrays(
    dtype=npst.floating_dtypes(sizes=64),
    shape=npst.array_shapes(min_dims=0, max_dims=1, min_side=0, max_side=5),
    elements=nicely_behaved_doubles,
)

when_strategy = st.sampled_from(
    ['end', 'begin', 'e', 'b', 0, 1, 'beginning', 'start', 'finish']
)
