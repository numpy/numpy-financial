from typing import TypeAlias

import numpy as np
import numpy.typing as npt

_ArrayF64: TypeAlias = npt.NDArray[np.float64]

def nper(
    rates: _ArrayF64,
    pmts: _ArrayF64,
    pvs: _ArrayF64,
    fvs: _ArrayF64,
    whens: _ArrayF64,
    out: _ArrayF64,
) -> None: ...
def npv(rates: _ArrayF64, values: _ArrayF64, out: _ArrayF64) -> None: ...
