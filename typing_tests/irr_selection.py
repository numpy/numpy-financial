"""Static IRR selector contract, checked by the repository-wide mypy job."""
import numpy as np
import numpy.typing as npt

import numpy_financial as npf


def select_python_float(candidates: npt.NDArray[np.float64]) -> float:
    return float(candidates[0])


def select_numpy_float(candidates: npt.NDArray[np.float64]) -> np.float64:
    return candidates[0]


def select_numpy_float32(candidates: npt.NDArray[np.float64]) -> np.float32:
    return np.float32(candidates[0])


def select_array(candidates: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return candidates


cashflows = [-5, 10.5, 1, -8, 1]
default_result: float = npf.irr(cashflows)
python_result: float = npf.irr(cashflows, selection_logic=select_python_float)
numpy_result: float = npf.irr(cashflows, selection_logic=select_numpy_float)
float32_result: float = npf.irr(cashflows, selection_logic=select_numpy_float32)
npf.irr(cashflows, selection_logic=select_array)  # type: ignore[arg-type]
