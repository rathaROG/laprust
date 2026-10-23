# Copyright (c) 2026 Ratha SIV | MIT License

"""Static signatures and documentation for the laprust entry points."""

from typing import Any, List, Optional, Tuple, Union, overload
from typing import Literal, Protocol
import numpy as np

_Precision = Literal["native", "float32", "float64"]
_Output = Literal["indices", "pairs"]

class _SingleSolver(Protocol):
    @overload
    def __call__(
        self, cost: Any,
        extend_cost: bool = False,
        cost_limit: float = ...,
        return_cost: Literal[True] = True,
        *,
        precision: _Precision = "native",
        output: Literal["indices"] = "indices",
    ) -> Tuple[float, np.ndarray, np.ndarray]: ...

    @overload
    def __call__(
        self, cost: Any,
        extend_cost: bool,
        cost_limit: float,
        return_cost: Literal[False],
        *,
        precision: _Precision = "native",
        output: Literal["indices"] = "indices",
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    @overload
    def __call__(
        self, cost: Any,
        extend_cost: bool = False,
        cost_limit: float = ...,
        *,
        return_cost: Literal[False],
        precision: _Precision = "native",
        output: Literal["indices"] = "indices",
    ) -> Tuple[np.ndarray, np.ndarray]: ...

    @overload
    def __call__(
        self, cost: Any,
        extend_cost: bool = False,
        cost_limit: float = ...,
        return_cost: bool = True,
        *,
        precision: _Precision = "native",
        output: Literal["indices"] = "indices",
    ) -> Union[Tuple[float, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]: ...

    @overload
    def __call__(
        self, cost: Any,
        extend_cost: bool = False,
        cost_limit: float = ...,
        return_cost: Literal[True] = True,
        *,
        precision: _Precision = "native",
        output: Literal["pairs"],
    ) -> Tuple[float, np.ndarray]: ...

    @overload
    def __call__(
        self, cost: Any,
        extend_cost: bool,
        cost_limit: float,
        return_cost: Literal[False],
        *,
        precision: _Precision = "native",
        output: Literal["pairs"],
    ) -> np.ndarray: ...

    @overload
    def __call__(
        self, cost: Any,
        extend_cost: bool = False,
        cost_limit: float = ...,
        *,
        return_cost: Literal[False],
        precision: _Precision = "native",
        output: Literal["pairs"],
    ) -> np.ndarray: ...

    @overload
    def __call__(
        self, cost: Any,
        extend_cost: bool = False,
        cost_limit: float = ...,
        return_cost: bool = True,
        *,
        precision: _Precision = "native",
        output: Literal["pairs"],
    ) -> Union[Tuple[float, np.ndarray], np.ndarray]: ...

    @overload
    def __call__(
        self, cost: Any, extend_cost: bool = False,
        cost_limit: float = ..., return_cost: bool = True,
        *, precision: _Precision = "native", output: _Output = "indices",
    ) -> Union[Tuple[float, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[float, np.ndarray], np.ndarray]: ...

class _BatchSolver(Protocol):
    @overload
    def __call__(
        self, costs: Any,
        extend_cost: bool = False,
        cost_limit: float = ...,
        return_cost: Literal[True] = True,
        n_threads: Optional[int] = 0,
        *,
        precision: _Precision = "native",
        output: Literal["indices"] = "indices",
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]: ...

    @overload
    def __call__(
        self, costs: Any,
        extend_cost: bool,
        cost_limit: float,
        return_cost: Literal[False],
        n_threads: Optional[int] = 0,
        *,
        precision: _Precision = "native",
        output: Literal["indices"] = "indices",
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]: ...

    @overload
    def __call__(
        self, costs: Any,
        extend_cost: bool = False,
        cost_limit: float = ...,
        *,
        return_cost: Literal[False],
        n_threads: Optional[int] = 0,
        precision: _Precision = "native",
        output: Literal["indices"] = "indices",
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]: ...

    @overload
    def __call__(
        self, costs: Any,
        extend_cost: bool = False,
        cost_limit: float = ...,
        return_cost: bool = True,
        n_threads: Optional[int] = 0,
        *,
        precision: _Precision = "native",
        output: Literal["indices"] = "indices",
    ) -> Union[Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]], Tuple[List[np.ndarray], List[np.ndarray]]]: ...

    @overload
    def __call__(
        self, costs: Any,
        extend_cost: bool = False,
        cost_limit: float = ...,
        return_cost: Literal[True] = True,
        n_threads: Optional[int] = 0,
        *,
        precision: _Precision = "native",
        output: Literal["pairs"],
    ) -> Tuple[np.ndarray, List[np.ndarray]]: ...

    @overload
    def __call__(
        self, costs: Any,
        extend_cost: bool,
        cost_limit: float,
        return_cost: Literal[False],
        n_threads: Optional[int] = 0,
        *,
        precision: _Precision = "native",
        output: Literal["pairs"],
    ) -> List[np.ndarray]: ...

    @overload
    def __call__(
        self, costs: Any,
        extend_cost: bool = False,
        cost_limit: float = ...,
        *,
        return_cost: Literal[False],
        n_threads: Optional[int] = 0,
        precision: _Precision = "native",
        output: Literal["pairs"],
    ) -> List[np.ndarray]: ...

    @overload
    def __call__(
        self, costs: Any,
        extend_cost: bool = False,
        cost_limit: float = ...,
        return_cost: bool = True,
        n_threads: Optional[int] = 0,
        *,
        precision: _Precision = "native",
        output: Literal["pairs"],
    ) -> Union[Tuple[np.ndarray, List[np.ndarray]], List[np.ndarray]]: ...

    @overload
    def __call__(
        self, costs: Any, extend_cost: bool = False,
        cost_limit: float = ..., return_cost: bool = True,
        n_threads: Optional[int] = 0,
        *, precision: _Precision = "native", output: _Output = "indices",
    ) -> Union[Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]], Tuple[List[np.ndarray], List[np.ndarray]], Tuple[np.ndarray, List[np.ndarray]], List[np.ndarray]]: ...

lapjv: _SingleSolver
lapjv_batch: _BatchSolver
