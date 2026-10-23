"""Native Rust linear assignment solvers with implicit rejection and batch solving."""
from ._lapjv import lapjv, lapjv_batch

__version__ = "1.0.0"
__author__ = 'Ratha SIV'
__description__ = 'Native Rust linear assignment solvers with implicit rejection and batch solving.'
__homepage__ = 'https://github.com/rathaROG/laprust'
__all__ = ["lapjv", "lapjv_batch"]
