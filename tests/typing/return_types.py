"""Checked with pyright; exercise every documented return shape."""
from typing import List, Tuple
import numpy as np
import laprust

a = np.zeros((2, 3))
b = np.zeros((2, 2, 3))
single: Tuple[float, np.ndarray, np.ndarray] = laprust.lapjv(a, extend_cost=True)
indices: Tuple[np.ndarray, np.ndarray] = laprust.lapjv(a, True, return_cost=False)
pairs: np.ndarray = laprust.lapjv(a, True, return_cost=False, output="pairs")
with_cost: Tuple[float, np.ndarray] = laprust.lapjv(a, True, output="pairs")
batch: Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]] = laprust.lapjv_batch(b, True)
batch_indices: Tuple[List[np.ndarray], List[np.ndarray]] = laprust.lapjv_batch(b, True, return_cost=False)
batch_pairs: List[np.ndarray] = laprust.lapjv_batch(b, True, return_cost=False, output="pairs")
batch_cost: Tuple[np.ndarray, List[np.ndarray]] = laprust.lapjv_batch(b, True, output="pairs")
