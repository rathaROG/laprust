"""Run after installing laprust."""
import numpy as np
import laprust

cost = np.array([[3., 1.], [1., 4.]])
total, rows, cols = laprust.lapjv(cost)
print("single:", total, rows, cols)
totals, rows, cols = laprust.lapjv_batch(np.stack([cost, cost]), n_threads=2)
print("batch:", totals, rows, cols)
cost = np.array([[.1, .8, .9], [.9, .7, .8]], dtype=np.float32)
print("rejection:", laprust.lapjv(cost, cost_limit=.5))
