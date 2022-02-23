import numpy as np

m = 100_000_000

x = np.random.uniform(size=m)
y = np.random.uniform(size=m)

hit = x**2 + y**2 <= 1.0

pi_est = 4.0 * np.sum(hit) / m