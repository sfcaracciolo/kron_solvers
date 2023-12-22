from src.kron_solvers.core import KronArray, _kapg, kapg, Stopping
import numpy as np
import timeit

rng = np.random.default_rng(seed=0)

p, q = rng.integers(10, 50), rng.integers(10, 50)
n, k = rng.integers(10, 50), 1

Z = KronArray(
    np.asfortranarray(rng.random((n,k))),
    np.asfortranarray(rng.random((p,q)))
)
b = np.asfortranarray(rng.random(Z.shape[0]))
x0 = np.zeros(Z.shape[1], order='F')

alpha, beta = .5, .5
stop = Stopping(max_iter=1e4, rtol=1e-3, atol=1e-12)
L = Z.norm(ord=2)**2 + alpha
t = 2./L


start = timeit.default_timer()
n = _kapg(Z, b, x0, t, alpha, beta, stop=stop)
np_secs = timeit.default_timer() - start

_x0 = x0.copy()
x0.fill(0.)

start = timeit.default_timer()
n = kapg(Z, b, x0, t, alpha, beta, stop=stop)
cy_secs = timeit.default_timer() - start

print(f"Cython is x{np_secs/cy_secs:.1f} faster than NumPy")

assert np.allclose(_x0, x0)
print('OK')

