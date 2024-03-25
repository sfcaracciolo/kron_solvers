from src.kron_solvers.core import KronArray, _kapg, kapg, Stopping
import numpy as np
import timeit
from regularization_tools import Ridge
rng = np.random.default_rng(seed=0)

p, q = rng.integers(10, 50), rng.integers(10, 50)
n, k = rng.integers(10, 50), 1

# A = Ridge.ill_cond_matrix(p, k=8, seed=0)
# q = p

Z = KronArray(
    np.asfortranarray(rng.random((n,k))),
    # np.asfortranarray(A)
    np.asfortranarray(rng.random((p,q)))
)

x = np.ones(Z.shape[1], dtype=np.float64, order='F')
b = Z.matvec(x)
# b = np.asfortranarray(rng.random(Z.shape[0]))
x0 = np.zeros_like(x)

alpha, beta = .5, .5
stop = Stopping(max_iter=1e5, rtol=1e-8, atol=1e-12)
L = Z.norm(ord=2)**2 + alpha
t = 1./L


start = timeit.default_timer()
n = _kapg(Z, b, x0, t, alpha, beta, stop=stop)
np_secs = timeit.default_timer() - start
print('Numpy')
print(n)
print(x0)

_x0 = x0.copy()
x0.fill(0.)

start = timeit.default_timer()
n = kapg(Z, b, x0, t, alpha, beta, stop=stop)
cy_secs = timeit.default_timer() - start
print('Cython')
print(n)
print(x0)

print(f"Cython is x{np_secs/cy_secs:.1f} faster than NumPy")

assert np.allclose(_x0, x0)
print('OK')

