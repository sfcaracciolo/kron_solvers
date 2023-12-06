from src.kron_solvers.core import KronArray, KGEN, _kbcd, kbcd
import numpy as np
import timeit
from kron_groupper import Groupper

rng = np.random.default_rng(seed=0)

p, q = rng.integers(10, 50), rng.integers(10, 50)
n, k = rng.integers(10, 50), rng.integers(10, 50)

Z = KronArray(
    np.asfortranarray(rng.random((n,k))),
    np.asfortranarray(rng.random((p,q)))
)
y = np.asfortranarray(rng.random(Z.shape[0]))
x0 = np.zeros(Z.shape[1], order='F')
zetas = np.zeros(Z.shape[1], order='F')

zeta = .5
int_max_iter = 1e5
int_tol = 1e-3
ext_max_iter = 1e3
ext_tol = 1e-3

gr = Groupper.from_constant(1, (q, k))
# print((q,k), q*k, ixs)
ki = KGEN(Z, gr)
Ls = np.fromiter(ki.lipschitz(zeta), dtype=np.float64)
ts = 1./Ls 

start = timeit.default_timer()
n = _kbcd(Z, y, x0, gr.ixs, zeta, zetas, ts, ext_max_iter, ext_tol, int_max_iter, ext_tol)
np_secs = timeit.default_timer() - start

_x0 = x0.copy()
x0.fill(0.)

start = timeit.default_timer()
n = kbcd(Z, y, x0, gr.ixs, zeta, zetas, ts, ext_max_iter, ext_tol, int_max_iter, ext_tol)
cy_secs = timeit.default_timer() - start

print(f"Cython is x{np_secs/cy_secs:.1f} faster than NumPy")

assert np.allclose(_x0, x0)
print('OK')


