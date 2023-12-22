from src.kron_solvers.core import KronArray, KGEN, Stopping
import numpy as np
import timeit
from kron_groupper import Groupper
from regularization_tools import Ridge

rng = np.random.default_rng(seed=0)

p, q = 4, 5
n, k = 3, 4

B = np.asfortranarray(rng.random((n,k)))
A = np.asfortranarray(rng.random((p,q)))
y = np.asfortranarray(rng.random((p*n)))
Z = KronArray(B, A)

n_lambdas = 50

model = Ridge(np.kron(B, A))
lambdas = model.lambdaspace(Z.norm(ord=2), Z.norm(ord=-2), n_lambdas) # decreasing order
start = timeit.default_timer()
xtkh = model.solve(y, lambdas)
tkh_secs = timeit.default_timer() - start

solver_params = dict(
    alpha = 0.,
    int_stop = Stopping(max_iter=1e6, rtol=1e-8, atol=1e-12),
    ext_stop = Stopping(max_iter=1e6, rtol=1e-8, atol=1e-12)
)

gr = Groupper.from_constant(1, (q, k))
ki = KGEN(Z, gr)
xbcd = np.empty_like(xtkh)
N = y.size 

start = timeit.default_timer()
xbcd = ki.solve(y, (lambdas**2), **solver_params).reshape((n_lambdas, Z.q*Z.k, 1), order='F')
bcd_secs = timeit.default_timer() - start
print(f"Thk0 is x{bcd_secs/tkh_secs:.1f} faster than Cython")

assert np.allclose(xtkh, xbcd, rtol=1e-3, atol=1e-12)
print('OK')


