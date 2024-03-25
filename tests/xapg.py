from src.kron_solvers.core import KronArray, KGEN, _kapg, kapg, Stopping
import numpy as np
import timeit

N = 10
rng = np.random.default_rng(seed=0)
alpha, beta = .5, .5
int_stop = Stopping(max_iter=1e4, rtol=1e-3, atol=1e-8)

_np = 'x0.fill(0.); _kapg(Z, y, x0, t, alpha, beta, stop=int_stop)'
_cy = 'x0.fill(0.); kapg(Z, y, x0, t, alpha, beta, stop=int_stop)'

for j in range(1, N):
    pn = q = int(np.ceil(np.sqrt(10**j)))
    p = n = int(np.ceil(np.sqrt(pn)))
    B = np.asfortranarray(rng.random((n, 1)))
    A = np.asfortranarray(np.eye(p, q))
    # A = np.asfortranarray(rng.random((p, q)))
    Z = KronArray(
        B,
        A
    )
    x = np.ones(Z.shape[1], order='F')
    y = Z._matvec(x)
    x0 = np.empty(Z.shape[1], order='F')
    L = Z.norm(ord=2)**2 + alpha
    t = 1./L

    np_secs = timeit.timeit(_np, number=3, globals=globals())
    _x0 = x0.copy(order='F')
    cy_secs = timeit.timeit(_cy, number=3, globals=globals())
    assert np.allclose(_x0, x0)
    print(f'dim = {Z.shape} -> {Z.size():.1e}\t x = {np_secs/cy_secs:.1f}')

