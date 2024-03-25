from src.kron_solvers.core import KronArray, KGEN, _kbcd, kbcd, Stopping
import numpy as np
import timeit
from kron_groupper import Groupper

repeats = 1
alpha, lambd = .95, 1e-2
int_stop = Stopping(max_iter=1e4, rtol=1e-4, atol=1e-8)
ext_stop = Stopping(max_iter=1e4, rtol=1e-4, atol=1e-8)

_np = 'x0.fill(0.); _kbcd(Z, y, x0, gr.ixs, zeta, zetas, ts, int_stop=int_stop, ext_stop=ext_stop);'
_cy = 'x0.fill(0.); kbcd(Z, y, x0, gr.ixs, zeta, zetas, ts, int_stop=int_stop, ext_stop=ext_stop);'

M = 50
xs = np.empty(M)
ys = np.empty(M)
x2s = np.empty(M)

l = 0
for j in np.geomspace(1e2, 1e12, num=M):
    pn = qk = int(np.ceil(np.sqrt(j)))
    p = n = int(np.ceil(np.sqrt(pn)))
    q = k = int(np.ceil(np.sqrt(qk)))
    B = np.asfortranarray(np.eye(n, k))
    A = np.asfortranarray(np.eye(p, q))
    Z = KronArray(
        B,
        A
    )
    y = np.ones(Z.shape[0], order='F')
    gr = Groupper.from_constant(1, (q, k))
    ki = KGEN(Z, gr)
    zeta = (1.-alpha)*lambd
    Ls = np.fromiter(ki.lipschitz(zeta), dtype=np.float64)
    ts = 1./Ls 
    zetas = ki.etas*alpha*lambd
    x0 = np.empty(Z.shape[1], order='F')

    np_secs = timeit.timeit(_np, number=repeats, globals=globals())
    _x0 = x0.copy(order='F')
    cy_secs = timeit.timeit(_cy, number=repeats, globals=globals())
    assert np.allclose(_x0, x0)
    xs[l] = Z.size()
    x2s[l] = Z.size()/k
    ys[l] = np_secs/cy_secs
    print(f'{l:02d}\t{Z.size():.2e}\t{A.size*B.shape[0]:.2e}\t{np_secs/cy_secs:.1f}\t{np_secs/60:.1f}\t{2*np_secs/60:.1f}')
    l += 1

