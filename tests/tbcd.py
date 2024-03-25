from src.kron_solvers.core import KronArray, KGEN, kbcd, _kbcd, Stopping
import numpy as np
import timeit
from kron_groupper import Groupper
from geometric_plotter import Plotter 

repeats = 1
alpha, lambd = .95, 1e-2
int_stop = Stopping(max_iter=1e4, rtol=1e-4, atol=1e-8)
ext_stop = Stopping(max_iter=1e4, rtol=1e-4, atol=1e-8)

_cy = 'kbcd(Z, y, x0, gr.ixs, zeta, zetas, ts, int_stop=int_stop, ext_stop=ext_stop);'
_np = '_kbcd(Z, y, x0, gr.ixs, zeta, zetas, ts, int_stop=int_stop, ext_stop=ext_stop);'

M = 50
xs = np.empty(M)
ys = np.empty(M)
y2s = np.empty(M)

l = 0
for j in np.geomspace(1e2, 1e9, num=M):
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

    x0.fill(0.)
    cy_secs = timeit.timeit(_cy, number=repeats, globals=globals())
    x0.fill(0.)
    np_secs = timeit.timeit(_np, number=repeats, globals=globals())
    xs[l] = Z.size()
    ys[l] = np_secs
    y2s[l] = cy_secs
    print(f'{l:02d}\t{Z.size():.2e}\t{cy_secs:.1f}\t{np_secs:.1f}\t{np_secs/cy_secs:.1f}')
    l += 1

p = Plotter(_2d=True)
p.axs.loglog(xs, ys, '-k')
p.axs.loglog(xs, ys, '.k', markersize=10)

p.axs.loglog(xs, y2s, '-r')
p.axs.loglog(xs, y2s, '.r', markersize=10)
Plotter.show()
