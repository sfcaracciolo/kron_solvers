# Kronecker Group Elastic-Net

Method for solving a linear regression problem subject to group LASSO and ridge penalisation when the model has a Kronecker structure. 

Let $A\in \mathbb{R}^{p\times q}$, $D\in \mathbb{R}^{n\times k}$, $y \in \mathbb{R^{pn}}$, $\lambda, \eta_i > 0$ and $\alpha \in [0, 1]$,

$$ \underset{\theta}{\min} \; \frac{1}{2}\|\sum_{i \in \Gamma} Z_i \theta_i - y\|_2^2 + \lambda \left[(1-\alpha) \frac{1}{2} \|\theta\|_2^2 + \alpha \sum_{i\in \Gamma} \eta_i \|\theta_i\|_2\right] $$ 

where $Z = D \otimes A$ and $\Gamma$ is a partition of $\{0,\dots,qk-1\}$ with the following constraints for each i-group: $\exists r_i, a_i, b_i \in \mathbb{N}$ such as,

$$ i \;\text{div}\; q = \{r_i,\dots,r_i\} \\ i \;\text{mod}\; q = \{a_i,a_i+1,\dots,b_i-2, b_i-1\} $$ 

where $\text{div}$ is the integer division and $\text{mod}$ is the module operator. The [kron_groupper](https://github.com/sfcaracciolo/kron_groupper) lib has been developed to handle this constraints. 

The solver consists of an accelerated proximal gradient method and a block coordinate descent algorithm coded in NumPy (underscored version `_kapg` & `_kbcd`) and Cython (noscored version `kapg` & `kbcd`). The Cython version uses scipy's BLAS wrappers to speed up vector/matrix operations.

## Install

Clone the repo and compile the cython source through 
```bash
cythonize -i src\kron_solvers\solvers.pyx
```
Or try with pip
```bash
pip install git+https://github.com/sfcaracciolo/kron_solvers.git
```

## Examples
### APG method
Run the following to test apg method and use the `KronArray` and `Stopping` classes
```python
from kron_solvers import KronArray, _kapg, kapg, Stopping
import numpy as np
import timeit
rng = np.random.default_rng(seed=0)

p, q = rng.integers(10, 50), rng.integers(10, 50)
n, k = rng.integers(10, 50), 1

Z = KronArray(
    np.asfortranarray(rng.random((n,k))),
    np.asfortranarray(rng.random((p,q)))
)

x = np.ones(Z.shape[1], dtype=np.float64, order='F')
b = Z.matvec(x)
x0 = np.zeros_like(x)

alpha, beta = .5, .5
stop = Stopping(max_iter=1e5, rtol=1e-8, atol=1e-12)
L = Z.norm(ord=2)**2 + alpha
t = 1./L

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
```
### BCD method
and this for `bcd` algorithm using `Groupper` and `KGEN` classes
```python
from kron_solvers import KronArray, KGEN, _kbcd, kbcd, Stopping
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
int_stop = Stopping(max_iter=1e5, rtol=1e-3, atol=1e-12)
ext_stop = Stopping(max_iter=1e4, rtol=1e-3, atol=1e-12)

gr = Groupper.from_constant(1, (q, k))
ki = KGEN(Z, gr)
Ls = np.fromiter(ki.lipschitz(zeta), dtype=np.float64)
ts = 1./Ls 

start = timeit.default_timer()
n = _kbcd(Z, y, x0, gr.ixs, zeta, zetas, ts, int_stop=int_stop, ext_stop=ext_stop)
np_secs = timeit.default_timer() - start

_x0 = x0.copy()
x0.fill(0.)

start = timeit.default_timer()
n = kbcd(Z, y, x0, gr.ixs, zeta, zetas, ts, int_stop=int_stop, ext_stop=ext_stop)
cy_secs = timeit.default_timer() - start

print(f"Cython is x{np_secs/cy_secs:.1f} faster than NumPy")

assert np.allclose(_x0, x0)
```
