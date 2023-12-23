from typing import TypedDict
import numpy as np 
import scipy as sp 
from .solvers import apg, bcd
from kron_groupper import Groupper
from regularization_tools import AbstractRegularizer 

class Stopping(TypedDict):
    max_iter : int
    rtol : float 
    atol : float 
    
class KronArray(sp.sparse.linalg.LinearOperator):

    def __init__(self, B: np.ndarray, A: np.ndarray) -> None:

        if not A.flags.f_contiguous:
            raise ValueError('A should be a Fortran array.')

        if A.dtype != np.float64:
            raise ValueError('A should be float64 dtype.')
        
        if B.dtype != np.float64:
            raise ValueError('B should be float64 dtype.')

        self.p, self.q = A.shape
        self.n, self.k = B.shape

        self.B = B
        self.A = A

        super().__init__(shape=(self.p*self.n ,self.q*self.k), dtype=np.float64)
    
    def _matvec(self, v: np.ndarray) -> np.ndarray:
        V = np.reshape(v, (self.q, self.k), order='F')
        return np.linalg.multi_dot([self.A, V, self.B.T]).reshape(-1, order='F')

    def _rmatvec(self, v: np.ndarray) -> np.ndarray:
        V = np.reshape(v, (self.p, self.n), order='F')
        return np.linalg.multi_dot([self.A.T, V, self.B]).reshape(-1, order='F')
    
    def _bnorm(self, ord: str | int = 'fro') -> float:
        return np.linalg.norm(self.B, ord=ord)
    

    def norm(self, ord: str | int = 'fro') -> float:
        return self._anorm(ord=ord) * self._bnorm(ord=ord)

    def _anorm(self, ord: str | int = 'fro') -> float:
        return np.linalg.norm(self.A, ord=ord)
    
    def _asub(self, a:int, b:int) -> np.ndarray:
        i = a % self.q
        j = b % self.q if b % self.q else self.q
        Ai = self.A[:,i:j] # view
        return Ai 
    
    def _bsub(self, a:int) -> np.ndarray:
        k = a // self.q 
        Bi = self.B[:, k:k+1]
        return Bi

    def sub(self, a: int, b: int):
        Bi = self._bsub(a)
        Ai = self._asub(a, b)
        return KronArray(Bi, Ai)

class KGEN(AbstractRegularizer):

    def __init__(self, Z: KronArray, gr: Groupper):
        self.Z = Z
        self.gr = gr
        self.etas = np.sqrt(gr.get_group_sizes())
        self.etas /= np.sum(self.etas)
        
    def norms(self, ord: str | int ='fro') -> float:
        for _, (a, b) in self.gr.it():
            Zi = self.Z.sub(a, b)
            yield Zi.norm(ord=ord)

    def xnorms(self, x: np.ndarray, ord: str | int ='fro') -> float:
        for _, (a, b) in self.gr.it():
            xi = x[a:b]
            yield np.linalg.norm(xi, ord=ord)

    def lipschitz(self, zeta: float) -> float: 
        for norm in self.norms(ord=2):
            yield norm**2 + zeta
    
    def lambda_max(self, y: np.ndarray, alpha: float):
        # N = y.size
        dcs = np.fromiter(self.discard_conds(y), dtype=np.float64)
        w = np.max(dcs/self.etas)
        return w/alpha
        # return w/(N*alpha)
    
    def discard_conds(self, y: np.ndarray):
        
        if np.allclose(y, 0):
            raise ValueError('Y must be not null.')

        for _, (a, b) in self.gr.it():
            Zi = self.Z.sub(a, b)
            yield np.linalg.norm(Zi.rmatvec(y), ord=2)

    def penalization(self, X: np.ndarray, alpha: float):
        n_lambdas = X.shape[0]
        P = np.empty(n_lambdas)
        for i in range(n_lambdas):
            x = X[i].reshape(-1, order='F')
            l2_norm = np.fromiter(self.xnorms(x, ord=2), dtype=np.float64)
            l21_norm = np.inner(self.etas, l2_norm)
            P[i] = (1.-alpha)*.5*np.linalg.norm(x, ord=2)**2 + alpha*l21_norm
        return P 
    
    def residual(self, X: np.ndarray, y: np.ndarray):
        n_lambdas = X.shape[0]
        R = np.empty(n_lambdas)
        for i in range(n_lambdas):
            x = X[i].reshape(-1, order='F')
            R[i] = np.linalg.norm(self.Z.matvec(x) - y, ord=2) # **2/2
        return R
    
    # def lambdaspace(self, l_max: float, epsilon: float = 1e-3, num: int = 100):
    #     return super().lambdaspace(start=l_max, end=l_max*epsilon, num=num)
    
    def sols(self, X: np.ndarray):
        return np.tensordot(X, self.Z.B, axes=(2, 1))
    
    def solve(
            self, 
            y: np.ndarray, 
            lambdas: np.ndarray, 
            alpha: float=1.,
            int_stop: Stopping = {'max_iter': 1e4, 'rtol': 1e-3, 'atol': 1e-8},
            ext_stop: Stopping = {'max_iter': 1e4, 'rtol': 1e-3, 'atol': 1e-8},
        ):

        if lambdas[0] < lambdas[-1]:
            raise ValueError('Lambdas must be in decreasing order.')
        
        n_groups = self.gr.get_n_groups()
        X = np.zeros((lambdas.size, self.Z.q, self.Z.k), order='F')
        x0 = np.zeros((self.Z.shape[1]), order='F')
        zetas = np.empty(n_groups, order='F')
        L0s = np.fromiter(self.lipschitz(0), dtype=np.float64)
        ts = np.empty(n_groups, order='F')

        for i, lambd in enumerate(lambdas):
            zeta = (1.-alpha)*lambd
            zetas[:] = self.etas*alpha*lambd
            ts[:] = 1./(L0s + zeta)

            n = kbcd(
                self.Z,
                y,
                x0,
                self.gr.ixs,
                zeta,
                zetas,
                ts,
                int_stop,
                ext_stop,
            )
            print(f"lambda = {i}\titers {n}/{ext_stop['max_iter']:.0f}")
            X[i, ...] = x0.reshape((self.Z.q, self.Z.k), order='F')
        return X

def kbcd(
        Z: KronArray,
        y: np.ndarray,
        x0: np.ndarray,
        ix: np.ndarray,
        zeta: float,
        zetas: np.ndarray,
        int_ts: np.ndarray,
        int_stop: Stopping = {'max_iter': 1e4, 'rtol': 1e-3, 'atol': 1e-8},
        ext_stop: Stopping = {'max_iter': 1e4, 'rtol': 1e-3, 'atol': 1e-8},
    ):
    """Optimize (1/2)*||Zx-y||_2^2 + (zeta/2)*||x||_2^2 + ||x||_w21 being ||x||_w21 the  l21 norm weighting by zetas.
    solution is returned on x0"""

    N, qk = Z.shape
    r = np.empty(N, order='F', dtype=np.float64)
    x1 = np.empty(qk, order='F', dtype=np.float64)
    dx = np.empty(qk, order='F', dtype=np.float64)

    aux_pk = np.empty((Z.p, Z.k), order='F', dtype=np.float64)
    aux_p = np.empty(Z.p, order='F', dtype=np.float64)
    aux_q = np.empty(Z.q, order='F', dtype=np.float64)

    # apg memory allocation 
    int_y0 = np.empty(Z.q, order='F', dtype=np.float64)
    int_r = np.empty(N, order='F', dtype=np.float64)
    int_grad_f = np.empty(Z.q, order='F', dtype=np.float64)
    int_x1 = np.empty(Z.q, order='F', dtype=np.float64)
    int_dx = np.empty(Z.q, order='F', dtype=np.float64)
    int_aux_p = np.empty(Z.p, order='F', dtype=np.float64)

    return bcd(Z.A, Z.B, y, x0, ix, zeta, zetas, int_ts, int_stop['max_iter'], int_stop['rtol'], int_stop['atol'], ext_stop['max_iter'], ext_stop['rtol'], ext_stop['atol'], r, x1, dx, aux_pk, aux_p, aux_q, int_y0, int_r, int_grad_f, int_x1, int_dx, int_aux_p)

def _kbcd(
        Z: KronArray,
        y: np.ndarray,
        x0: np.ndarray,
        ixs: np.ndarray,
        zeta: float,
        zetas: np.ndarray,
        int_ts: np.ndarray,
        int_stop: Stopping = {'max_iter': 1e4, 'rtol': 1e-3, 'atol': 1e-8},
        ext_stop: Stopping = {'max_iter': 1e4, 'rtol': 1e-3, 'atol': 1e-8},
    ):
    """Optimize (1/2)*||Zx-y||_2^2 + (zeta/2)*||x||_2^2 + ||x||_w21 being ||x||_w21 the  l21 norm weighting by zetas solution is returned on x0"""

    if not x0.flags.f_contiguous:
        raise ValueError('X0 should be a Fortran array.')

    if not y.flags.f_contiguous:
        raise ValueError('y should be a Fortran array.')

    ext_max_iter = int(ext_stop['max_iter'] + 1)
    n_groups = ixs.shape[0] - 1
    r = y.copy(order='F')
    r -= Z.matvec(x0) # init total residual

    dx = np.empty_like(x0, order='F')
    x1 = x0.copy(order='F')

    for j in range(1, ext_max_iter):

        for i in range(n_groups):

            a, b = ixs[i], ixs[i+1]
            xi = x1[a:b] # view
            Zi = Z.sub(a, b)
            r += Zi.matvec(xi)# partial residual
            
            if np.linalg.norm(Zi.rmatvec(r)) > zetas[i]:

                _ = _kapg(
                    Zi,
                    r,
                    xi,
                    t=int_ts[i],
                    alpha=zeta,
                    beta=zetas[i],
                    stop=int_stop
                )
                r -= Zi.matvec(xi) # total residual
            else:
                xi.fill(0)

        # stopping criteria
        dx[:] = x1 - x0
        if np.linalg.norm(dx) <= np.linalg.norm(x0)*ext_stop['rtol'] + ext_stop['atol']:
            break

        # updates
        x0[:] = x1

    else:
        print('BCD: Max iter reached.')
    
    return j

def kapg(
        Z: KronArray,
        b: np.ndarray,
        x0: np.ndarray,
        t: float,
        alpha: float,
        beta: float,
        stop: Stopping = {'max_iter': 1e4, 'rtol': 1e-3, 'atol': 1e-8},
    ):
    
    """Accelerated proximal gradient method for optimize the function: (1/2)*||Ax-b||_2^2 + (alpha/2)*||x||_2^2 + beta*||x||_2
    solution is returned on x0
    t is convergence rate
    """
    y0 = np.empty_like(x0, order='F')
    r = np.empty_like(b, order='F')
    grad_f = np.empty_like(x0, order='F')
    x1 = np.empty_like(x0, order='F')
    dx = np.empty_like(x0, order='F')
    aux_p = np.empty(Z.p, order='F')
    return apg(Z.A, Z.B, b, x0, t, alpha, beta, stop['max_iter'], stop['rtol'], stop['atol'], y0, r, grad_f, x1, dx, aux_p)

def _kapg(
        Z: KronArray,
        b: np.ndarray,
        x0: np.ndarray,
        t: float,
        alpha: float,
        beta: float,
        stop: Stopping = {'max_iter': 1e4, 'rtol': 1e-3, 'atol': 1e-8},
    ):
    """Accelerated proximal gradient method for optimize the function: (1/2)*||Ax-b||_2^2 + (alpha/2)*||x||_2^2 + beta*||x||_2 solution is returned on x0. t is convergence rate
    """
    max_iter = int(stop['max_iter'] + 1)
    y1 = x0.copy(order='F')

    r = b.copy(order='F')
    r -= Z.matvec(x0)

    grad_f = np.empty_like(x0, order='F')
    x1 = np.empty_like(x0, order='F')
    dx = np.empty_like(x0, order='F')
    grad_f = np.empty_like(x0, order='F')
    v = np.empty_like(x0, order='F')

    for k in range(1, max_iter):
        
        grad_f[:] = - Z.rmatvec(r) + alpha * y1
        v[:] = y1 - t * grad_f
        x1[:] = np.maximum(0, 1.-t*beta/np.linalg.norm(v)) * v 
        dx[:] = x1 - x0
        if np.linalg.norm(dx) <= np.linalg.norm(x0)*stop['rtol'] + stop['atol']:
            break

        # updates
        r -= Z.matvec(dx)
        y1[:]= x1 + k/(k+3.) * dx # nesterov's acceleration
        x0[:] = x1


    else:
        print('PGM: Max iter reached.')

    return k