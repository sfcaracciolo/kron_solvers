import numpy as np 
import scipy as sp 
from .solvers import apg, bcd
from kron_groupper import Groupper
from regularization_tools import AbstractRegularizer 

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
        
    def norms(self, ord: str | int ='fro') -> float:
        for _, (a, b) in self.gr.it():
            Zi = self.Z.sub(a, b)
            yield Zi.norm(ord)

    def xnorms(self, x: np.ndarray, ord: str | int ='fro') -> float:
        for _, (a, b) in self.gr.it():
            xi = x[a:b]
            yield np.linalg.norm(xi, ord=ord)

    def lipschitz(self, zeta: float) -> float: 
        for norm in self.norms(ord=2):
            yield norm ** 2 + zeta
    
    def lambda_max(self, y: np.ndarray, alpha: float):
        N = y.size
        dcs = np.fromiter(self.discard_conds(y), dtype=np.float64)
        w = np.max(dcs/self.etas)
        return w/(N*alpha)
    
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
        N = y.size
        R = np.empty(n_lambdas)
        for i in range(n_lambdas):
            x = X[i].reshape(-1, order='F')
            R[i] = np.linalg.norm(self.Z.matvec(x) - y, ord=2)**2/(2*N)
        return R
    
    def lambdaspace(self, l_max: float, epsilon: float = 1e-3, num: int = 100):
        return super().lambdaspace(start=l_max, end=l_max*epsilon, num=num)
    
    def sols(self, X: np.ndarray):
        return np.tensordot(X, self.Z.B, axes=(2, 1))
    
    def solve(
            self, 
            y: np.ndarray, 
            lambdas: np.ndarray, 
            alpha: float=1., 
            int_max_iter=1e4, 
            int_tol=1e-3,
            ext_max_iter=1e4, 
            ext_tol=1e-3, 
        ):

        if lambdas[0] < lambdas[-1]:
            raise ValueError('Lambdas must be in decreasing order.')
        
        n_groups, N = self.gr.get_n_groups(), y.size
        norm_groups = np.fromiter(self.norms(ord=2), dtype=np.float64)

        X = np.zeros((lambdas.size, self.Z.q, self.Z.k), order='F')
        x0 = np.zeros((self.Z.shape[1]), order='F')
        zetas = np.empty(n_groups)
        int_ts = np.empty(n_groups)

        for i, lambd in enumerate(lambdas):
            zeta = (1.-alpha)*lambd*N
            zetas[:] = self.etas*alpha*lambd*N
            int_ts[:] = 1./(norm_groups**2 + zeta)
            n = kbcd(
                self.Z,
                y,
                x0,
                self.gr.ixs,
                zeta,
                zetas,
                int_ts,
                ext_max_iter,
                ext_tol,
                int_max_iter,
                int_tol
            )
            print(f'lambda = {i}\titers {n}/{ext_max_iter:.0f}')
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
        ext_max_iter: int = 1e4,
        ext_tol: float = 1e-3,
        int_max_iter: int = 1e4,
        int_tol: float = 1e-3,
    ):
    """Optimize (1/2)*||Zx-y||_2^2 + (zeta/2)*||x||_2^2 + ||x||_w21 being ||x||_w21 the  l21 norm weighting by zetas
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

    return bcd(Z.A, Z.B, y, x0, ix, zeta, zetas, int_ts, ext_max_iter, ext_tol, int_max_iter, int_tol, r, x1, dx, aux_pk, aux_p, aux_q, int_y0, int_r, int_grad_f, int_x1, int_dx, int_aux_p)

def _kbcd(
        Z: KronArray,
        y: np.ndarray,
        x0: np.ndarray,
        ixs: np.ndarray,
        zeta: float,
        zetas: np.ndarray,
        int_ts: np.ndarray,
        ext_max_iter: int = 1e4,
        ext_tol: float = 1e-3,
        int_max_iter: int = 1e4,
        int_tol: float = 1e-3,
    ):
    """Optimize (1/2)*||Zx-y||_2^2 + (zeta/2)*||x||_2^2 + ||x||_w21 being ||x||_w21 the  l21 norm weighting by zetas
    solution is returned on x0"""

    if not x0.flags.f_contiguous:
        raise ValueError('X0 should be a Fortran array.')

    if not y.flags.f_contiguous:
        raise ValueError('y should be a Fortran array.')

    ext_max_iter = int(ext_max_iter + 1)
    n_groups = ixs.shape[0] - 1
    r = y.copy(order='F')
    r -= Z.matvec(x0) # init total residual

    # dr = np.empty_like(r0, order='F')
    # r1 = r0.copy(order='F')
    dx = np.empty_like(x0, order='F')
    x1 = x0.copy(order='F')

    for j in range(1, ext_max_iter):

        for i in range(n_groups):

            a, b = ixs[i], ixs[i+1]
            xi = x1[a:b] # view
            Zi = Z.sub(a, b)
            r += Zi.matvec(xi)# partial residual
            
            if np.linalg.norm(Zi.rmatvec(r)) > zetas[i]:

                # print(f"{np.linalg.norm(Z._asub(a, b)):.6f}\t{np.linalg.norm(Z._bsub(a)):.6f}\t{np.linalg.norm(r1):.6f}\t{np.linalg.norm(x):.6f}\t{int_ts[i]:.6f}\t{zeta:.6f}\t{zetas[i]:.6f}\t{int_max_iter:.0f}\t{int_tol:.6f}")

                _ = _kapg(
                    Zi,
                    r,
                    xi,
                    t=int_ts[i],
                    alpha=zeta,
                    beta=zetas[i],
                    max_iter=int_max_iter,
                    tol=int_tol
                )
                r -= Zi.matvec(xi) # total residual
            else:
                xi.fill(0)

        # stopping criteria
        dx[:] = x1
        dx -= x0
        if np.linalg.norm(dx) <= np.linalg.norm(x0) * ext_tol:
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
        max_iter: int = 1e4,
        tol: float = 1e-3
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
    return apg(Z.A, Z.B, b, x0, t, alpha, beta, max_iter, tol, y0, r, grad_f, x1, dx, aux_p)

def _kapg(
        Z: KronArray,
        b: np.ndarray,
        x0: np.ndarray,
        t: float,
        alpha: float,
        beta: float,
        max_iter: int = 1e4,
        tol: float = 1e-3
    ):
    
    """Accelerated proximal gradient method for optimize the function: (1/2)*||Ax-b||_2^2 + (alpha/2)*||x||_2^2 + beta*||x||_2
    solution is returned on x0
    t is convergence rate
    """

    max_iter = int(max_iter + 1)
    y0 = x0.copy(order='F')

    r = b.copy(order='F')
    r -= Z.matvec(x0)

    grad_f = np.empty_like(x0, order='F')
    x1 = np.empty_like(x0, order='F')
    dx = np.empty_like(x0, order='F')
    y1 = v = grad_f # views

    for k in range(1, max_iter):
        
        # gradient of f
        grad_f[:] = y0
        grad_f *= alpha
        grad_f -= Z.rmatvec(r)

        # gradient step (view of grad_f)
        v *= -t
        v += y0

        # proximal of ||x||_2 (view of v)
        s = np.maximum(0, 1.-t*beta/np.linalg.norm(v))
        x1[:] = v 
        x1 *= s

        # stopping criteria
        dx[:] = x1
        dx -= x0
        if np.linalg.norm(dx) < np.linalg.norm(x0)*tol:
            break
        # print(f'{np.linalg.norm(dx):.6f}')

        # nesterov's acceleration
        y1[:]= dx
        y1 *= k/(k+3.) 
        y1 += x1

        # print(np.linalg.norm(dx)/np.linalg.norm(x0))
        # updates
        x0[:] = x1
        y0[:] = y1
        r -= Z.matvec(dx) 


    else:
        print('PGM: Max iter reached.')

    return k