# cython: profile=False, cdivision=True, boundscheck=False, wraparound = False, language_level=3str

from libc.stdio cimport printf
from libc.math cimport fabs
from scipy.linalg.cython_blas cimport dnrm2, dgemv, dcopy, dscal, daxpy, dger, dgemm
cimport cython

cdef inline double fmax(double x, double y) noexcept nogil:
    if x > y:
        return x
    return y


cdef inline void matvec(
        int* M, 
        int* N, 
        double* ALPHA_0, 
        double* A, 
        double* X,
        int* INC,
        double* BETA,
        double* Y,
        int* L,
        double* ALPHA_1, 
        double* B,
        double* C
    ) noexcept nogil:

    dgemv('n', M, N, ALPHA_0, A, M, X, INC, BETA, Y, INC) # Y[:] = ALPHA_0*np.dot(A, X) + BETA*Y
    dger(M, L, ALPHA_1, Y, INC, B, INC, C, M) # C = ALPHA_1*np.outer(Y, B) + C
    

cdef inline void rmatvec(
        int* M, 
        int* N, 
        double* ALPHA_0, 
        double* X, 
        double* B,
        int* INC,
        double* BETA_0,
        double* Y,
        int* L, 
        double* ALPHA_1, 
        double* A, 
        double* BETA_1,
        double* C ,
    ) noexcept nogil:
    dgemv('n', M, N, ALPHA_0, X, M, B, INC, BETA_0, Y, INC) # Y[:] = ALPHA_0*np.dot(X, B) + BETA_0*Y
    dgemv('t', M, L, ALPHA_1, A, M, Y, INC, BETA_1, C, INC) # C[:] = ALPHA_1*np.dot(A.T, Y) + BETA_1*C


cpdef int apg(
        double[::1, :] X2, # A
        double[::1, :] X1, # D
        double[::1] b,
        double[::1] x0,
        double t,
        double alpha,
        double beta,
        int max_iter,
        double tol,
        double[::1] y0, # reserved memory 
        double[::1] r, # reserved memory 
        double[::1] grad_f, # reserved memory
        double[::1] x1, # reserved memory
        double[::1] dx, # reserved memory
        double[::1] aux_p, # reserved memory
    ) noexcept nogil:
    
    # Constant definitions
    cdef int p = X2.shape[0]
    cdef int q = X2.shape[1]
    cdef int n = X1.shape[0]
    cdef int k
    cdef int max_iter_1 = max_iter + 1
    cdef int CONST_INT_1 = 1
    cdef double CONST_DOUBLE_1 = 1.
    cdef double CONST_DOUBLE_m1 = -1.
    cdef double CONST_DOUBLE_0 = 0.
    cdef double CONST_DOUBLE_3 = 3.
    cdef int N = r.shape[0]
    cdef double neg_t = -t
    cdef double s
    cdef double h

    # Arrays allocations in bcd
    cdef double* v = &grad_f[0]
    cdef double* y1 = v
    
    dcopy(&q, &x0[0], &CONST_INT_1, &y0[0], &CONST_INT_1) # y0[:] = x0 

    dcopy(&N, &b[0], &CONST_INT_1, &r[0], &CONST_INT_1) # r[:] = b
    matvec(&p, &q, &CONST_DOUBLE_1, &X2[0,0], &x0[0], &CONST_INT_1, &CONST_DOUBLE_0, &aux_p[0], &n, &CONST_DOUBLE_m1, &X1[0,0], &r[0]) # r -= A.matvec(x0)

    for k in range(1, max_iter_1):
        # gradient of f
        # grad_f[:] = y0
        # grad_f *= alpha
        # grad_f -= A.rmatvec(r)
        dcopy(&q, &y0[0], &CONST_INT_1, &grad_f[0], &CONST_INT_1) 
        rmatvec(&p, &n, &CONST_DOUBLE_1, &r[0], &X1[0,0], &CONST_INT_1, &CONST_DOUBLE_0, &aux_p[0], &q, &CONST_DOUBLE_m1, &X2[0,0], &alpha, &grad_f[0])

        # gradient step (view of grad_f)
        # v *= -t
        # v += y0
        dscal(&q, &neg_t, v, &CONST_INT_1)
        daxpy(&q, &CONST_DOUBLE_1, &y0[0], &CONST_INT_1, v, &CONST_INT_1)

        # proximal of ||x||_2 (view of v)
        # s = np.maximum(0, 1.-t*beta/np.linalg.norm(v))
        # x1[:] = v 
        # x1 *= s
        s = fmax(CONST_DOUBLE_0, CONST_DOUBLE_1 + (neg_t * beta) / dnrm2(&q, v, &CONST_INT_1))
        dcopy(&q, v, &CONST_INT_1, &x1[0], &CONST_INT_1) 
        dscal(&q, &s, &x1[0], &CONST_INT_1)

        # stopping criteria
        # dx[:] = x1
        # dx -= x0
        dcopy(&q, &x1[0], &CONST_INT_1, &dx[0], &CONST_INT_1) 
        daxpy(&q, &CONST_DOUBLE_m1, &x0[0], &CONST_INT_1, &dx[0], &CONST_INT_1)
        # if np.linalg.norm(dx)/t < tol: # norm of generalized gradient
            # break
        if dnrm2(&q, &dx[0], &CONST_INT_1) <= dnrm2(&q, &x0[0], &CONST_INT_1) * tol:
            break
        # printf('%f\n', dnrm2(&q, &dx[0], &CONST_INT_1))


        # nesterov's acceleration
        # y1[:]= dx
        # y1 *= k/(k+3.) 
        # y1 += x1
        h = k / (k + CONST_DOUBLE_3)
        dcopy(&q, &dx[0], &CONST_INT_1, y1, &CONST_INT_1) 
        dscal(&q, &h, y1, &CONST_INT_1)
        daxpy(&q, &CONST_DOUBLE_1, &x1[0], &CONST_INT_1, y1, &CONST_INT_1)

        # updates
        # x0[:] = x1
        # y0[:] = y1
        # r -= A.matvec(dx) 
        dcopy(&q, &x1[0], &CONST_INT_1, &x0[0], &CONST_INT_1) 
        dcopy(&q, y1, &CONST_INT_1, &y0[0], &CONST_INT_1) 
        matvec(&p, &q, &CONST_DOUBLE_1, &X2[0,0], &dx[0], &CONST_INT_1, &CONST_DOUBLE_0, &aux_p[0], &n, &CONST_DOUBLE_m1, &X1[0,0], &r[0])

    else:
        printf('APG: Max iter reached.\n')

    return k

cpdef int bcd(
        double[::1, :] X2,
        double[::1, :] X1,
        double[::1] y,
        double[::1] x0,
        long long[::1] ixs,
        double zeta,
        double[::1] zetas,
        double[::1] int_ts,
        int ext_max_iter,
        double ext_tol,
        int int_max_iter,
        double int_tol,
        double[::1] r, # reserved memory
        double[::1] x1, # reserved memory
        double[::1] dx, # reserved memory
        double[::1,:] aux_pk, # reserved memory
        double[::1] aux_p, # reserved memory
        double[::1] aux_q, # reserved memory 
        double[::1] int_y0, # apg reserved memory 
        double[::1] int_r, # apg reserved memory 
        double[::1] int_grad_f, # apg reserved memory
        double[::1] int_x1, # apg reserved memory
        double[::1] int_dx, # apg reserved memory
        double[::1] int_aux_p, # apg reserved memory
    ) noexcept nogil:

    # Constant definitions
    cdef int ext_max_iter_1 = ext_max_iter + 1
    cdef int j, aux
    cdef int CONST_INT_1 = 1
    cdef double CONST_DOUBLE_1 = 1.
    cdef double CONST_DOUBLE_m1 = -1.
    cdef double CONST_DOUBLE_0 = 0.
    cdef int N = y.shape[0]# pxn
    cdef int p = X2.shape[0]
    cdef int q = X2.shape[1]
    cdef int n = X1.shape[0]
    cdef int k = X1.shape[1]
    cdef int qk = q*k # qxk
    cdef int n_groups = ixs.shape[0] - 1 # amount of groups
    cdef Py_ssize_t i
    cdef Py_ssize_t a, b, c, d, e
    cdef Py_ssize_t _amax 
    cdef int f # size of group
    cdef double[::1, :] X2i,
    cdef double[::1, :] X1i,
    cdef double[::1] xi,
    cdef double[::1] int_y0i,
    cdef double[::1] int_grad_fi,
    cdef double[::1] int_x1i,
    cdef double[::1] int_dxi
    # r[:] = y
    # r[:] -= X2@X0@X1.T
    dcopy(&N, &y[0], &CONST_INT_1, &r[0], &CONST_INT_1)
    dgemm('n', 'n', &p, &k, &q, &CONST_DOUBLE_1, &X2[0,0], &p, &x0[0], &q, &CONST_DOUBLE_0, &aux_pk[0,0], &p)
    dgemm('n', 't', &p, &n, &k, &CONST_DOUBLE_m1, &aux_pk[0,0], &p, &X1[0,0], &n, &CONST_DOUBLE_1, &r[0], &p)
    
    # # r1[:] = r0
    # dcopy(&N, &r0[0], &CONST_INT_1, &r1[0], &CONST_INT_1)

    # x1[:] = x0
    dcopy(&qk, &x0[0], &CONST_INT_1, &x1[0], &CONST_INT_1)

    for j in range(1, ext_max_iter_1):

        for i in range(n_groups):

            a = ixs[i]
            b = ixs[i+1]
            f = b-a # group size
            
            xi = x1[a:b] # view
            c = a // q # X1 column
            X1i = X1[:,c:c+1]
            d = a % q # X2 first column
            e = d + f # X2 last column
            X2i = X2[:,d:e]
            
            # r += Zi.matvec(xi)
            matvec(&p, &f, &CONST_DOUBLE_1, &X2i[0,0], &xi[0], &CONST_INT_1, &CONST_DOUBLE_0, &aux_p[0], &n, &CONST_DOUBLE_1, &X1i[0,0], &r[0])
            
            # Zi.rmatvec(r1)
            rmatvec(&p, &n, &CONST_DOUBLE_1, &r[0], &X1i[0,0], &CONST_INT_1, &CONST_DOUBLE_0, &aux_p[0], &f, &CONST_DOUBLE_1, &X2i[0,0], &CONST_DOUBLE_0, &aux_q[0])

            # if np.linalg.norm(Zi.rmatvec(r1)) > zetas[i]:
            if dnrm2(&f, &aux_q[0], &CONST_INT_1) > zetas[i]:
                int_y0i = int_y0[:f]
                int_grad_fi = int_grad_f[:f]
                int_x1i = int_x1[:f]
                int_dxi = int_dx[:f]

                # aux = f*p
                # printf("%f\t", dnrm2(&aux, &X2i[0,0], &CONST_INT_1))
                # printf("%f\t", dnrm2(&n, &X1i[0,0], &CONST_INT_1))
                # printf("%f\t", dnrm2(&N, &r1[0], &CONST_INT_1))
                # printf("%f\t", dnrm2(&f, &xi[0], &CONST_INT_1))
                # printf("%f\t", int_ts[i])
                # printf("%f\t", zeta)
                # printf("%f\t", zetas[i])
                # printf("%d\t", int_max_iter)
                # printf("%f\n", int_tol)

                apg(X2i, X1i, r, xi, int_ts[i], zeta, zetas[i], int_max_iter, int_tol, int_y0i, int_r, int_grad_fi, int_x1i, int_dxi, int_aux_p)
                
                # r -= Zi.matvec(xi)
                matvec(&p, &f, &CONST_DOUBLE_1, &X2i[0,0], &xi[0], &CONST_INT_1, &CONST_DOUBLE_0, &aux_p[0], &n, &CONST_DOUBLE_m1, &X1i[0,0], &r[0])

            else:
                # xi.fill(0)
                dscal(&f, &CONST_DOUBLE_0, &xi[0], &CONST_INT_1)

        # dx[:] = x1
        # dx -= x0 
        dcopy(&qk, &x1[0], &CONST_INT_1, &dx[0], &CONST_INT_1)
        daxpy(&qk, &CONST_DOUBLE_m1, &x0[0], &CONST_INT_1, &dx[0], &CONST_INT_1)
        # _amax = idamax(&N, &dr[0], &CONST_INT_1)-1 # BLAS is one-based index

        if dnrm2(&qk, &dx[0], &CONST_INT_1) <= dnrm2(&qk, &x0[0], &CONST_INT_1) * ext_tol:
            break

        # if fabs(dr[_amax]) < ext_tol:
            # break

        # x0[:] = x1
        dcopy(&qk, &x1[0], &CONST_INT_1, &x0[0], &CONST_INT_1)
    
    else:
        printf('BCD: Max iter reached.\n')

    return j