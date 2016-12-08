from __future__ import division
import numpy as np
import NMF_linear_solve as lis
import NMF_Iter_solve as its
from Heat_PDE_settings import *
from time import *
np.set_printoptions(precision=12, linewidth=300)

def PDE_Forward_Euler(x_left, x_right, tau_final, f, g_left, g_right, M, N):
    '''
    Forward Euler solution for heat pde u_tau = u_xx
    :param x0: the point of interest
    :param x_left: left boundary of the interval
    :param x_right: right boundary of the interval
    :param tau_final: upper boundary of the interval
    :param f: lower boundary condition
    :param g_left: left boundary condition
    :param g_right: right boundary condition
    :return: the discrete nodes x and solution at (x_i, tau_final) for each x_i in x
    '''
    # Discretization settings
    dx = (x_right - x_left) / N
    dtau = tau_final / M
    alpha = dtau / (dx ** 2)
    # print alpha
    x = np.linspace(x_left, x_right, N+1)
    tau = np.linspace(0, tau_final, M+1)
    u_approx = np.zeros([M+1, N+1])
    # Plug in boundary conditions
    # u_approx[0, :] = np.apply_along_axis(f, 0, x)
    for c in xrange(N+1):
        u_approx[0, c] = f(x[c])
    u_approx[:, 0] = np.apply_along_axis(g_left, 0, tau)
    u_approx[1:, N] = np.apply_along_axis(g_right, 0, tau[1:])

    # Execute Forward Euler
    for m in xrange(1, M+1):
        for n in xrange(1, N):
            u_approx[m, n] = alpha * u_approx[m-1, n-1] - (2*alpha - 1) * u_approx[m-1, n] + alpha * u_approx[m-1, n+1]
    return u_approx, x, tau


def PDE_Backward_Euler(x_left, x_right, tau_final, f, g_left, g_right, M, N, solver='LU'):
    '''
    Forward Euler solution for heat pde u_tau = u_xx
    :param x_left: left boundary of the interval
    :param x_right: right boundary of the interval
    :param tau_final: upper boundary of the interval
    :param f: lower boundary condition
    :param g_left: left boundary condition
    :param g_right: right boundary condition
    :return: the discrete nodes x and solution at (x_i, tau_final) for each x_i in x
    '''
    # Discretization settings
    dx = (x_right - x_left) / N
    dtau = tau_final / M
    alpha = dtau / (dx ** 2)
    x = np.linspace(x_left, x_right, N+1)
    tau = np.linspace(0, tau_final, M+1)
    u_approx = np.zeros([M+1, N+1])
    # Plug in boundary conditions
    # u_approx[0, :] = np.apply_along_axis(f, 0, x)
    for c in xrange(N+1):
        u_approx[0, c] = f(x[c])
    u_approx[:, 0] = np.apply_along_axis(g_left, 0, tau)
    u_approx[:, N] = np.apply_along_axis(g_right, 0, tau)

    # Execute Backward Euler
    # Initialize the tri-diagonal matrix A
    A = np.zeros([N - 1, N - 1])
    A[0, 0], A[0, 1] = 1 + 2 * alpha, - alpha
    for i in xrange(1, N - 2):
        A[i, i - 1], A[i, i], A[i, i + 1] = -alpha, 1 + 2 * alpha, -alpha
    A[N - 2, N - 3], A[N - 2, N - 2] = -alpha, 1 + 2 * alpha
    if solver == 'LU':
        # LU decomposition
        [L, U] = lis.lu_no_pivoting_banded(A, 2)
        # Solve linear system
        for m in xrange(1, M+1):
            b = u_approx[m-1, 1:N].copy()
            b[0] += u_approx[m, 0] * alpha
            b[-1] += u_approx[m, -1] * alpha
            b = np.reshape(b, (N-1,1))
            y = lis.forward_subst_banded(L, 2, b)
            u_approx[m, 1:N] = np.reshape(lis.backward_subst_banded(U, 2, y), N-1)

    elif solver == "SOR":
        for m in xrange(1, M+1):
            b = u_approx[m - 1, 1:N].copy()
            b[0] += u_approx[m, 0] * alpha
            b[-1] += u_approx[m, -1] * alpha
            b = np.reshape(b, (N - 1, 1))
            u_approx[m, 1:N] = np.reshape(its.SOR_iter_banded(A, 2, b, np.reshape(u_approx[m-1, 1:N], (N-1, 1)), tol=10**(-6), omega=1.2), N - 1)
    return u_approx, x, tau


def PDE_Crank_Nicolson(x_left, x_right, tau_final, f, g_left, g_right, M, N, solver='LU'):
    '''
    Forward Euler solution for heat pde u_tau = u_xx
    :param x_left: left boundary of the interval
    :param x_right: right boundary of the interval
    :param tau_final: upper boundary of the interval
    :param f: lower boundary condition
    :param g_left: left boundary condition
    :param g_right: right boundary condition
    :return: the discrete nodes x and solution at (x_i, tau_final) for each x_i in x
    '''
    # Discretization settings
    dx = (x_right - x_left) / N
    dtau = tau_final / M
    alpha = dtau / (dx ** 2)
    x = np.linspace(x_left, x_right, N+1)
    tau = np.linspace(0, tau_final, M+1)
    u_approx = np.zeros([M+1, N+1])
    # Plug in boundary conditions
    # u_approx[0, :] = np.apply_along_axis(f, 0, x)
    for c in xrange(N+1):
        u_approx[0, c] = f(x[c])
    u_approx[:, 0] = np.apply_along_axis(g_left, 0, tau)
    u_approx[1:, N] = np.apply_along_axis(g_right, 0, tau[1:])

    # Execute Crank Nicolson
    # Initialize the tri-diagonal matrix A
    A = np.zeros([N-1, N-1])
    A[0, 0], A[0, 1] = 1 + alpha, - alpha / 2
    for i in xrange(1, N-2):
        A[i, i - 1], A[i, i], A[i, i + 1] = -alpha / 2, 1 + alpha, -alpha / 2
    A[N-2, N-3], A[N-2, N-2] = -alpha / 2, 1 + alpha
    # Initialize tri-diagonal matrix B
    B = np.zeros([N-1, N-1])
    B[0, 0], B[0, 1] = 1 - alpha, alpha / 2
    for i in xrange(1, N-2):
        B[i, i-1], B[i, i], B[i, i+1] = alpha / 2, 1 - alpha, alpha / 2
    B[N-2, N-3], B[N-2, N-2] = alpha / 2, 1 - alpha
    # LU decomposition
    if solver == 'LU':
        [L, U] = lis.lu_no_pivoting_banded(A, 2)
        # Solve linear system
        for m in xrange(1, M+1):
            b = np.dot(B, np.reshape(u_approx[m-1, 1:N], (N-1, 1)))
            b[0] += (u_approx[m, 0] + u_approx[m-1, 0]) * alpha / 2
            b[-1] += (u_approx[m, -1] + u_approx[m-1, -1]) * alpha / 2

            y = lis.forward_subst_banded(L, 2, b)
            u_approx[m, 1:N] = np.reshape(lis.backward_subst_banded(U, 2, y), N-1)

    elif solver == 'SOR':
        for m in xrange(1, M + 1):
            b = np.dot(B, np.reshape(u_approx[m - 1, 1:N], (N - 1, 1)))
            b[0] += (u_approx[m, 0] + u_approx[m - 1, 0]) * alpha / 2
            b[-1] += (u_approx[m, -1] + u_approx[m - 1, -1]) * alpha / 2

            # u_approx[m, 1:N] = np.reshape(its.SOR_iter_banded(A, 2, b, np.reshape(u_approx[m-1, 1:N], (N-1, 1)), tol=10 ** (-6), omega=1.2), N - 1)
            u_approx[m, 1:N] = np.reshape(its.SOR_iter(A, b, np.reshape(u_approx[m-1, 1:N], (N-1, 1)), tol=10 ** (-6), res_cri = 0, omega=1.2), N - 1)
    return u_approx, x, tau


def max_pointwise_error(u_approx, u_exact):
    N = u_approx.shape[0]
    return max(abs(u_approx[1:N] - u_exact[1:N]))


def RMS_error(u_approx, u_exact):
    r_e = abs(u_approx - u_exact) / abs(u_exact) # relative error
    return np.sqrt(np.mean(r_e ** 2))


if __name__ == "__main__":

    M, N = 8, 8
    # print PDE_Forward_Euler(-2, 2, 1, f, g_left, g_right, M, N)[1]
    print PDE_Backward_Euler(-2, 2, 1, f, g_left, g_right, M, N, solver="SOR")[0]
    # print PDE_Crank_Nicolson(-2, 2, 1, f, g_left, g_right, M, N, solver='SOR')[0]
    #
    # alpha = 0.125
    # if alpha == 0.125:
    #     N_lst = [4, 8, 16, 32]
    # elif alpha == 0.5:
    #     N_lst = [8, 16, 32, 64]
    # else:
    #     N_lst = [16, 32, 64, 128]
    #
    # start = time()
    # for N in N_lst:
    #     M = int((N ** 2) / (16 * alpha))
    #     # u_approx, x_knot, tau_knot = PDE_Forward_Euler(-2, 2, 1, f, g_left, g_right, M, N)
    #     # u_approx, x_knot, tau_knot = PDE_Backward_Euler(-2, 2, 1, f, g_left, g_right, M, N, solver='SOR')
    #     u_approx, x_knot, tau_knot = PDE_Crank_Nicolson(-2, 2, 1, f, g_left, g_right, M, N, solver='SOR')
    #     u_approx = u_approx[-1, :]
    #     u_exa = np.apply_along_axis(u_exact_final, 0, x_knot)
    #     cur = time()
    #     print max_pointwise_error(u_approx, u_exa), RMS_error(u_approx, u_exa)
    #     start = cur
