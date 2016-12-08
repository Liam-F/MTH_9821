from __future__ import division
from Option import *
from NMF_Binomial_Tree import *
import NMF_Heat_PDE as pde
import Finite_Difference_Pricer as fd

class Option_Dis_Div(Option):
    def __init__(self, S0, K, T, sigma, cp, ae, div_dict):
        Option.__init__(self, S0, K, T, 0, sigma, cp, ae)
        self.dividend = div_dict # A dictionary saving all the time spot of dividend and dividend ratio (proportional dividend are written in strings, fixed dividends in numbers)

def Discretization_div(opt_div, r):
    # Assume the option pays only 1 dividend
    td = opt_div.dividend.keys()[0]
    d = opt_div.dividend[td]
    S0, K, T, q, sigma = opt_div.spot, opt_div.strike, opt_div.maturity, 0, opt_div.vol
    x_left = np.log(S0 / K) + (r - q - sigma ** 2 / 2) * T - 3 * sigma * np.sqrt(T)
    x_right = np.log(S0 / K) + (r - q - sigma ** 2 / 2) * T + 3 * sigma * np.sqrt(T)
    tau_final = sigma ** 2 * T / 2
    tau_div = (T - td) * sigma ** 2 / 2
    # Ex dividend value
    # x_left += np.log(1 - d)
    # x_right += np.log(1 - d)
    return x_left, x_right, tau_final, tau_div, d

def finite_diff_discrete_div(opt_div, r, M_1=16, PDE_Solver="Crank_Nicolson", Linear_Solver='LU', Greek=False):
    S0, K, T, q, sigma = opt_div.spot, opt_div.strike, opt_div.maturity, 0, opt_div.vol
    a = (r - q) / sigma ** 2 - 1 / 2
    b = ((r - q) / sigma ** 2 + 1 / 2) ** 2 + 2 * q / sigma ** 2
    x_left_temp, x_right_temp, tau_final, tau_div, d = Discretization_div(opt_div, r)
    x_compute = np.log(S0/K) + np.log(1 - d)

    # First solve the PDE from tau = 0 to tau_div
    def f(x):
        return K * np.exp(a * x) * (np.exp(x) - 1) * int(x > 0)

    def g_left(tau):
        return 0

    def g_right(tau):
        return K * np.exp(a * x_right + b * tau) * (- np.exp(2 * r * tau / sigma ** 2) + np.exp(x_right - 2 * q * tau / sigma ** 2))

    alpha_1 = 0.4
    if PDE_Solver == "Crank_Nicolson":
        alpha_1 = 4

    # N, alpha_1 = fd.mesh(x_left, x_right, tau_div, M_1, alpha_1)
    dtau_1 = tau_div / M_1
    dx = np.sqrt(dtau_1 / alpha_1)
    N_left = np.ceil((x_compute - x_left_temp) / dx)
    N_right = np.ceil((x_right_temp - x_compute) / dx)
    N = int(N_left + N_right)
    x_left = x_compute - N_left * dx
    x_right = x_compute + N_right * dx

    if PDE_Solver == 'Forward_Euler':
        u_approx_grid, x_knot, tau_knot = pde.PDE_Forward_Euler(x_left, x_right, tau_div, f, g_left, g_right, M_1, N)
    elif PDE_Solver == "Backward_Euler":
        if Linear_Solver == 'SOR':
            u_approx_grid, x_knot, tau_knot = pde.PDE_Backward_Euler(x_left, x_right, tau_div, f, g_left, g_right, M_1, N, solver='SOR')
        elif Linear_Solver == 'LU':
            u_approx_grid, x_knot, tau_knot = pde.PDE_Backward_Euler(x_left, x_right, tau_div, f, g_left, g_right, M_1, N, solver='LU')
    elif PDE_Solver == "Crank_Nicolson":
        if Linear_Solver == 'SOR':
            u_approx_grid, x_knot, tau_knot = pde.PDE_Crank_Nicolson(x_left, x_right, tau_div, f, g_left, g_right, M_1, N, solver='SOR')
        elif Linear_Solver == 'LU':
            u_approx_grid, x_knot, tau_knot = pde.PDE_Crank_Nicolson(x_left, x_right, tau_div, f, g_left, g_right, M_1, N, solver='LU')
    print u_approx_grid
    u_approx_plus = u_approx_grid[-1, :] # This gives the boundry condition for tau_div -> tau_final

    # Then solve the PDE for the rest part of the domain
    # shift back to original values, denoted as "new values"
    x_left -= np.log(1 - d)
    x_right -= np.log(1 - d)
    dx = (x_right - x_left) / N
    dtau_2_temp = alpha_1 * dx ** 2
    M_2 = int(np.ceil((tau_final - tau_div) / dtau_2_temp))
    dtau_2 = (tau_final - tau_div) / M_2
    alpha_2 = dtau_2 / (dx ** 2) # Generate a new mesh with same grid on x, thinner grid on tau

    # print M_2, alpha_2, N, x_left + np.log(1 - d), x_right + np.log(1 - d), x_left, x_right, x_left, tau_div, tau_div / M_1, dtau_2, dx

    u_tau_div = {}
    x_knot_new = np.linspace(x_left, x_right, N + 1)
    for i in xrange(len(x_knot_new)):
        u_tau_div[x_knot_new[i]] = u_approx_plus[i]

    def f_2(x):
        return u_tau_div[x]

    def g_left(tau):
        return 0

    def g_right(tau):
        return K * np.exp(a * x_right + b * tau) * (- np.exp(2 * r * tau / sigma ** 2) + np.exp(x_right - 2 * q * tau / sigma ** 2))

    if PDE_Solver == 'Forward_Euler':
        u_approx_grid, x_knot, tau_knot = pde.PDE_Forward_Euler(x_left, x_right, tau_final - tau_div, f_2, g_left, g_right, M_2, N)
    elif PDE_Solver == "Backward_Euler":
        if Linear_Solver == 'SOR':
            u_approx_grid, x_knot, tau_knot = pde.PDE_Backward_Euler(x_left, x_right, tau_final - tau_div, f_2, g_left, g_right, M_2, N, solver='SOR')
        elif Linear_Solver == 'LU':
            u_approx_grid, x_knot, tau_knot = pde.PDE_Backward_Euler(x_left, x_right, tau_final - tau_div, f_2, g_left, g_right, M_2, N, solver='LU')
    elif PDE_Solver == "Crank_Nicolson":
        if Linear_Solver == 'SOR':
            u_approx_grid, x_knot, tau_knot = pde.PDE_Crank_Nicolson(x_left, x_right, tau_final - tau_div, f_2, g_left, g_right, M_2, N, solver='SOR')
        elif Linear_Solver == 'LU':
            u_approx_grid, x_knot, tau_knot = pde.PDE_Crank_Nicolson(x_left, x_right, tau_final - tau_div, f_2, g_left, g_right, M_2, N, solver='LU')
    print u_approx_grid
    u_approx = u_approx_grid[-1, :]

    def linear_interp_1(S0, K):
        x_compute = np.log(S0 / K)
        dx = (x_right - x_left) / N
        i = int(np.floor((x_compute - x_left) / dx))
        x_lo = x_knot[i]
        x_hi = x_knot[i + 1]
        S_lo = K * np.exp(x_lo)
        S_hi = K * np.exp(x_hi)
        V_lo = np.exp(-a * x_lo - b * tau_final) * u_approx[i]
        V_hi = np.exp(-a * x_hi - b * tau_final) * u_approx[i + 1]
        V_FD = (V_lo * (S_hi - S0) + V_hi * (S0 - S_lo)) / (S_hi - S_lo)
        return V_FD

    def linear_interp_2(S0, K):
        x_compute = np.log(S0 / K)
        dx = (x_right - x_left) / N
        i = np.floor((x_compute - x_left) / dx)
        x_lo = x_knot[i]
        x_hi = x_knot[i + 1]
        u_compute = (u_approx[i] * (x_hi - x_compute) + u_approx[i + 1] * (x_compute - x_lo)) / (x_hi - x_lo)
        V_FD = np.exp(-a * x_compute - b * tau_final) * u_compute
        return V_FD

    # V_FD = linear_interp_1(S0, K)
    u_compute = u_approx[int(N_left)]
    V_FD = np.exp(-a * x_compute - b * tau_final) * u_compute
    if not Greek:
        return V_FD
    else:
        # x_compute = np.log(S0 / K)
        # dx = (x_right - x_left) / N
        # i = int(np.floor((x_compute - x_left) / dx))
        i = int(N_left)
        x_lo = x_knot[i - 1]
        x_hi = x_knot[i + 1]
        S_lo = K * np.exp(x_lo)
        S_hi = K * np.exp(x_hi)

        V_lo = np.exp(-a * x_lo - b * tau_final) * u_approx[i - 1]
        V_hi = np.exp(-a * x_hi - b * tau_final) * u_approx[i + 1]
        Delta = (V_hi - V_lo) / (S_hi - S_lo) # Delta Central

        Gamma = ((S0- S_lo) * V_hi - (S_hi - S_lo) * V_FD + (S_hi - S0) * V_lo) / ((S0 - S_lo) * (S_hi - S0) * (S_hi - S_lo) / 2) # Gamma Central

        d_tau = tau_knot[-1] - tau_knot[-2]
        dt = - 2 * d_tau / sigma ** 2
        V_FD_pre = np.exp(-a * x_lo - b * (tau_final - d_tau)) * u_approx_grid[-2, i]
        Theta = (V_FD - V_FD_pre) / dt
        # print u_compute, V_FD, Delta, Gamma, Theta
        return V_FD, Delta, Gamma, Theta

if __name__ == "__main__":
    div_dict = {5/12: 0.02}
    euc = Option_Dis_Div(S0=52, K=50, T=1, sigma=0.3, cp='C', ae='EU', div_dict=div_dict)
    r = 0.03
    # for M_1 in [4, 16, 64, 256]:
    for M_1 in [4]:
    #     finite_diff_discrete_div(euc, r, M_1, PDE_Solver='Forward_Euler', Greek=True)
        finite_diff_discrete_div(euc, r, M_1, PDE_Solver='Forward_Euler', Greek=True)
