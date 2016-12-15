from __future__ import division
import numpy as np
from Option import *
import NMF_Black_Scholes as BS
import NMF_Heat_PDE as pde
from time import *


def Discretization(opt, r, c=3):
    '''
    Generate the discretization domain boundary of a given option
    :param opt: option to be priced
    :param r: interest rate
    :return: x_left, x_right, tau_final
    '''
    S0, K, T, q, sigma = opt.spot, opt.strike, opt.maturity, opt.div_rate, opt.vol
    x_left = np.log(S0 / K) + (r - q - sigma ** 2 / 2) * T - c * sigma * np.sqrt(T)
    x_right = np.log(S0 / K) + (r - q - sigma ** 2 / 2) * T + c * sigma * np.sqrt(T)
    tau_final = sigma ** 2 * T / 2
    return x_left, x_right, tau_final


def mesh(x_left, x_right, tau_final, M, alpha_temp):
    '''
    Generate the grid for heat PDE
    '''
    dtau = tau_final / M
    dx = np.sqrt(dtau / alpha_temp)
    N = np.floor((x_right - x_left) / dx)
    dx = (x_right - x_left) / N
    alpha = dtau / (dx ** 2)
    return int(N), alpha


def boundry_config(opt, r, c=3):
    S0, K, T, q, sigma = opt.spot, opt.strike, opt.maturity, opt.div_rate, opt.vol
    x_left, x_right, tau_final = Discretization(opt, r, c)
    a = (r - q) / sigma ** 2 - 1 / 2
    b = ((r - q) / sigma ** 2 + 1 / 2) ** 2 + 2 * q / sigma ** 2

    if opt.ae == "EU":
        if opt.cp == "P":
            def f(x):
                return K * np.exp(a * x) * (1 - np.exp(x)) * int(x < 0)
            def g_left(tau):
                return K * np.exp(a * x_left + b * tau) * (np.exp(- 2 * r * tau / sigma ** 2) - np.exp(x_left - 2 * q * tau / sigma ** 2))
            def g_right(tau):
                return 0
        elif opt.cp == "C":
            def f(x):
                return K * np.exp(a * x) * (np.exp(x) - 1) * int(x > 0)
            def g_left(tau):
                return 0
            def g_right(tau):
                return K * np.exp(a * x_right + b * tau) * (- np.exp(2 * r * tau / sigma ** 2) + np.exp(x_right - 2 * q * tau / sigma ** 2))

    elif opt.ae == "AM":
        if opt.cp == "P":
            def f(x):
                return K * np.exp(a * x) * (1 - np.exp(x)) * int(x < 0)
            def g_left(tau):
                return K * np.exp(a * x_left + b * tau) * (1 - np.exp(x_left))
                # Here x_left correspond to -3 sigma fluctuation of the Stock price, it is going to be optimal to early exercise the American Put.
            def g_right(tau):
                return 0
        elif opt.cp == "C":
            def f(x):
                return K * np.exp(a * x) * (np.exp(x) - 1) * int(x > 0)
            def g_left(tau):
                return 0
            def g_right(tau):
                return K * np.exp(a * x_right + b * tau) * (np.exp(x_right) - 1)
                # early exercise

    return f, g_left, g_right


def finite_diff(opt, r, M=64, c=3, alpha_temp=0.5, PDE_Solver="Backward_Euler", Linear_Solver='LU', Greek=False, print_grid=False):
    S0, K, T, q, sigma = opt.spot, opt.strike, opt.maturity, opt.div_rate, opt.vol
    a = (r - q) / sigma ** 2 - 1 / 2
    b = ((r - q) / sigma ** 2 + 1 / 2) ** 2 + 2 * q / sigma ** 2
    x_left, x_right, tau_final = Discretization(opt, r, c)
    f, g_left, g_right = boundry_config(opt, r, c)
    N, alpha = mesh(x_left, x_right, tau_final, M, alpha_temp)
    if PDE_Solver == 'Forward_Euler':
        u_approx_grid, x_knot, tau_knot = pde.PDE_Forward_Euler(x_left, x_right, tau_final, f, g_left, g_right, M, N)
    elif PDE_Solver == "Backward_Euler":
        if Linear_Solver == 'SOR':
            u_approx_grid, x_knot, tau_knot = pde.PDE_Backward_Euler(x_left, x_right, tau_final, f, g_left, g_right, M, N, solver='SOR')
        elif Linear_Solver == 'LU':
            u_approx_grid, x_knot, tau_knot = pde.PDE_Backward_Euler(x_left, x_right, tau_final, f, g_left, g_right, M, N, solver='LU')
    elif PDE_Solver == "Crank_Nicolson":
        if Linear_Solver == 'SOR':
            u_approx_grid, x_knot, tau_knot = pde.PDE_Crank_Nicolson(x_left, x_right, tau_final, f, g_left, g_right, M, N, solver='SOR')
        elif Linear_Solver == 'LU':
            u_approx_grid, x_knot, tau_knot = pde.PDE_Crank_Nicolson(x_left, x_right, tau_final, f, g_left, g_right, M, N, solver='LU')
    if print_grid:
        print u_approx_grid[::-1, :]
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

    def RMS_error():
        S_knot = K * np.exp(x_knot)
        V_approx, V_knot = np.zeros([len(S_knot)]), np.zeros([len(S_knot)])
        for i in xrange(len(S_knot)):
            Si = S_knot[i]
            opt.spot = Si
            V_knot[i] = BS.Black_Scholes_Pricing(opt, r)
            V_approx[i] = np.exp(-a * x_knot[i] - b * tau_final) * u_approx[i]
        opt.spot = S0
        vec_RMS = abs(V_approx - V_knot) > 0.00001 * S0
        error_RMS = np.sqrt(np.mean(((V_approx[vec_RMS] - V_knot[vec_RMS]) ** 2) / (V_knot[vec_RMS] ** 2)))
        return error_RMS

    V_FD = linear_interp_1(S0, K)
    if not Greek:
        return V_FD
    else:
        x_compute = np.log(S0 / K)
        dx = (x_right - x_left) / N
        i = int(np.floor((x_compute - x_left) / dx))
        x_lo = x_knot[i]
        x_hi = x_knot[i + 1]
        S_lo = K * np.exp(x_lo)
        S_hi = K * np.exp(x_hi)
        V_lo = np.exp(-a * x_lo - b * tau_final) * u_approx[i]
        V_hi = np.exp(-a * x_hi - b * tau_final) * u_approx[i + 1]
        Delta = (V_hi - V_lo) / (S_hi - S_lo)

        x_llo = x_knot[i - 1]
        S_llo = K * np.exp(x_llo)
        V_llo = np.exp(-a * x_llo - b * tau_final) * u_approx[i - 1]
        x_hhi = x_knot[i + 2]
        S_hhi = K * np.exp(x_hhi)
        V_hhi = np.exp(-a * x_hhi - b * tau_final) * u_approx[i + 2]
        Gamma = ((V_hhi - V_hi) / (S_hhi - S_hi) - (V_lo - V_llo) / (S_lo - S_llo)) / ((S_hhi + S_hi) / 2 - (S_lo + S_llo) / 2)

        d_tau = tau_knot[-1] - tau_knot[-2]
        dt = - 2 * d_tau / sigma ** 2
        V_lo_pre = np.exp(-a * x_lo - b * (tau_final - d_tau)) * u_approx_grid[-2, i]
        V_hi_pre = np.exp(-a * x_hi - b * (tau_final - d_tau)) * u_approx_grid[-2, i + 1]
        V_FD_pre = (V_lo_pre * (S_hi - S0) + V_hi_pre * (S0 - S_lo)) / (S_hi - S_lo)
        Theta = (V_FD - V_FD_pre) / dt
        # print u_approx[i], u_approx[i+1], V_FD
        return V_FD, Delta, Gamma, Theta


def finite_diff_amer(opt, r, M=64, c=3, alpha_temp=0.5, PDE_Solver="Forward_Euler", Linear_Solver='LU', Greek=False, print_grid=False):
    S0, K, T, q, sigma = opt.spot, opt.strike, opt.maturity, opt.div_rate, opt.vol
    a = (r - q) / sigma ** 2 - 1 / 2
    b = ((r - q) / sigma ** 2 + 1 / 2) ** 2 + 2 * q / sigma ** 2
    x_left, x_right, tau_final = Discretization(opt, r, c)
    f, g_left, g_right = boundry_config(opt, r, c)
    N, alpha = mesh(x_left, x_right, tau_final, M, alpha_temp)

    if PDE_Solver == 'Forward_Euler':
        u_approx_grid, x_knot, tau_knot = pde.PDE_Forward_Euler_Amer(x_left, x_right, tau_final, f, g_left, g_right, M, N, opt, r)
    # Use Projected SOR
    elif PDE_Solver == "Crank_Nicolson":
        u_approx_grid, x_knot, tau_knot = pde.PDE_Crank_Nicolson_Amer(x_left, x_right, tau_final, f, g_left, g_right, M, N, opt, r, solver='SOR')
    if print_grid:
        print u_approx_grid[::-1, :]
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

    def RMS_error():
        S_knot = K * np.exp(x_knot)
        V_approx, V_knot = np.zeros([len(S_knot)]), np.zeros([len(S_knot)])
        for i in xrange(len(S_knot)):
            Si = S_knot[i]
            opt.spot = Si
            V_knot[i] = BS.Black_Scholes_Pricing(opt, r)
            V_approx[i] = np.exp(-a * x_knot[i] - b * tau_final) * u_approx[i]
        opt.spot = S0
        vec_RMS = abs(V_approx - V_knot) > 0.00001 * S0
        error_RMS = np.sqrt(np.mean(((V_approx[vec_RMS] - V_knot[vec_RMS]) ** 2) / (V_knot[vec_RMS] ** 2)))
        return error_RMS

    V_FD = linear_interp_1(S0, K)
    if not Greek:
        return V_FD
    else:
        x_compute = np.log(S0 / K)
        dx = (x_right - x_left) / N
        i = int(np.floor((x_compute - x_left) / dx))
        x_lo = x_knot[i]
        x_hi = x_knot[i + 1]
        S_lo = K * np.exp(x_lo)
        S_hi = K * np.exp(x_hi)
        V_lo = np.exp(-a * x_lo - b * tau_final) * u_approx[i]
        V_hi = np.exp(-a * x_hi - b * tau_final) * u_approx[i + 1]
        Delta = (V_hi - V_lo) / (S_hi - S_lo)

        x_llo = x_knot[i - 1]
        S_llo = K * np.exp(x_llo)
        V_llo = np.exp(-a * x_llo - b * tau_final) * u_approx[i - 1]
        x_hhi = x_knot[i + 2]
        S_hhi = K * np.exp(x_hhi)
        V_hhi = np.exp(-a * x_hhi - b * tau_final) * u_approx[i + 2]
        Gamma = ((V_hhi - V_hi) / (S_hhi - S_hi) - (V_lo - V_llo) / (S_lo - S_llo)) / ((S_hhi + S_hi) / 2 - (S_lo + S_llo) / 2)

        d_tau = tau_knot[-1] - tau_knot[-2]
        dt = - 2 * d_tau / sigma ** 2
        V_lo_pre = np.exp(-a * x_lo - b * (tau_final - d_tau)) * u_approx_grid[-2, i]
        V_hi_pre = np.exp(-a * x_hi - b * (tau_final - d_tau)) * u_approx_grid[-2, i + 1]
        V_FD_pre = (V_lo_pre * (S_hi - S0) + V_hi_pre * (S0 - S_lo)) / (S_hi - S_lo)
        Theta = (V_FD - V_FD_pre) / dt
        # print u_approx[i], u_approx[i + 1], V_FD,
        return V_FD, Delta, Gamma, Theta

if __name__ == "__main__":
    amp = Option(S0=41, K=40, T=0.75, q=0.02, sigma=0.35, ae='AM', cp='P')
    r = 0.04
    # print BS.Black_Scholes_Pricing(eup, r, Greek=True)
    print finite_diff(amp, r, Greek=True)
