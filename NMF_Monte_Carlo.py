#Monte Carlo Pricing for plain vanilla options
from __future__ import division
from Option import *
import NMF_RND as rnd
import NMF_Black_Scholes as BS
import numpy as np


def Monte_Carlo_Plain_Vanilla(Op, r, N, Greek=False):
    '''
    Return the result of Monte Carlo simulation of a plain vanilla European option
    :param Op: Option, should be European
    :param r: float, interest rate in unit
    :param N: int, MC simulation steps
    :return: float, MC price and Greeks
    '''
    S0, K, sigma, T, q = Op.spot, Op.strike, Op.vol, Op.maturity, Op.div_rate
    z = rnd.std_normal_inverse_transform(N)
    S = [S0 * np.exp((r - q - sigma ** 2 / 2) * T + sigma * np.sqrt(T) * zi) for zi in z]  # MC final stock price
    if Op.cp == 'C':
        p_vec = [np.exp(-r * T) * max(Si - K, 0) for Si in S]
        p_MC = np.average(p_vec)
        if Greek:
            Delta_vec = [int(Si > K) * np.exp(-r * T) * Si / S0 for Si in S]
            Vega_vec = [int(S[i] > K) * np.exp(-r * T) * S[i] * (-sigma * T + z[i] * np.sqrt(T)) for i in range(0, N)]
            Delta_MC = np.average(Delta_vec)
            Vega_MC = np.average(Vega_vec)

    if Op.cp == 'P':
        p_vec = [np.exp(-r * T) * max(- Si + K, 0) for Si in S]
        p_MC = np.average(p_vec)
        if Greek:
            Delta_vec = [-int(Si < K) * np.exp(-r * T) * Si / S0 for Si in S]
            Vega_vec = [-int(S[i] < K) * np.exp(-r * T) * S[i] * (-sigma * T + z[i] * np.sqrt(T)) for i in range(0, N)]
            Delta_MC = np.average(Delta_vec)
            Vega_MC = np.average(Vega_vec)

    if not Greek:
        return p_MC
    else:
        return p_MC, Delta_MC, Vega_MC

