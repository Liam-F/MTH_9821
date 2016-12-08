from __future__ import division
from Option import *
from Barrier_Option import *
import numpy as np
import scipy.stats as stats

def Black_Scholes_Pricing(Option, r, Greek = False):
    '''
    return the BS price of an European option.
    :param Option: An European Option
    :param r: risk free rate in unit
    :return: the BS price
    '''
    # if Option.ae != 'EU':
    #     return 'Wrong option kind'

    S, K, T, q, sigma = Option.spot, Option.strike, Option.maturity, Option.div_rate, Option.vol

    d1 = (np.log(S / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    Nd1 = stats.norm.cdf(d1)
    Nd2 = stats.norm.cdf(d2)

    pbs = 0
    if Option.cp == 'C':
        pbs = S * np.exp(-q * T) * Nd1 - K * np.exp(-r * T) * Nd2
        Delta = Nd1 * np.exp(-q * T)
        Gamma = np.exp(-q * T) * stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        Theta = -np.exp(-q * T) * S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * Nd2 + q * S * np.exp(-q * T) * Nd1
    if Option.cp == 'P':
        pbs = K * np.exp(-r * T) * (1 - Nd2) - S * np.exp(-q * T) * (1 - Nd1)
        Delta = - (1 - Nd1) * np.exp(-q * T)
        Gamma = np.exp(-q * T) * stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        Theta = -np.exp(-q * T) * S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * (1 - Nd2) - q * S * np.exp(-q * T) * (1 - Nd1)
    if not Greek:
        return pbs
    else:
        return (pbs, Delta, Gamma, Theta)

def Down_N_Out_Pricing(dno, r, Greek=False):
    '''
    Return the value of a down-and out option by closed formula
    :param dno: A down and out option
    :param r: interest rate
    :param Greek: calculate Greeks if True
    :return: Value and Greeks
    '''
    V = 0
    S, K, T, q, sigma, B = dno.spot, dno.strike, dno.maturity, dno.div_rate, dno.vol, dno.Barrier
    if dno.cp == "C":
        dno.spot = B ** 2 / S
        C2 = Black_Scholes_Pricing(dno, r)
        dno.spot = S
        C1 = Black_Scholes_Pricing(dno, r)
        a = (r - q) / sigma ** 2 - 1 / 2
        V = C1 - C2 * (B / S) ** (2 * a)
    return V

if __name__ == "__main__":
    eup = Option(S0 = 41, K = 40, T = 1, q = 0.01, sigma = 0.3, cp = 'P', ae = 'EU')
    print Black_Scholes_Pricing(eup, 0.03, Greek = True)
