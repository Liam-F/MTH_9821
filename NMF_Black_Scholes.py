from __future__ import division
from Option import *
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
        Vega = S * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T)
    if Option.cp == 'P':
        pbs = K * np.exp(-r * T) * (1 - Nd2) - S * np.exp(-q * T) * (1 - Nd1)
        Delta = - (1 - Nd1) * np.exp(-q * T)
        Gamma = np.exp(-q * T) * stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
        Theta = -np.exp(-q * T) * S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * (1 - Nd2) - q * S * np.exp(-q * T) * (1 - Nd1)
        Vega = S * np.exp(-q * T) * stats.norm.pdf(d1) * np.sqrt(T)
    if not Greek:
        return pbs
    else:
        return (pbs, Delta, Gamma, Theta, Vega)

if __name__ == "__main__":
    # euc = Option(S0 = 40, K = 42, T = 0.75, q = 0.01, sigma = 0.25, cp = 'C', ae = 'EU')
    eup = Option(S0 = 50, K = 50, T = 0.5, q = 0, sigma = 0.35, cp = 'P', ae = 'EU')
    # print Black_Scholes_Pricing(euc, 0.03, Greek=True)
    print Black_Scholes_Pricing(eup, 0.05, Greek = True)
    # print stats.norm.cdf(2)
