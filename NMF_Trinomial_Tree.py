from __future__ import division
import numpy as np
from Option import *
from NMF_Black_Scholes import *
import copy
import matplotlib as plt

class TrinomialTree:
    def __init__(self, N, r, Op):
        '''
        Generates the trinomial tree framework for a certain option
        :param N: Time steps
        :param r: risk free interest rate in units
        :param Op: An Option class instance
        '''
        S0, K, T, q, sigma = Op.spot, Op.strike, Op.maturity, Op.div_rate, Op.vol
        self.tstep = N
        self.riskfree = r
        dt = T / N
        self.u = np.exp(sigma * np.sqrt(3 * dt))
        self.d = 1 / self.u
        # risk neutral probability for the asset to go up, mid or down
        self.p_u = 1/6 + (r - q - sigma ** 2 / 2) * np.sqrt(dt / (12 * sigma ** 2))
        self.p_m = 2/3
        self.p_d = 1 - self.p_u - self.p_m


def Trinomial_Tree_Pricing(Op, TMT, Greek = False):
    '''
    Calculate the risk neutral price of a certain option with general Trinomial Tree model.
    :param Op: An instance from class Option
    :param TMT: An trinomial tree instance with step N, rate r, upfactor u and downfactor d
    :return: The risk-neutral trinomial tree price (by taking risk neutral expectation)
    '''
    S0, K, T, q = Op.spot, Op.strike, Op.maturity, Op.div_rate
    N, r, u, d = TMT.tstep, TMT.riskfree, TMT.u, TMT.d
    p_u, p_m, p_d = TMT.p_u, TMT.p_m, TMT.p_d,
    dt = T / N
    fv = Op.finalvalue_tri(N, u, d)
    for j in range(N - 1, -1, -1):
        for i in range(0, 2 * j + 1):
            if Op.ae == 'EU':
                fv[i] = np.exp(-r * dt) * (p_u * fv[i] + p_m * fv[i + 1] + p_d * fv[i + 2])
            elif Op.ae == 'AM':
                Sc = S0 * u ** (j - i)
                if Op.cp == 'C':
                    fv[i] = max(np.exp(-r * dt) * (p_u * fv[i] + p_m * fv[i + 1] + p_d * fv[i + 2]), Sc - K)
                elif Op.cp == 'P':
                    fv[i] = max(np.exp(-r * dt) * (p_u * fv[i] + p_m * fv[i + 1] + p_d * fv[i + 2]), K - Sc)
        if j == 2:
            fv2 = [float(fv[0]), float(fv[1]), float(fv[2]), float(fv[3]), float(fv[4])]
            fs2 = [S0 * u ** 2, S0 * u, S0, S0 * d, S0 * d ** 2]
        if j == 1:
            fv1 = [float(fv[0]), float(fv[1]), float(fv[2])]
            fs1 = [S0 * u, S0, S0 * d]

    if not Greek:
        return fv[0][0]
    else:
        Delta = (fv1[0] - fv1[2]) / (fs1[0] - fs1[2])
        Gamma = ((fv2[0] - fv2[2]) / (fs2[0] - fs2[2]) - (fv2[2] - fv2[4]) / (fs2[2] - fs2[4])) / (fs1[0] - fs1[2])
        Theta = (fv1[1] - fv[0][0]) / dt
        return (fv[0][0], Delta, Gamma, Theta)
#
#
# def Avg_Trinomial_Tree_Pricing(Op, TMT, Greek = False):
#     '''
#     Calculate the risk neutral price of a certain option with Average Trinomial Tree model.
#     :param Op: An instance from class Option
#     :param TMT: An trinomial tree instance with step N, rate r, upfactor u and downfactor d
#     :return: The risk-neutral trinomial tree price (by taking risk neutral expectation)
#     '''
#     TMT_1 = TrinomialTree(TMT.tstep + 1, TMT.riskfree, Op)
#     if Greek:
#         old = Trinomial_Tree_Pricing(Op, TMT, Greek)
#         new = Trinomial_Tree_Pricing(Op, TMT_1, Greek)
#         avgbmt_tuple = tuple([(old[i] + new[i]) / 2 for i in range(4)])
#         return avgbmt_tuple
#     else:
#         return (Trinomial_Tree_Pricing(Op, TMT) + Trinomial_Tree_Pricing(Op, TMT_1)) / 2


def Trinomial_Black_Scholes(Op, TMT, Greek = False):
    '''
    Calculate the risk neutral price of a certain option with Trinomial Black Scholes model.
    :param Op: An instance from class Option
    :param TMT: An trinomial tree instance with step N, rate r, upfactor u and downfactor d
    :param Greek: A boolean variable whether the output should contain greeks
    :return: The risk-neutral trinomial BS price (by taking risk neutral expectation)
    '''
    S0, K, T, q = Op.spot, Op.strike, Op.maturity, Op.div_rate
    N, r, u, d = TMT.tstep, TMT.riskfree, TMT.u, TMT.d
    p_u, p_m, p_d = TMT.p_u, TMT.p_m, TMT.p_d,
    dt = T / N
    # Generate the first step option value using BS formula
    fv = np.zeros([2 * N - 1, 1])
    Op.maturity = dt
    Op.spot = u ** (N - 1) * S0
    fv[0] = Black_Scholes_Pricing(Op, r)
    for i in range(1, 2 * N - 1):
        Op.spot *= d
        if Op.ae == 'EU':
            fv[i] = Black_Scholes_Pricing(Op, r)
        if Op.ae == 'AM':
            if Op.cp == 'C':
                fv[i] = max(Black_Scholes_Pricing(Op, r), Op.spot - K)
            elif Op.cp == 'P':
                fv[i] = max(Black_Scholes_Pricing(Op, r), K - Op.spot)
    # print fv
    for j in range(N - 2, -1, -1):
        for i in range(0, 2 * j + 1):
            if Op.ae == 'EU':
                fv[i] = np.exp(-r * dt) * (p_u * fv[i] + p_m * fv[i + 1] + p_d * fv[i + 2])
            if Op.ae == 'AM':
                 Sc = S0 * u ** (j - i)
                 if Op.cp == 'C':
                     fv[i] = max(np.exp(-r * dt) * (p_u * fv[i] + p_m * fv[i + 1] + p_d * fv[i + 2]), Sc - K)
                 elif Op.cp == 'P':
                     fv[i] = max(np.exp(-r * dt) * (p_u * fv[i] + p_m * fv[i + 1] + p_d * fv[i + 2]), K - Sc)
                # fv[i] = np.exp(-r * dt) * (p * fv[i] + (1 - p) * fv[i + 1])
            # if j == 3:
                # print Sc, K, K - Sc

        if j == 2:
            fv2 = [float(fv[0]), float(fv[1]), float(fv[2]), float(fv[3]), float(fv[4])]
            fs2 = [S0 * u ** 2, S0 * u, S0, S0 * d, S0 * d ** 2]
        if j == 1:
            fv1 = [float(fv[0]), float(fv[1]), float(fv[2])]
            fs1 = [S0 * u, S0, S0 * d]

    Op.spot = S0
    Op.maturity = T
    if not Greek:
        return fv[0][0]
    else:
        Delta = (fv1[0] - fv1[2]) / (fs1[0] - fs1[2])
        Gamma = ((fv2[0] - fv2[2]) / (fs2[0] - fs2[2]) - (fv2[2] - fv2[4]) / (fs2[2] - fs2[4])) / (fs1[0] - fs1[2])
        Theta = (fv1[1] - fv[0][0]) / dt
        return (fv[0][0], Delta, Gamma, Theta)


def Trinomial_Black_Scholes_Richardson(Op, TMT, Greek = False):
    '''
    Calculate the risk neutral price of a certain option using Trinomial Black Scholes model with Richardson's extrapolation.
    :param Op: An instance from class Option
    :param TMT: An trinomial tree instance with step N, rate r, upfactor u and downfactor d
    :return: The risk-neutral trinomial BS Richardson price (by taking risk neutral expectation)
    '''
    N, r = TMT.tstep, TMT.riskfree
    TMT_2 = TrinomialTree(int(N / 2), r, Op)
    if Greek:
        full = Trinomial_Black_Scholes(Op, TMT, Greek)
        half = Trinomial_Black_Scholes(Op, TMT_2, Greek)
        pbbsr_tuple = tuple([full[i] * 2 - half[i] for i in range(4)])
        return pbbsr_tuple
    else:
        return Trinomial_Black_Scholes(Op, TMT) * 2 - Trinomial_Black_Scholes(Op, TMT_2)


if __name__ == "__main__":
    eup = Option(S0 = 41, K = 40, T = 1, q = 0.01, sigma = 0.3, cp = 'P', ae = 'EU')
    amp = Option(S0 = 41, K = 40, T = 1, q = 0.01, sigma = 0.3, cp = 'P', ae = 'AM')
    tmt = TrinomialTree(N = 10, r = 0.03, Op = eup)
    # peup = Trinomial_Tree_Pricing(eup, bmt)
    # pavgeup = Avg_Trinomial_Tree_Pricing(eup, bmt)
    # pbbs = Trinomial_Black_Scholes(eup, bmt)
    # pbbsr = Trinomial_Black_Scholes_Richardson(eup, bmt)
    # print peup
    # print pavgeup
    # print pbbs
    # print pbbsr
    # (peup, deltae, gammae, thetae) = Trinomial_Tree_Pricing(eup, tmt, True)
    #(pamp, deltaa, gammaa, thetaa) = Trinomial_Tree_Pricing(amp, bmt, True)
    # print [peup, deltae, gammae, thetae]
    #print [pamp, deltaa, gammaa, thetaa]
    # pabt, Deltaabt, Gammaabt, Thetaabt = Avg_Trinomial_Tree_Pricing(eup, bmt, True)
    # print pabt, Deltaabt, Gammaabt, Thetaabt
    # pbbs, Deltabbs, Gammabbs, Thetabbs = Trinomial_Black_Scholes(amp, tmt, True)
    # print pbbs, Deltabbs, Gammabbs, Thetabbs
    pbbsr, Deltabbsr, Gammabbsr, Thetabbsr = Trinomial_Black_Scholes_Richardson(eup, tmt, True)
    print pbbsr, Deltabbsr, Gammabbsr, Thetabbsr