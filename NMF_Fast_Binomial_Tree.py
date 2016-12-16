from __future__ import division
import numpy as np
from time import *
from Option import *
from copy import *

class BinomialTree:
    def __init__(self, N, r, Op):
        '''
        Generates the binomial tree framework for a certain option
        :param N: Time steps
        :param r: risk free interest rate in units
        :param Op: An Option class instance
        '''
        self.tstep = N
        self.riskfree = r
        dt = Op.maturity / N
        self.u = np.exp(Op.vol * np.sqrt(dt))
        self.d = 1 / self.u
        self.p = (np.exp((r - Op.div_rate) * dt) - self.d) / (self.u - self.d)

    def finalvalue(self, Op):
        '''
        Generates the final state value of an option with a N step binomial tree model
        :return: a [N + 1, 1] vector containing the final values
        '''
        N, u, d = self.tstep, self.u, self.d
        fv = np.zeros([N + 1, 1])
        fs = np.zeros([N + 1, 1])
        fs[0] = u ** N * Op.spot
        fv[0] = Op.value(fs[0])
        for i in xrange(1, N + 1):
            fs[i] = fs[i - 1] * d / u
            fv[i] = Op.value(fs[i])
        return fv

def Fast_Binomial_Tree_Pricing(Op, BMT, Greek=False):
    '''
    Fast Binomial Tree pricer using numba, works only with plain vanilla European and American call and puts
    :return: The risk-neutral binomial tree price (by taking risk neutral expectation)
    '''
    S0, K, T, q, sigma = Op.spot, Op.strike, Op.maturity, Op.div_rate, Op.vol
    N, r, u, d, p = BMT.tstep, BMT.riskfree, BMT.u, BMT.d, BMT.p
    dt = T / N
    # u = np.exp(sigma * np.sqrt(dt))
    # d = 1 / u
    # p = (np.exp((r - q) * dt) - d) / (u - d) # risk neutral probability for the stock price to go up

    fv = np.zeros([N + 1, 1])
    fs = np.zeros([N + 1, 1])
    fs[0] = u ** N * S0
    for i in xrange(1, N + 1):
        fs[i] = fs[i - 1] * d / u
        if Op.cp == "C":
            fv[i] = max(fs[i] - K, 0)
        else:
            fv[i] = max(K - fs[i], 0)

    Sc = fs.copy()
    for j in xrange(N - 1, -1, -1):
        fv[0:j+1] = np.exp(-r * dt) * (p * fv[0:j+1] + (1-p) * fv[1:j+2])
        if Op.ae == "AM":
            Sc *= d
            if Op.cp == "C":
                fv[0:j+1] = np.maximum(fv[0:j+1], Sc[0:j+1] - K)
            else:
                fv[0:j + 1] = np.maximum(fv[0:j + 1], K - Sc[0:j + 1])
        # for i in xrange(0, j + 1):
        #     if EU:
        #         fv[i] = np.exp(-r * dt) * (p * fv[i] + (1 - p) * fv[i + 1])
        #     else:
        #         Sc = S0 * u ** (j - i) * d ** i
        #         if Call:
        #             fv[i] = max(np.exp(-r * dt) * (p * fv[i] + (1 - p) * fv[i + 1]), Sc - K)
        #         else:
        #             fv[i] = max(np.exp(-r * dt) * (p * fv[i] + (1 - p) * fv[i + 1]), K - Sc)
        if Greek:
            if j == 2:
                fv2 = fv[:3].copy()
                fs2 = Sc[:3].copy()
            if j == 1:
                fv1 = fv[:2].copy()
                fs1 = Sc[:2].copy()

    if not Greek:
        return float(fv[0][0])
    else:
        Delta = (fv1[0] - fv1[1]) / (fs1[0] - fs1[1])
        Gamma = ((fv2[0] - fv2[1]) / (fs2[0] - fs2[1]) - (fv2[1] - fv2[2]) / (fs2[1] - fs2[2])) / ((fs2[0] - fs2[2]) / 2)
        Theta = (fv2[1] - fv[0][0]) / (2 * dt)
        return (fv[0][0], Delta, Gamma, Theta)


def implied_vol(opt, r, p_m, sigma_0, sigma_n1, tol=10 ** -4, N=2500):
    '''
    Compute the implied volatility with secant method on a binomial tree
    :param opt: Option, whose implied vol need to be determined
    :param sigma_0: initial_guess 0
    :param sigma_n1: initial_guess -1
    :param tol: tolerance of iteration
    :param p_m: market price of the option
    :param N: time step of the tree, default 2500
    :return: the implied vol
    '''
    def f(sigma):
        opt.vol = sigma
        bmt = BinomialTree(N, r, opt)
        p_sigma = Fast_Binomial_Tree_Pricing(opt, bmt)
        return p_sigma - p_m

    sigma_old, sigma_new = sigma_n1, sigma_0
    ic = 0
    while abs(sigma_new - sigma_old) > tol:
        sigma_oldest = sigma_old
        sigma_old = sigma_new
        sigma_new = sigma_old - f(sigma_old) * (sigma_old - sigma_oldest) / (f(sigma_old) - f(sigma_oldest))
        ic += 1
        print ic, sigma_new
    # print "ic=", ic
    return sigma_new


if __name__ == "__main__":
    start = time()
    S0, K, T, sigma, q, r = 50, 52, 11/12, 0.3, 0.01, 0.03
    euc = Option(S0, K, T, q, sigma, ae="EU", cp="C")
    amc = Option(S0, K, T, q, sigma, ae="AM", cp="C")
    N = 15000
    bmt = BinomialTree(N, r, euc)
    V_EU = Fast_Binomial_Tree_Pricing(euc, bmt, Greek=True)
    cur = time()
    print cur - start
    V_AM = Fast_Binomial_Tree_Pricing(amc, bmt, Greek=True)
    end = time()
    print end - cur