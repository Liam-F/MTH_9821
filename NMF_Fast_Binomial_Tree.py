from __future__ import division
import numpy as np
from numba import *
from time import *

# @jit
# @jit(float64(float64, float64, float64, float64, float64, float64, int32, boolean, boolean, boolean), nopython=True)
# @jit(float64(float64, float64, float64, float64, float64, float64, int32, boolean, boolean, boolean))
def Fast_Binomial_Tree_Pricing(S0, K, T, sigma, q, r, N, EU=True, Call=True, Greek=False):
    '''
    Fast Binomial Tree pricer using numba, works only with plain vanilla European and American call and puts
    :return: The risk-neutral binomial tree price (by taking risk neutral expectation)
    '''
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d) # risk neutral probability for the stock price to go up

    fv = np.zeros([N + 1, 1])
    fs = np.zeros([N + 1, 1])
    fs[0] = u ** N * S0
    for i in xrange(1, N + 1):
        fs[i] = fs[i - 1] * d / u
        if Call:
            fv[i] = max(fs[i] - K, 0)
        else:
            fv[i] = max(K - fs[i], 0)

    for j in xrange(N - 1, -1, -1):
        for i in xrange(0, j + 1):
            if EU:
                fv[i] = np.exp(-r * dt) * (p * fv[i] + (1 - p) * fv[i + 1])
            else:
                Sc = S0 * u ** (j - i) * d ** i
                if Call:
                    fv[i] = max(np.exp(-r * dt) * (p * fv[i] + (1 - p) * fv[i + 1]), Sc - K)
                else:
                    fv[i] = max(np.exp(-r * dt) * (p * fv[i] + (1 - p) * fv[i + 1]), K - Sc)
        # if Greek:
        #     if j == 2:
        #         fv2 = np.array([float(fv[0]), float(fv[1]), float(fv[2])])
        #         fs2 = np.array([S0 * u ** 2, S0 * u * d, S0 * d ** 2])
        #     if j == 1:
        #         fv1 = np.array([float(fv[0]), float(fv[1])])
        #         fs1 = np.array([S0 * u, S0 * d])

    if not Greek:
        return float(fv[0][0])
    # else:
    #     Delta = (fv1[0] - fv1[1]) / (fs1[0] - fs1[1])
    #     Gamma = ((fv2[0] - fv2[1]) / (fs2[0] - fs2[1]) - (fv2[1] - fv2[2]) / (fs2[1] - fs2[2])) / ((fs2[0] - fs2[2]) / 2)
    #     Theta = (fv2[1] - fv[0][0]) / (2 * dt)
    #     return (fv[0][0], Delta, Gamma, Theta)

if __name__ == "__main__":
    start = time()
    S0, K, T, sigma, q, r = 50, 52, 11/12, 0.3, 0.01, 0.03
    N = 1500
    V_BT = Fast_Binomial_Tree_Pricing(S0, K, T, sigma, q, r, N)
    end = time()
    print end - start