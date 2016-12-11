from __future__ import division
from Option import *
from NMF_Black_Scholes import *
import NMF_RND as rnd
from time import *

start = time()
S0, K, T, q, sigma = 50, 55, 0.5, 0, 0.3
r = 0.04
eup = Option(S0, K, T, q, sigma, ae="EU", cp="P")
V_BS = Black_Scholes_Pricing(eup, r)

N_lst = [2 ** i * 10000 for i in range(10)]
# N_lst = [160000]
Z = rnd.std_normal_Box_Muller(N_lst[-1])

for N in N_lst:
    z = Z[:N]
    n = len(z)
    S = [S0 * np.exp((r - q - sigma ** 2 / 2) * T + sigma * np.sqrt(T) * zi) for zi in z]  # MC final stock price
    S_hat = np.average(S)
    V = [np.exp(-r * T) * max(- Si + K, 0) for Si in S]
    V_hat = np.average(V)
    S_dif = [Si - S_hat for Si in S]
    V_dif = [Vi - V_hat for Vi in V]
    b = sum([S_dif[i] * V_dif[i] for i in range(n)]) / sum([S_dif[i] ** 2 for i in range(n)])
    W = [V[i] - b * (S[i] - np.exp(r * T) * S0) for i in range(n)]
    V_CV = np.mean(W)
    print V_CV, abs(V_BS - V_CV),time() - start
    start = time()

