from __future__ import division
from Option import *
import NMF_Black_Scholes as BS
import NMF_RND as rnd
from time import *

np.set_printoptions(linewidth=500)

class Option_Dis_Div(Option):
    def __init__(self, S0, K, T, sigma, cp, ae, div_dict):
        Option.__init__(self, S0, K, T, 0, sigma, cp, ae)
        self.dividend = div_dict # A dictionary saving all the time spot of dividend and dividend ratio (proportional dividend are written in strings, fixed dividends in numbers)


if __name__ == "__main__":
    S0, K, T, q, sigma = 50, 55.55, 7/12, 0, 0.3
    r = 0.02
    div_dict = {2 / 12: 0.50, 4 / 12: "0.01", 6 / 12: 0.75}
    eup = Option_Dis_Div(S0, K, T, sigma, cp='P', ae='EU', div_dict=div_dict)
    V_BS = BS.Black_Scholes_Pricing(eup, r)
    print V_BS
    N_lst = [4 * 2 ** i * 10000 for i in range(9)]
    # N_lst = [160000]
    Z = rnd.std_normal_Box_Muller(N_lst[-1])
    t = sorted(div_dict.keys())
    start = time()
    for N in N_lst:
        z = Z[:N]
        n = int(len(z) / 4)
        S_1 = [S0 * np.exp((r - q - sigma ** 2 / 2) * t[0] + sigma * np.sqrt(t[0]) * z[i * 4]) - 0.50 for i in xrange(n)]  # After first div payment
        S_2 = [S_1[i] * np.exp((r - q - sigma ** 2 / 2) * (t[1] - t[0]) + sigma * np.sqrt(t[1] - t[0]) * z[i * 4 + 1]) * 0.99 for i in xrange(n)]  # After second div payment
        S_3 = [S_2[i] * np.exp((r - q - sigma ** 2 / 2) * (t[2] - t[1]) + sigma * np.sqrt(t[2] - t[1]) * z[i * 4 + 2]) - 0.75 for i in xrange(n)]  # After third div payment
        S = [S_3[i] * np.exp((r - q - sigma ** 2 / 2) * (T - t[2]) + sigma * np.sqrt(T - t[2]) * z[i * 4 + 3]) for i in xrange(n)]  # MC final stock price
        V = [np.exp(-r * T) * max(- Si + K, 0) for Si in S]
        V_MC = np.mean(V)

        # Control Variate stock value and Option value
        S_tilde = [S0 * np.exp((r - q - sigma ** 2)* T + sigma * (np.sqrt(t[0]) * z[i * 4] + np.sqrt(t[1] - t[0]) * z[i * 4 + 1] + np.sqrt(t[2] - t[1]) * z[i * 4 + 2] + np.sqrt(T - t[2]) * z[i * 4 + 3])) for i in xrange(n)]
        V_tilde = [np.exp(-r * T) * max(- Si + K, 0) for Si in S_tilde]
        V_tilde_mean = np.mean(V_tilde)

        # Control Variate Technique
        dif_tilde = [V_t - V_tilde_mean for V_t in V_tilde]
        dif = [V[i] - V_tilde[i] for i in xrange(len(V))]
        b = sum([dif[i] * dif_tilde[i] for i in xrange(len(dif))]) / sum([d ** 2 for d in dif_tilde])
        W = [V[i] - b * (V_tilde[i] - V_BS) for i in xrange(len(V))]
        V_CV = np.mean(W)
        Delta_vec = [-int(S[i] < K) * np.exp(-r * T) * S_tilde[i] * 0.99 / S0 for i in xrange(len(S))]
        Delta = np.mean(Delta_vec)
        Delta_tilde_vec = [-int(S_tilde[i] < K) * np.exp(-r * T) * S_tilde[i] / S0 for i in xrange(len(S))]
        Delta_tilde = np.mean(Delta_tilde_vec)
        print n, V_MC, Delta, V_CV, Delta_tilde
        # print V_MM, abs(V_BS - V_MM),time() - start
        start = time()

