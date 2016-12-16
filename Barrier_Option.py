from __future__ import division
from Option import *
import numpy as np

class Down_N_Out_Option(Option):
    def __init__(self, S0, K, B, T, q, sigma, cp, ae="DNO"):
        Option.__init__(self, S0, K, T, q, sigma, cp, ae)
        self.Barrier = B # Barrier

    def value(self, S):
        value = 0
        if self.cp == 'C':
            value = (S - self.strike) * int(S >= self.strike) * int(S >= self.Barrier)
        if self.cp == 'P':
            value = (self.strike - S) * int(S <= self.strike) * int(S >= self.Barrier)
        return value

class Double_Barrier_Option(Option):
    def __init__(self, S0, K, B_up, B_down, T, q, sigma, cp, ae='Double'):
        Option.__init__(self, S0, K, T, q, sigma, cp, ae)
        self.Up_Barrier = B_up
        self.Down_Barrier = B_down

    def value(self, S):
        return max(S - self.strike - 2, -2) * int(S > self.Down_Barrier) * int(S < self.Up_Barrier) + 2

    def value_path(self, S_path):
        S_path = np.array(S_path)
        B_up, B_down = self.Up_Barrier, self.Down_Barrier
        if min(S_path) > B_down and max(S_path) < B_up:
            return (S_path[-1] - self.strike) * int(S_path[-1] > self.strike)
        else:
            # return 2
            down_vec = S_path > B_down
            up_vec = S_path < B_up
            bool_vec = down_vec * up_vec
            hit = np.where(bool_vec == False)[0][0]
            tstep = len(S_path)
            dt = self.maturity / (tstep - 1)
            remain_time = dt * (tstep - hit - 1)
            return 2 * np.exp(0.02 * remain_time) # r = 0.02

    def finalvalue_tri(self, N, u, d):
        '''
        Generates the final state value of an option with a N step trinomial tree model
        :param N: How many steps are there in the binomial tree
        :param u: Trinomial Tree parameter of stock price going up
        :param d: Trinomial Tree parameter of stock price going down
        :return: a [2N + 1, 1] vector containing the final values
        '''
        fv = np.zeros([2 * N + 1, 1])
        fs = np.zeros([2 * N + 1, 1])
        fs[0] = u ** N * self.spot
        fv[0] = self.value(fs[0])

        for i in range(1, 2 * N + 1):
            fs[i] = fs[i - 1] / u
            if fs[i] > self.Down_Barrier and fs[i] < self.Up_Barrier:
                fv[i] = self.value(fs[i])
            else:
                fv[i] = 2
        return fv

if __name__ == "__main__":
    op1 = Double_Barrier_Option(52, 48, 60, 40, 1, 0.01, 0.3, "C")
    print op1.value_path([52,51,51,53,55,65,66,35])