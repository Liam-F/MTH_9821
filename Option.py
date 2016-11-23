from __future__ import division
import numpy as np


class Option:
    def __init__(self, S0, K, T, q, sigma, cp, ae, r=0.02):
        self.spot = S0 # Spot price of UA
        self.strike = K # Strike of the option
        self.maturity = T # Maturity of the option
        self.div_rate = q # dividend ratio of UA
        self.vol = sigma # volatility of the stock
        self.cp = cp # 'C' for call and 'P' for put
        self.ae = ae # 'AM' for american and 'EU' for european
        self.a = (r - q) / sigma ** 2 - 1 / 2
        self.b = ((r - q) / sigma ** 2 + 1 / 2) ** 2 + 2 * q / sigma ** 2

    def value(self, S):
        '''
        Return the value of an option when the stocck is at price S
        :param S: stock price
        :return: value
        '''
        value = 0
        if self.cp == 'C':
            value = (S - self.strike) * int(S >= self.strike)
        if self.cp == 'P':
            value = (self.strike - S) * int(S <= self.strike)
        return value

    def finalvalue(self, N, u, d):
        '''
        Generates the final state value of an option with a N step binomial tree model
        :param N: How many steps are there in the binomial tree
        :param u: Binomial Tree parameter of stock price going up
        :param d: Binomial Tree parameter of stock price going down
        :return: a [N + 1, 1] vector containing the final values
        '''
        fv = np.zeros([N + 1, 1])
        fs = np.zeros([N + 1, 1])
        fs[0] = u ** N * self.spot
        fv[0] = self.value(fs[0])
        for i in range(1, N + 1):
            fs[i] = fs[i - 1] * d / u
            fv[i] = self.value(fs[i])
        return fv

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
            fv[i] = self.value(fs[i])
        return fv

    def f(x):
        return K * np.exp(a * x) * (1 - np.exp(x)) * int(x < 0)

    def g_left(tau):
        return K * np.exp(a * x_left + b * tau) * (np.exp(- 2 * r * tau / sigma ** 2) - np.exp(x_left - 2 * q * tau / sigma ** 2))

    def g_right(tau):
        return 0