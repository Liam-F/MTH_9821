# Definition of a simple Option class
from __future__ import division
import numpy as np

class Option:
    def __init__(self, S0, K, T, q, sigma, cp, ae):
        self.spot = S0 # Spot price of UA
        self.strike = K # Strike of the option
        self.maturity = T # Maturity of the option
        self.div_rate = q # dividend ratio of UA
        self.vol = sigma # volatility of the stock
        self.cp = cp # 'C' for call and 'P' for put
        self.ae = ae # 'AM' for american and 'EU' for european

        
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