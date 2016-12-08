from __future__ import division
from Option import *

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

if __name__ == "__main__":
    op1 = Down_N_Out_Option(41, 40, 35, 1, 0.01, 0.3, "P")
    print op1.value(34)