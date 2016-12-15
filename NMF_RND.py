# Pseudorandom number generators for MTH 9821 Numerical Methods for Finance
# Used in Monte Carlo methods
from __future__ import division
import numpy as np
import time
import scipy.stats as stats


def uniform_linear_congrunential(n, seed=1):
    '''
    Generate a U(0,1) i.i.d. sample of size n
    :param n: int sample size
    :return:  list[float] random sample
    '''
    k, a, c = 2 ** 31 - 1, 39373, 0
    x = seed
    u = [0] * n
    i = 0
    while i < n:
        x_new = (a * x + c) % k
        x = x_new
        u_new = x_new / k
        u[i] = u_new
        i += 1
    return x, u


def inv_normal_cdf(u):
    '''
    Simulates the inverse of std normal cdf function using Beasley-Springer-Moro algorithm
    :param u: float u = P(Z < z)
    :return:  float z
    '''
    if u < 0 or u > 1:
        print "invalid variable"
        return
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209, 0.0276438810333863, 0.0038405729373609, 0.0003951896511919, 0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
    y = u - 0.5
    if abs(y) < 0.42:
        r = y ** 2
        x = y * (((a[3] * r + a[2]) * r + a[1]) * r + a[0]) / ((((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1)
    else:
        r = u
        if y > 0:
            r = 1 - u
        r = np.log(-np.log(r))
        x = c[0] + r * (c[1] + r * (c[2] + r * (c[3] + r * (c[4] + r * (c[5] + r * (c[6] + r * (c[7] + r * c[8])))))))
        if y < 0:
            x = -x
    return x


def std_normal_inverse_transform(n):
    '''
    Generate N i.i.d. Std normal samples by using inverse cdf function
    :param n: int, sample size
    :return: list[int], random sample
    '''
    u = uniform_linear_congrunential(n)[1]
    z = [inv_normal_cdf(i) for i in u]
    return z


def std_normal_it(U):
    z = [inv_normal_cdf(i) for i in U]
    return z


def std_normal_acceptance_rejection(n):
    '''
    Generate Std normal samples by using acceptance-rejection method
    :param n: int, sample size
    :return: list[float], random variable
    '''
    seed = 1
    z = []
    while len(z) < n:
        seed, u = uniform_linear_congrunential(3, seed)
        x = -np.log(u[0])
        if u[1] > np.exp(- (x - 1) ** 2 / 2):
            continue
        else:
            if u[2] <= 0.5:
                x = -x
        z.append(x)
    return z


def std_normal_ar(U):
    z = []
    n = len(U)
    i = 0
    while i + 2 < n:
        u = U[i : i + 3]
        i += 3
        x = -np.log(u[0])
        if u[1] > np.exp(- (x - 1) ** 2 / 2):
            continue
        else:
            if u[2] <= 0.5:
                x = -x
        z.append(x)
    return z

def std_normal_Box_Muller(n):
    '''
    Generate Std normal sample with Box-Muller method
    :param n: int, sample size
    :return: list[float], sample
    '''
    seed = 1
    x = 10
    z = []
    while len(z) < n:
        while x > 1:
            seed, u = uniform_linear_congrunential(2, seed)
            u[0] = 2 * u[0] - 1
            u[1] = 2 * u[1] - 1
            x = u[0] ** 2 + u[1] ** 2
        y = np.sqrt(-2 * np.log(x) / x)
        z += [u[0] * y, u[1] * y]
        x = 10
    return z


def std_normal_bm(U):
    z = []
    x = 10
    n = len(U)
    i = 0
    while i + 1 < n:
        while x > 1:
            if i + 1 >= n:
                return z
            u = U[i: i + 2]
            i += 2
            u[0] = 2 * u[0] - 1
            u[1] = 2 * u[1] - 1
            x = u[0] ** 2 + u[1] ** 2
        y = np.sqrt(-2 * np.log(x) / x)
        z += [u[0] * y, u[1] * y]
        x = 10
    return z

if __name__ == "__main__":
    # s = time.time()
    # U = uniform_linear_congrunential(100)[1]
    # print time.time() - s
    # z = std_normal_ar(U)
    # z = std_normal_bm(U)
    # print z
    # print len(z)
    # print time.time() - s
    print stats.norm.cdf(1.96)
