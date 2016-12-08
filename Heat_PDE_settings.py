import numpy as np

def u_exact(x, tau):
    return np.exp(x + tau)


def g_left(tau):
    return np.exp(tau - 2)


def g_right(tau):
    return np.exp(tau + 2)


def f(x):
    return np.exp(x)


def u_exact_final(x):
    return np.exp(x + 1)