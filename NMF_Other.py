from __future__ import division
import numpy as np
import NMF_linear_solve as LS
import math

def lin_reg(Y, X, constant = True):
    '''
    Calculates the OLS regression coefficient vector B s.t. Y = XB + e
    :param Y: Dependent vaiable, n * 1 nparray
    :param X: Independent variable, n * k nparray
    :param constant: Boolean variable, whether the regression includes constant term or not
    :return: B, k * 1 numpy array of the regression parameter
    '''
    n = Y.shape[0]
    k = X.shape[1]
    if constant:
        ones = np.ones([n, 1])
        X = np.column_stack((ones, X))
    B = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
    return B

def cov_mat(T):
    '''
    Calculate the covariance matrix given the data T
    :param T: n * k original data
    :return: Sample covariance matrix k * k
    '''
    n = T.shape[0]
    k = T.shape[1]
    mean = np.ndarray.mean(T, 0)
    meanmat = np.zeros([n, k])
    for w in range(n):
        meanmat[w, :] = mean
    dmT = T - meanmat
    Sigma = np.dot(dmT.T, dmT) / (n - 1)
    return Sigma