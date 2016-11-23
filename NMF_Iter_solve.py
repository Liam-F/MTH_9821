import numpy as np
import math
import NMF_linear_solve as nmfls
'''
This code contains Iterative solvers to linear systems for MTH 9821
Jacobi
Gauss-Siedel (banded)
SOR (banded)
'''


def norm(v):
    return math.sqrt(np.dot(np.transpose(v),v))

def Gauss_Siedel_iter(A, b, x0, tol, res_cri = 1, show_ic = False):
    x = x0
    r = b - np.dot(A, x0)
    N = int(math.sqrt(A.shape[0]))
    nr0 = norm(r)
    L = np.tril(A,-1)
    U = np.triu(A,1)
    D = np.diag(np.diag(A))
    M = D + L
    b_new = nmfls.forward_subst(D + L, b)
    ic = 0
    r_cri = 1
    while(r_cri > tol):
        prevx = x
        x = - nmfls.forward_subst(M, np.dot(U,x)) + b_new
        r = b - np.dot(A, x)
        r_cri = norm(r) / nr0
        if res_cri != 1:
            r_cri = norm(x - prevx)
        ic += 1
        if ic <= 3:
            print ic
            print x
    if show_ic:
        print "GS ic = ", ic
    return x

def Jacobi_iter(A, b, x0, tol, res_cri = 1, show_ic = False):
    x = x0
    r = b - np.dot(A, x0)
    nr0 = norm(r)
    L = np.tril(A,-1)
    U = np.triu(A,1)
    D = np.diag(np.diag(A))

    D_inv = np.diag(1 / np.diag(A))
    b_new = np.dot(D_inv, b)
    ic = 0
    r_cri = 1
    while(r_cri > tol):
        prevx = x
        x = - np.dot(D_inv, (np.dot(L, x) + np.dot(U, x))) + b_new
        r = (b - np.dot(A, x))
        r_cri = norm(r) / nr0
        if res_cri != 1:
            r_cri = norm(x - prevx)
        ic += 1
        if ic <= 3:
            print ic
            print x
    if show_ic:
        print "Jacobi ic = ", ic
    return x

def SOR_iter(A, b, x0, tol, omega, res_cri = 1, show_ic = False):
    x = x0
    r = b - np.dot(A, x0)
    nr0 = norm(r)
    L = np.tril(A,-1)
    U = np.triu(A,1)
    D = np.diag(np.diag(A))
    M = D + omega * L
    b_new = nmfls.forward_subst(M, b)
    ic = 0
    r_cri = 1
    while(r_cri > tol):
        prevx = x
        x = nmfls.forward_subst(M, np.dot((1 - omega)* D - omega * U, x)) + omega * b_new
        r = b - np.dot(A, x)
        r_cri = norm(r) / nr0
        if res_cri != 1:
            r_cri = norm(x - prevx)
        ic += 1
        if ic <= 3:
            print ic
            print x
    if show_ic:
        print "SOR ic = ", ic
    return x

def Gauss_Siedel_iter_banded(A, m, b, x0, tol, res_cri = 1, show_ic = False):
    x = x0
    r = b - np.dot(A, x0)
    nr0 = norm(r)
    L = np.tril(A,-1)
    U = np.triu(A,1)
    D = np.diag(np.diag(A))

    b_new = nmfls.forward_subst_banded(D + L, m, b)
    ic = 0
    r_cri = 1
    while(r_cri > tol):
        prevx = x
        x = - nmfls.forward_subst_banded(D + L, m, np.dot(U,x)) + b_new
        r = b - np.dot(A, x)
        r_cri = norm(r) / nr0
        if res_cri != 1:
            r_cri = norm(x - prevx)
        ic += 1
    if show_ic:
        print "GS banded ic = ", ic
    return x

def SOR_iter_banded(A, m, b, x0, tol, omega, res_cri = 1, show_ic = False):
    x = x0
    r = b - np.dot(A, x0)
    nr0 = norm(r)
    L = np.tril(A,-1)
    U = np.triu(A,1)
    D = np.diag(np.diag(A))
    M = D + omega * L
    P = (1 - omega)* D - omega * U
    b_new = nmfls.forward_subst_banded(M, m, b) * omega
    ic = 0
    r_cri = 1
    while(r_cri > tol):
        prevx = x
        x = nmfls.forward_subst_banded(M, m, np.dot(P, x)) + b_new
        r = b - np.dot(A, x)
        r_cri = norm(r) / nr0
        if res_cri != 1:
            r_cri = norm(x - prevx)
        ic += 1
    if show_ic:
        print "SOR banded ic = ", ic
    return x