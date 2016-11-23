import numpy as np
# this python module is used for MTH 9821 Numerical Methods for Finance
# Basic methods used in solving linear systems are defined in this file
# backward_subst
# forward_subst
# backward_subst_banded
# forward_subst_banded
# lu_no_pivoting
# lu_row_pivoting
# check_banded
# lu_no_pivoting_banded
# lu_row_pivoting_banded
# linear_solve_lu_no_pivoting
# inverse
# linear_solve_lu_row_pivoting
# linear_solve_lu_no_pivoting_banded
# Cholesky
# Cholesky_banded
# linear_solve_Cholesky
# linear_solve_Cholesky_banded
# Important! when initializing the matrix or vectors in this file, make sure that the data type is float not integer!

def backward_subst (U, b):
    # U is a non-singular upper triangular matrix
    # b is a n*1 vector
    # return x a n*1 vector satisfies Ux = b
    n = b.shape[0]
    x = np.zeros([n, 1])
    x[n - 1] = b[n - 1] / U[n - 1, n - 1]
    for j in range(n - 2, -1, -1):
        sum = 0
        for k in range(j + 1, n):
            sum += U[j, k] * x[k]
        x[j] = (b[j] - sum) / U[j, j]
    return x

def backward_subst_banded (U, m, b):
    # U is a non-singular upper triangular matrix with band m
    # b is a n*1 vector
    # return x a n*1 vector satisfies Ux = b
    n = b.shape[0]
    x = np.zeros([n, 1])
    x[n - 1] = b[n - 1] / U[n - 1, n - 1]
    for j in range(n - 2, -1, -1):
        sum = 0
        for k in range(j + 1, min(n, j + m + 1)):
            sum += U[j, k] * x[k]
        x[j] = (b[j] - sum) / U[j, j]
    return x

def forward_subst (L, b):
    # L is a non-singular lower triangular matrix
    # b is a n*1 vector
    # return x a n*1 vector satisfies Lx = b
    n = b.shape[0]
    x = np.zeros([n,1])
    x[0] = b[0] / L[0, 0]
    for j in range(1, n):
        sum = 0
        for k in range(0, j):
            sum += L[j, k] * x[k]
        x[j] = (b[j] - sum) / L[j, j]
    return x

def forward_subst_banded (L, m, b):
    # L is a non-singular lower triangular matrix
    # b is a n*1 vector
    # return x a n*1 vector satisfies Lx = b
    n = b.shape[0]
    x = np.zeros([n,1])
    x[0] = b[0] / L[0, 0]
    for j in range(1, n):
        sum = 0
        for k in range(max(0, j - m), j):
            sum += L[j, k] * x[k]
        x[j] = (b[j] - sum) / L[j, j]
    return x

def lu_no_pivoting(A):
    # A is a square matrix
    # return a lower triangular matrix L with 1 on the main diagonal and an upper triangular matrix U, s.t. LU = A
    A = A.copy()
    n = A.shape[0]
    # initialize the matrix L and U
    U = np.zeros([n, n])
    L = np.zeros([n, n])
    for i in range(0, n - 1):
        for j in range(i, n):
            U[i, j] = A[i, j]
            L[j, i] = A[j, i] / U[i, i]
        for j in range(i + 1, n):
            for k in range(i + 1, n):
                A[j, k] -= L[j, i] * U[i, k]
    L[n - 1, n - 1] = 1
    U[n - 1, n - 1] = A[n - 1, n - 1]
    return [L, U]

def lu_row_pivoting(A, P_mat = True):
    # A is a square matrix
    # P is a permutation matrix simply changes the line of A
    # return a lower triangular matrix L with 1 on the main diagonal and an upper triangular matrix U, s.t. LU = PA
    A = A.copy()
    n = A.shape[0]
    # initialize the matrix P, L and U
    P = range(0, n)
    U = np.zeros([n, n])
    L = np.zeros([n, n])  # initialized as zero matrix
    for i in range(0, n - 1):
        i_max = i + np.argmax(abs(A[range(i, n), i]))
        # permute the lines
        vv = A[i, range(i, n)]
        A[i, i:n] = A[i_max, range(i,n)]
        A[i_max, range(i,n)] = vv
        # update P
        cc = P[i]
        P[i] = P[i_max]
        P[i_max] = cc
        if i > 0:
            ww = L[i, range(0, n)]
            L[i, range(0, n)] = L[i_max, range(0, n)]
            L[i_max, range(0, n)] = ww

        for j in range(i, n):
            U[i, j] = A[i, j]
            L[j, i] = A[j, i] / U[i, i]
        for j in range(i + 1, n):
            for k in range(i + 1, n):
                A[j, k] -= L[j, i] * U[i, k]
        del vv, cc
        if i > 0:
            del ww
    L[n - 1, n - 1] = 1
    U[n - 1, n - 1] = A[n - 1, n - 1]
    # create matrix form of P
    Pm = np.zeros([n,n])
    for i in range(n):
        Pm[i, P[i]] = 1
    if P_mat == 0:
        return [P, L, U]
    else:
        return [Pm, L, U]

def check_banded(A, m):
    # A is a matrix and m is an integer
    # check if A is a sparse banded matrix with band m
    n = A.shape[0]
    for j in range(0, n):
        lo = max(0, j - m)
        hi = min(n, j + m + 1)
        for k in range(0, lo):
            if A[j, k] != 0:
                return False
        for k in range(hi, n):
            if A[j, k] != 0:
                return False
    return True

def lu_no_pivoting_banded(A, m):
    # A is a banded square matrix with band m; m = 0: diagonal; m = 1: tri-diagonal; m = n-1 full;
    # return a lower triangular matrix L with 1 on the main diagonal and an upper triangular matrix U, each with band m, s.t. LU = A
    A = A.copy()
    n = A.shape[0]
    if m > (n - 1):
        print "Input error"
        return
    if not check_banded(A, m):
        print "Matrix is not banded with band m"
        return
    # initialize the matrix L and U
    U = np.zeros([n, n])
    L = np.zeros([n, n])
    for i in range(0, n - 1):
        for j in range(i, min(n, i + m)):
            U[i, j] = A[i, j]
            L[j, i] = A[j, i] / U[i, i]
        for j in range(i + 1, min(n, i + m + 1)):
            for k in range(i + 1, min(n, i + m + 1)):
                A[j, k] -= L[j, i] * U[i, k]
    L[n - 1, n - 1] = 1
    U[n - 1, n - 1] = A[n - 1, n - 1]
    return [L, U]

def lu_row_pivoting_banded(A, m):
    # A is a square matrix
    # P is a permutation matrix simply changes the line of A
    # return a lower triangular matrix L with 1 on the main diagonal and an upper triangular matrix U, s.t. LU = PA
    A = A.copy()
    n = A.shape[0]
    if m > (n - 1):
        print "Input error"
        return
    if not check_banded(A, m):
        print "Matrix is not banded with band m"
        return
    # initialize the matrix P, L and U
    P = range(0, n)
    U = np.zeros([n, n])
    L = np.zeros([n, n])  # initialized as identity matrix
    for i in range(0, n - 1):
        i_max = i + np.argmax(abs(A[range(i,n), i]))
        # permute the lines
        vv = A[i, range(i,n)]
        #print vv
        A[i, i:n] = A[i_max, range(i,n)]
        A[i_max, range(i,n)] = vv
        #print A
        # update P
        cc = P[i]
        P[i] = P[i_max]
        P[i_max] = cc
        if i > 0:
            ww = L[i, range(0, n)]
            L[i, range(0, n)] = L[i_max, range(0, n)]
            L[i_max, range(0, n)] = ww

        for j in range(0, n):
            U[i, j] = A[i, j]
            L[j, i] = A[j, i] / U[i, i]
        for j in range(i + 1, n):
            for k in range(i + 1, n):
                A[j, k] -= L[j, i] * U[i, k]
        del vv, cc
        if i > 0:
            del ww
    L[n - 1, n - 1] = 1
    U[n - 1, n - 1] = A[n - 1, n - 1]
    return [P, L, U]

def linear_solve_lu_no_pivoting(A, b):
    # returns solution to the linear system of Ax = b
    [L, U] = lu_no_pivoting(A)
    y = forward_subst(L, b)
    x = backward_subst(U, y)
    return x

def linear_solve_lu_row_pivoting(A, b):
    # returns solution to the linear system of Ax = b
    [P, L, U] = lu_row_pivoting(A)
    y = forward_subst(L, np.dot(P, b))
    x = backward_subst(U, y)
    return x

def inverse(A):
    # A is a nonsingular square matrix, return A **(-1)
    n = A.shape[0]
    A_inv = np.zeros([n, n]) # initialization
    [L, U] = lu_no_pivoting(A)
    for k in range(0, n):
        b = np.zeros([n])
        b[k] = 1
        y = forward_subst(L, b)
        x = backward_subst(U, y)
        x = x[:, 0]
        A_inv[range(0, n), k] = x
    return A_inv

def inverse_row_pivoting(A):
    # A is a nonsingular square matrix, return A **(-1)
    n = A.shape[0]
    A_inv = np.zeros([n, n]) # initialization
    [P, L, U] = lu_row_pivoting(A)
    for k in range(0, n):
        b = np.zeros([n, 1])
        b[k] = 1
        y = forward_subst(L, np.dot(P, b))
        x = backward_subst(U, y)
        x = x[:, 0]
        A_inv[range(0, n), k] = x
    return A_inv

def Cholesky(A):
    # A is a SPD matrix
    # U is an upper triangular matrix s.t. U'U = A
    n = A.shape[0]
    U = np.zeros([n, n])
    for i in range(0, n):
        U[i, i] = A[i, i] ** 0.5
        for k in range(i + 1, n):
            U[i, k] = A[i, k] / U[i, i]
        for j in range(i + 1, n):
            #for k in range(0, j):
                #A[j, k] = A[k, j]  # not needed?
            for k in range(j, n):
                A[j, k] = A[j, k] - U[i, j] * U[i, k]
    U[n - 1, n - 1] = A[n - 1, n - 1] ** 0.5
    return U

def Cholesky_banded(A, m):
    # A is a SPD matrix with band m
    # U is an upper triangular matrix with band m s.t. U'U = A
    n = A.shape[0]
    if m > (n - 1):
        print "Input error"
        return
    if not check_banded(A, m):
        print "Matrix is not banded with band m"
        return
    U = np.zeros([n, n])
    for i in range(0, n):
        U[i, i] = A[i, i] ** 0.5
        for k in range(i + 1, min(n, i + m + 1)):
            U[i, k] = A[i, k] / U[i, i]
        for j in range(i + 1, min(n, i + m + 1)):
            # for k in range(0, j):
            #    A[j, k] = A[k, j]
            for k in range(j, min(n, i + m + 1)):
                A[j, k] -= U[i, j] * U[i, k]
    U[n - 1, n - 1] = A[n - 1, n - 1] ** 0.5
    return U

def linear_solve_Cholesky(A, b):
    U = Cholesky(A)
    U_t = np.transpose(U)
    y = forward_subst(U_t, b)
    x = backward_subst(U, y)
    return x

def linear_solve_Cholesky_banded(A, m, b):
    U = Cholesky_banded(A, m)
    U_t = np.transpose(U)
    y = forward_subst_banded(U_t, m, b)
    x = backward_subst_banded(U, m, y)
    return x