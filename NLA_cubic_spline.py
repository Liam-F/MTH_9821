from __future__ import division
import numpy as np
import NMF_linear_solve as LS

def cubic_spline(x, v):
    """
    Natural cubic spline interpolation function
    :param x: interpolation nodes, numpy array with dim [n + 1, 1]
    :param v: interpolation values, numpy array with dim [n + 1, 1]
    :return: [a, b, c, d] cubic polynomials coefficients, 4 [n, 1] numpy arrays
    """
    n = x.shape[0] - 1
    z = np.zeros([n, 1])
    for i in range(1, n):
        z[i] = 6 * ((v[i + 1] - v[i]) / (x[i + 1] - x[i]) - (v[i] - v[i - 1]) / (x[i] - x[i - 1]))

    M = np.zeros([n, n])
    for i in range(1, n):
        M[i, i] = 2 * (x[i + 1] - x[i - 1])
    for i in range(1, n - 1):
        M[i, i + 1] = x[i + 1] - x[i]
    for i in range(2, n):
        M[i, i - 1] = x[i] - x[i - 1]
    w = np.zeros([n + 1, 1])
    w[1:n,:] = LS.linear_solve_lu_no_pivoting(M[1:n, 1:n], z[1:n])
    #print M[1:n, 1:n]

    a, b, c, d = np.zeros([n + 1,1]), np.zeros([n + 1,1]), np.zeros([n + 1,1]), np.zeros([n + 1,1])
    for i in range(1, n + 1):
        c[i] = (w[i - 1] * x[i] - w[i] * x[i - 1]) / (2 * (x[i] - x[i - 1]))
        d[i] = (w[i] - w[i - 1]) / (6 * (x[i] - x[i - 1]))
        q = np.zeros([n, 1])
        r = np.zeros([n + 1, 1])
    for i in range(1, n + 1):
        q[i - 1] = v[i - 1] - c[i] * x[i - 1] ** 2 - d[i] * x[i - 1] ** 3
        r[i] = v[i] - c[i] * x[i] ** 2 - d[i] * x[i] ** 3
    for i in range(1, n + 1):
        a[i] = (q[i - 1] * x[i] - r[i] * x[i - 1]) / (x[i] - x[i - 1])
        b[i] = (r[i] - q[i - 1]) / (x[i] - x[i - 1])

    return[a[1:n + 1], b[1:n + 1], c[1:n + 1], d[1:n + 1]]

def cubic_spline_interpolate(x_new, x, v):
    [a, b, c, d] = cubic_spline(x, v)
    n = x.shape[0] - 1
    v_new = 0
    for i in range(0, n):
        if x_new >= x[i] and x_new < x[i + 1]:
            a_v, b_v, c_v, d_v = a[i], b[i], c[i], d[i]
            v_new = a_v + b_v * x_new + c_v * x_new ** 2 + d_v * x_new ** 3
    return v_new

def main():
    t = np.array([0, 2/12, 6/12, 1, 20/12])
    r = np.array([0.005, 0.0065, 0.0085, 0.0105, 0.012])
    t = np.row_stack(t)
    r = np.row_stack(r)
    [a, b, c, d] = cubic_spline(t, r)
    print a, b, c, d

if __name__ == "__main__":
    main()
