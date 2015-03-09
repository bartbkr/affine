from __future__ import division

#import numpy as np
import numpy as np
cimport numpy as np

# half to be used later
cdef np.float64 half = 1 / 2
ctypedef np.float

def gen_pred_coef(self, int max_mat,
                  np.ndarray[DTYPE_t, ndim=2] lam_0,
                  np.ndarray[DTYPE_t, ndim=2] lam_1,
                  np.ndarray[DTYPE_t, ndim=2] delta_0,
                  np.ndarray[DTYPE_t, ndim=2] delta_1,
                  np.ndarray[DTYPE_t, ndim=2] mu,
                  np.ndarray[DTYPE_t, ndim=2] phi,
                  np.ndarray[DTYPE_t, ndim=2] sigma):
    """
    Returns tuple of arrays
    Generates prediction coefficient vectors A and B

    Parameters
    ----------
    lam_0 : numpy array
    lam_1 : numpy array
    delta_0 : numpy array
    delta_1 : numpy array
    mu : numpy array
    phi : numpy array
    sigma : numpy array

    Returns
    -------
    a_solve : numpy array
        Array of constants relating factors to yields
    b_solve : numpy array
        Array of coeffiencts relating factors to yields
    """
    cdef int mat
    cdef max_mat_m1 = max_mat - 1
    cdef int delta_1_rows=delta_1.shape[0]
    # generate predictions
    cdef np.ndarray a_pre = np.zeros((max_mat, 1), dtype=np.complex_)
    cdef np.ndarray b_pre = np.zeros((max_mat, delta_1_rows), dtype=np.complex_)

    cdef np.ndarray n_inv = 1 / np.add(range(max_mat), 1).reshape((max_mat, 1))
    cdef np.ndarray a_solve = -a_pre.copy()
    cdef np.ndarray b_solve = -b_pre.copy()

    a_pre[0] = -delta_0
    b_pre[0] = -delta_1[:,0]

    for mat in range(max_mat_m1):
        a_pre[mat + 1] = (a_pre[mat] + np.dot(b_pre[mat].T, \
                         (mu - np.dot(sigma, lam_0))) + \
                         (half)*np.dot(np.dot(np.dot(b_pre[mat].T, sigma),
                          sigma.T), b_pre[mat]) - delta_0)[0][0]
        b_pre[mat + 1] = np.dot((phi - np.dot(sigma, lam_1)).T, \
                                 b_pre[mat]) - delta_1[:, 0]

    return np.multiply(-a_pre, n_inv), np.multiply(-b_pre, n_inv)
