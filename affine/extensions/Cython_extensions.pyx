from __future__ import division

cimport cython

#import numpy as np
import numpy as np
cimport numpy as np

# half to be used later
ctypedef np.float64_t DTYPE_t
cdef np.float half = 1 / 2

@cython.boundscheck(False) # turn of bounds-checking for entire function
cdef cython_gen_pred_coef(unsigned int max_mat,
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
    cdef unsigned int mat
    cdef unsigned int max_mat_m1 = max_mat - 1
    cdef int inc = 1
    #sizes needed
    cdef unsigned int factors = mu.shape[0]
    cdef unsigned int mu_size = mu.size
    cdef unisgned int factors_sqr = phi.size

    # generate predictions
    cdef np.ndarray[DTYPE_t, ndim=2] a_pre = np.empty((max_mat, 1),
                                                      dtype=np.float_)
    cdef np.ndarray[DTYPE_t, ndim=2] b_pre = np.empty((max_mat, factors),
                                                       dtype=np.float_)

    cdef np.ndarray[DTYPE_t, ndim=2] n_inv = 1 / np.add(range(max_mat),
                                                        1).reshape((max_mat,
                                                                    1))
    cdef np.ndarray[DTYPE_t, ndim=2] a_solve = -a_pre.copy()
    cdef np.ndarray[DTYPE_t, ndim=2] b_solve = -b_pre.copy()

    #indexes
    cdef unsigned int zero = 0
    cdef unsigned int one = 1

    #intermediate arrays
    cdef np.ndarray[DTYPE_t, ndim=2] mu_sig_lam_0 = np.empty((factors, 1),
                                                             order="F")

    a_pre[0] = -delta_0
    b_pre[0] = -delta_1[:,0]

    cdef np.ndarray[DTYPE_t, ndim=2] b_el_holder = np.empty((factors, 1),
                                                            order="F")
    cdef np.ndarray[DTYPE_t, ndim=2] mu_sigma_lam0 = np.empty((1, 1),
                                                              order="F")
    cdef np.ndarray[DTYPE_t, ndim=2] a_b_mu_sig_lam = np.empty((1, 1),
                                                               order="F")
    cdef np.ndarray[DTYPE_t, ndim=2] b_sig = np.empty((1, factors), order="F")
    cdef np.ndarray[DTYPE_t, ndim=2] b_sig_sig = np.empty((1, factors),
                                                          order="F")
    cdef np.ndarray[DTYPE_t, ndim=2] half_b_sig_sig_b_delta = np.empty((1, 1),
                                                                       order="F")
    cdef np.ndarray[DTYPE_t, ndim=2] phi_sigma_lam_1 = np.empty((factors,
                                                                 factors),
                                                                order="F")
    cdef np.ndarray[DTYPE_t, ndim=2] b_pre_prep = np.empty((factors, 1),
                                                           order="F")

    for mat in range(max_mat_m1):
        # Set next value of a
        #NOTE:set these to arrays to be referenced uniquely
        b_el_holder = b_pre[mat]
        a_b_mu_sig_lam = a_pre[mat]

        # This creates a filler array that initially has values of mu
        dcopy(&mu_size, &mu, &inc, &mu_sig_lam_0, &inc)
        # This creates a filler array that initially has values of delta_0
        dcopy(1, &delta_0, &inc, &half_b_sig_sig_b_delta, &inc)

        # dgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
        #NOTE: 13 args
        dgemm("N", "N", &factors, 1, &factors, -1, &sigma, &factors, &lam_0,
              &factors, 1, &mu_sigma_lam0, &factors)
        #NOTE: should I use a pointer for the slice of b_pre?
        dgemm("T", "N", 1, 1, &factors, 1, &b_el_holder, &factors,
              &mu_sigma_lam0, &factors, 1, &a_b_mu_sig_lam, 1)

        dgemm("T", "N", 1, &factors, &factors, 1, &b_el_holder, &factors,
              &sigma, &factors, 0, &b_sig, 1)
        dgemm("N", "T", 1, &factors, &factors, 1, &b_sig, 1, &sigma, &factors,
              0, &b_sig_sig)
        dgemm("N", "N", 1, 1, &factors, &half, &b_sig_sig, 1, &b_el_holder,
              &factors, -1, &half_b_sig_sig_b_delta, 1)
        a_pre[mat + 1] = a_b_mu_sig_lam[zero, zero] + a_b_mu_sig_lam[zero,
                                                                     zero]

        # Filler array that has initial values of phi
        dcopy(&factors_sqr, &phi, &inc, &phi_sigma_lam_1, &inc)
        # Filler array that has initial value of delta_1
        dcopy(&factors, &delta_1, &inc, &b_pre_prep, &inc)
        # set next value of b
        dgemm("N", "N", &factors, &factors, &factors, -1, &sigma, &factors,
              &lam_1, &factors, 1, &phi_sigma_lam_1, &factors)
        dgemm("T", "N", &factors, 1, &factors, 1, &phi_sigma_lam_1, &factors,
              &b_el_holder, &factors, -1, &b_pre_prep, &factors)
        b_pre[mat + 1] = b_pre_prep[:, zero]

    return np.multiply(-a_pre, n_inv), np.multiply(-b_pre, n_inv)
