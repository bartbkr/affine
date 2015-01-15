"""
The class provides Affine, intended to solve affine models of the term
structure and AffineResult, which contains the results of an estimated
model.
This class inherits from the statsmodels LikelihoodModel class
"""
import numpy as np
import pandas as pa
import scipy.linalg as la
import warnings

from numpy import linalg as nla
from numpy import ma
from scipy.optimize import fmin_l_bfgs_b
from statsmodels.base.model import LikelihoodModel, LikelihoodModelResults
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.numdiff import approx_hess, approx_fprime
from statsmodels.tsa.statespace.api import MLEModel
from scipy import optimize
from util import transform_var1

try:
    from . import _C_extensions
    avail_fast_gen_pred = True
except:
    avail_fast_gen_pred = False

###################
# Warning classes #
###################

#class ParameterIgnoreWarning(Warning):



#############################################
# Create affine class system                #
#############################################

class Affine(object):
    """
    Class for construction of affine model of the term structure
    """
    def __init__(self, yc_data, var_data, mats, lam_0_e, lam_1_e, delta_0_e,
                 delta_1_e, mu_e, phi_e, sigma_e, latent=0, k_ar=None,
                 neqs=None, no_err=None, adjusted=False, use_C_extension=True):
        """
        Attempts to instantiate an affine model object
        yc_data : DataFrame
            yield curve data
        var_data : DataFrame
            data for var model (pass None if no observed factors)
        mats : list of int
            Maturities in periods of yields included in yc_data
        latent: int
            Number of latent variables to estimate
        no_err : list of ints
            list of the column indexes of yields to be measured without
            error
            ex: [0, 3, 4]
            (1st, 4th, and 5th columns in yc_data to be estimated
            without error)
        adjusted : boolean
            Indicates whether data for VAR has already been transformed into
            VAR(1)
        use_C_extension : boolean
            Indicates whether to use C extension
        k_ar : int
            number of lags for VAR system
            Only respected when adjusted=False and var_data not None
        neqs : int
            Number of observed factors included
            Only respected when adjusted=True

        For all estimate parameter arrays:
            elements marked with 'E' or 'e' are estimated
            n = number of variables in fully-specified VAR(1) at t

        lam_0_e : Numpy masked array, n x 1
            constant vector of risk pricing equation
        lam_1_e : Numpy masked array, n x n
            parameter array of risk pricing equation
        delta_0_e : Numpy masked array, 1 x 1
            constant in short-rate equation
        delta_1_e : Numpy masked array, n x 1
            parameter vector in short-rate equation
        mu_e : Numpy masked array, n x 1
            constant vector for VAR process
        phi_e : Numpy masked array, n x n
            parameter array for VAR process
        sigma_e : Numpy masked array, n x n
            covariance array for VAR process
        """
        self.yc_data = yc_data
        self.var_data = var_data
        self.yc_names = yc_data.columns
        self.num_yields = len(yc_data.columns)
        self.yobs = len(yc_data)
        self.obs_factors = bool(var_data)

        if not self.obs_factors:
            warnings.warn("k_ar and neqs ignored when obs_factors not used")

        if not k_ar:
            self.k_ar = 0
        if not neqs:
            self.neqs = 0

        # Number of latent factors
        self.latent = latent

        # Arrays for setting up affine system
        self.lam_0_e = lam_0_e
        self.lam_1_e = lam_1_e
        self.delta_0_e = delta_0_e
        self.delta_1_e = delta_1_e

        self.mu_e = mu_e
        self.phi_e = phi_e
        self.sigma_e = sigma_e

        # generates mats: list of mats in yield curve data
        self.mats = mats
        self.max_mat = max(mats)
        self.mats_ix = [mat - 1 for mat in mats]

        self.no_err = no_err
        if no_err:
            # parameters for identification of yields measured without error
            self.err = list(set(range(len(mats))).difference(no_err))
            self.no_err_mat, self.err_mat = self._gen_mat_list()
            # gen position list for processing list input to solver
            self.noerr_cols, self.err_cols = self._gen_col_names()

        # whether to use C extension
        self.use_C_extension = use_C_extension
        if avail_fast_gen_pred and use_C_extension:
            self.fast_gen_pred = True
        else:
            self.fast_gen_pred = False

        self.adjusted = adjusted
        if self.obs_factors:
            if adjusted:
                assert len(yc_data.dropna(axis=0)) == \
                       len(var_data.dropna(axis=0)), \
                    "Number of non-null values unequal in VAR and yield curve data"
                var_data_vert = self.var_data_vert = var_data[ \
                                                     var_data.columns[:-neqs]]
                var_data_vertm1 = self.var_data_vertm1 = var_data[ \
                                                         var_data.columns[neqs:]]

            else:
                assert len(yc_data.dropna(axis=0)) == len(var_data.dropna(axis=0)) \
                                                        - k_ar, \
                    "Number of non-null values unequal in VAR and yield curve data"

                x_t_na = transform_var1(var_data, k_ar)

                var_data_vert = self.var_data_vert = x_t_na[x_t_na.columns[:-neqs]]
                var_data_vertm1 = self.var_data_vertm1 = \
                    x_t_na[x_t_na.columns[neqs:]]

            self.var_data_vertc = self.var_data_vert.copy()
            self.var_data_vert_T = np.array(self.var_data_vert.T,
                                            dtype=np.complex_)
            self.var_data_vertc.insert(0, "constant",
                                       np.ones((len(var_data_vert), 1)))
        else:
            self.var_data_vert = None
            self.var_data_vertc = None
            self.var_data_vert_T = None

        self.periods = len(self.yc_data)
        self.guess_length = self._gen_guess_length()
        assert self.guess_length > 0, "guess_length must be at least 1"

        # final size checks
        self._size_checks()

        #super(Affine, self).__init__(var_data_vert)
        self._init_extend()

    def solve(self, guess_params, method, alg="newton", attempts=5,
              maxfev=10000, maxiter=10000, ftol=1e-4, xtol=1e-4,
              unobs_known=None, burn=5, penalty=False, upperbounds=None,
              lowerbounds=None, full_output=False, disp=True, **kwargs):
        """
        Returns tuple of arrays
        Attempt to solve affine model based on instantiated object.

        Parameters
        ----------
        guess_params : list
            List of starting values for parameters to be estimated
            In row-order and ordered as masked arrays

        method : string
            solution method
            nls = nonlinear least squares
            ml = direct maximum likelihood
            kalman = kalman filter derived maximum likelihood
        alg : str {'newton','nm','bfgs','powell','cg', or 'ncg'}
            algorithm used for numerical approximation
            Method can be:
                'newton' for Newton-Raphson,
                'nm' for Nelder-Mead,
                'bfgs' for Broyden-Fletcher-Goldfarb-Shanno,
                'powell' for modified Powell's method,
                'cg' for conjugate gradient, or
                'ncg' for Newton- conjugate gradient. `method`
            determines which solver from scipy.optimize is used.

            Each solver has several optional arguments that are not the
            same across solvers. See the notes section below (or
            scipy.optimize) for the available arguments.

        scipy.optimize params
        maxfev : int
            maximum number of calls to the function for solution alg
        maxiter : int
            maximum number of iterations to perform
        ftol : float
            relative error desired in sum of squares
        xtol : float
            relative error desired in the approximate solution
        full_output : bool
            non_zero to return all optional outputs
        disp : boolean, optional
            Set to True to print convergence messages.

        Kalman filter params
        unobs_known: list
            Initial values of unobserved factors for state space iteration
        burn: int
            Number of periods to burn in evaluation of log-likelihood

        Returns
        -------
        Returns tuple contains each of the parameter arrays with the
        optimized values filled in:
        lam_0 : numpy array
        lam_1 : numpy array
        delta_0 : numpy array
        delta_1 : numpy array
        mu : numpy array
        phi : numpy array
        sigma : numpy array

        The final A, B, and parameter set arrays used to construct the
        yields
        a_solve : numpy array
        b_solve : numpy array
        solve_params : list

        Other results are also attached, depending on the solution
        method
        if 'nls':
            solv_cov : numpy array
                Contains the implied covariance matrix of solve_params
        if 'ml' and 'latent' > 0:
            var_data_wunob : numpy
                The modified factor array with the unobserved factors
                attached
        """
        k_ar = self.k_ar
        neqs = self.neqs
        mats = self.mats
        latent = self.latent
        yc_data = self.yc_data
        var_data_vert = self.var_data_vert

        # TODO: this should not need to be true, could still be
        # represented by statespace form
        if method == "kalman" and not self.latent:
            raise NotImplementedError( \
            "Kalman filter not supported with no latent factors")

        elif method == "nls":
            func = self._affine_pred
            var_data_vert_tpose = var_data_vert.T
            # need to stack for scipy nls
            yield_stack = np.array(yc_data).reshape(-1, order='F').tolist()
            # run optimization
            reslt = optimize.curve_fit(func, var_data_vert_tpose, yield_stack,
                                       p0=guess_params, maxfev=maxfev,
                                       xtol=xtol, ftol=ftol, full_output=True,
                                       disp=disp, **kwargs)
            solve_params = reslt[0]
            solv_cov = reslt[1]

        elif method == "ml":
            assert len(self.no_err) == self.latent, \
                "Number of columns estimated without error must match " + \
                "number of latent variables"

            if method == "bfgs-b":
                func = self.nloglike
                bounds = self._gen_bounds(lowerbounds, upperbounds)
                reslt = fmin_l_bfgs_b(x0=guess_params, approx_grad=True,
                                      bounds=bounds, m=1e7, maxfun=maxfev,
                                      maxiter=maxiter, **kwargs)
                solve_params = reslt[0]
                score = self.score(solve_params)

            else:
                # create object specifically for estimating through direct ML
                affineml = AffineML(yc_data=self.yc_data,
                                    var_data=self.var_data, k_ar=self.k_ar,
                                    neqs=self.neqs, mats=self.mats,
                                    lam_0_e=self.lam_0_e, lam_1_e=self.lam_1_e,
                                    delta_0_e=self.delta_0_e,
                                    delta_1_e=self.delta_1_e, mu_e=self.mu_e,
                                    phi_e=self.phi_e, sigma_e=self.sigma_e,
                                    latent=self.latent, no_err=self.no_err,
                                    adjusted=self.adjusted,
                                    use_C_extension=self.use_C_extension)

                reslt = affineml.fit(start_params=guess_params, method=alg,
                                     maxiter=maxiter, maxfun=maxfev, xtol=xtol,
                                     ftol=ftol, disp=disp, **kwargs)
                solve_params = reslt.params
                score = affineml.score(solve_params)
                self.estimation_mlresult = reslt

        elif method == "kalman":
            affinekalman = AffineKalman(yc_data=self.yc_data,
                                        var_data=self.var_data, k_ar=self.k_ar,
                                        neqs=self.neqs, mats=self.mats,
                                        lam_0_e=self.lam_0_e,
                                        lam_1_e=self.lam_1_e,
                                        delta_0_e=self.delta_0_e,
                                        delta_1_e=self.delta_1_e,
                                        mu_e=self.mu_e, phi_e=self.phi_e,
                                        sigma_e=self.sigma_e,
                                        latent=self.latent, no_err=self.no_err,
                                        adjusted=self.adjusted,
                                        use_C_extension=self.use_C_extension)

            reslt = affinekalman.fit(start_params=guess_params, method=alg,
                                     maxiter=maxiter, maxfun=maxfev, xtol=xtol,
                                     ftol=ftol, disp=disp,
                                     average_loglike=True, bfgs_tune=True,
                                     **kwargs)

            solve_params = reslt.mlefit.params
            score = affinekalman.score(solve_params)
            self.estimation_kalmanresult = reslt

        # once more to get final filled in values
        lam_0, lam_1, delta_0, delta_1, mu, phi, sigma = \
                self.params_to_array(solve_params)

        import ipdb;ipdb.set_trace()

        a_solve, b_solve = self.gen_pred_coef(lam_0, lam_1, delta_0, delta_1,
                                              mu, phi, sigma)

        if latent and method == "ml":
            lat_ser, jacob, yield_errs = self._solve_unobs(a_in=a_solve,
                                                           b_in=b_solve)
            var_data_wunob = var_data_vert.join(lat_ser)

        return AffineResult(self, solve_params)

    def gen_pred_coef(self, lam_0, lam_1, delta_0, delta_1, mu, phi, sigma):
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
        max_mat = self.max_mat
        b_width = self.k_ar * self.neqs + self.latent
        half = float(1)/2
        # generate predictions
        a_pre = np.zeros((max_mat, 1), dtype=np.complex_)
        a_pre[0] = -delta_0
        b_pre = np.zeros((max_mat, b_width), dtype=np.complex_)
        b_pre[0] = -delta_1[:,0]

        n_inv = float(1) / np.add(range(max_mat), 1).reshape((max_mat, 1))
        a_solve = -a_pre.copy()
        b_solve = -b_pre.copy()

        for mat in range(max_mat-1):
            a_pre[mat + 1] = (a_pre[mat] + np.dot(b_pre[mat].T, \
                            (mu - np.dot(sigma, lam_0))) + \
                            (half)*np.dot(np.dot(np.dot(b_pre[mat].T, sigma),
                            sigma.T), b_pre[mat]) - delta_0)[0][0]
            b_pre[mat + 1] = np.dot((phi - np.dot(sigma, lam_1)).T, \
                                     b_pre[mat]) - delta_1[:, 0]

        #import ipdb;ipdb.set_trace()

        return np.multiply(-a_pre, n_inv), np.multiply(-b_pre, n_inv)

    def opt_gen_pred_coef(self, lam_0, lam_1, delta_0, delta_1, mu, phi,
                          sigma):
        """
        Returns tuple of arrays
        Generates prediction coefficient vectors A and B in fast
        C function

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
        max_mat = self.max_mat

        return _C_extensions.gen_pred_coef(lam_0, lam_1, delta_0, delta_1, mu,
                                           phi, sigma, max_mat)

    def params_to_array(self, params, return_mask=False):
        """
        Returns tuple of arrays
        Process params input into appropriate arrays

        Parameters
        ----------
        params : list
            list of values to fill in masked values
        return_mask : boolean


        Returns
        -------
        lam_0 : numpy array
        lam_1 : numpy array
        delta_0 : numpy array
        delta_1 : numpy array
        mu : numpy array
        phi : numpy array
        sigma : numpy array
        """
        lam_0_e = self.lam_0_e.copy()
        lam_1_e = self.lam_1_e.copy()
        delta_0_e = self.delta_0_e.copy()
        delta_1_e = self.delta_1_e.copy()
        mu_e = self.mu_e.copy()
        phi_e = self.phi_e.copy()
        sigma_e = self.sigma_e.copy()

        all_arrays = [lam_0_e, lam_1_e, delta_0_e, delta_1_e, mu_e, phi_e,
                      sigma_e]

        arg_sep = self._gen_arg_sep([ma.count_masked(struct) for struct in \
                                     all_arrays])

        for pos, struct in enumerate(all_arrays):
            struct[ma.getmask(struct)] = params[arg_sep[pos]:arg_sep[pos + 1]]
            if not return_mask:
                all_arrays[pos] = np.ascontiguousarray(struct,
                                                       dtype=np.complex_)

        return tuple(all_arrays)

    def params_to_array_zeromask(self, params):
        """
        Returns tuple of arrays + list
        Process params input into appropriate arrays by setting them to
        zero if param in params in zero and removing them from params,
        otherwise they stay in params and value remains masked

        Parameters
        ----------
        params : list
            list of values to fill in masked values

        Returns
        -------
        lam_0 : numpy array
        lam_1 : numpy array
        delta_0 : numpy array
        delta_1 : numpy array
        mu : numpy array
        phi : numpy array
        sigma : numpy array
        guesses : list
            List of remaining params after filtering and filling those
            that were zero
        """
        paramcopy = params[:]
        lam_0_e = self.lam_0_e.copy()
        lam_1_e = self.lam_1_e.copy()
        delta_0_e = self.delta_0_e.copy()
        delta_1_e = self.delta_1_e.copy()
        mu_e = self.mu_e.copy()
        phi_e = self.phi_e.copy()
        sigma_e = self.sigma_e.copy()

        all_arrays = [lam_0_e, lam_1_e, delta_0_e, delta_1_e, mu_e, phi_e,
                      sigma_e]

        arg_sep = self._gen_arg_sep([ma.count_masked(struct) for struct in \
                                     all_arrays])

        guesses = []
        # check if each element is masked or not
        for struct in all_arrays:
            it = np.nditer(struct.mask, flags=['multi_index'])
            while not it.finished:
                if it[0]:
                    val = paramcopy.pop(0)
                    if val == 0:
                        struct[it.multi_index] = 0
                    else:
                        guesses.append(val)
                it.iternext()

        return tuple(all_arrays + [guesses])

    def _affine_pred(self, data, *params):
        """
        Function based on lambda and data that generates predicted
        yields
        data : DataFrame
        params : tuple of floats
            parameter guess
        """
        mats = self.mats
        yc_data = self.yc_data

        lam_0, lam_1, delta_0, delta_1, mu, phi, sigma \
                = self.params_to_array(params)

        if self.fast_gen_pred:
            solve_a, solve_b = self.opt_gen_pred_coef(lam_0, lam_1, delta_0,
                                                      delta_1, mu, phi, sigma)

        else:
            solve_a, solve_b = self.gen_pred_coef(lam_0, lam_1, delta_0,
                                                  delta_1, mu, phi, sigma)

        pred = []
        for i in mats:
            pred.extend((solve_a[i-1] + np.dot(solve_b[i-1], data)).tolist())
        return pred

    def _gen_arg_sep(self, arg_lengths):
        """
        Generates list of positions
        """
        arg_sep = [0]
        pos = 0
        for length in arg_lengths:
            arg_sep.append(length + pos)
            pos += length
        return arg_sep

    def _gen_col_names(self):
        """
        Generate column names for err and noerr
        """
        yc_names = self.yc_names
        no_err = self.no_err
        err = self.err
        noerr_cols = []
        err_cols = []
        for index in no_err:
            noerr_cols.append(yc_names[index])
        for index in err:
            err_cols.append(yc_names[index])
        return noerr_cols, err_cols

    def _gen_mat_list(self):
        """
        Generate list of mats measured with and wihout error
        """
        yc_names = self.yc_names
        no_err = self.no_err
        mats = self.mats
        err = self.err

        no_err_mat = []
        err_mat = []

        for index in no_err:
            no_err_mat.append(mats[index])
        for index in err:
            err_mat.append(mats[index])

        return no_err_mat, err_mat

    def _construct_J(self, b_obs, b_unobs, meas_mat):
        """
        Consruct jacobian matrix
        meas_mat : array
        """
        k_ar = self.k_ar
        neqs = self.neqs
        latent = self.latent
        num_yields = self.num_yields
        num_obsrv = neqs * k_ar

        msize = neqs * k_ar + num_yields
        jacob = np.zeros([msize, msize])
        jacob[:num_obsrv, :num_obsrv] = np.identity(neqs*k_ar)

        jacob[num_obsrv:, :num_obsrv] = b_obs
        jacob[num_obsrv:, num_obsrv:num_obsrv + latent] = b_unobs
        jacob[num_obsrv:, num_obsrv + latent:] = meas_mat

        return jacob

    def _gen_guess_length(self):
        lam_0_e = self.lam_0_e
        lam_1_e = self.lam_1_e
        delta_0_e = self.delta_0_e
        delta_1_e = self.delta_1_e
        mu_e = self.mu_e
        phi_e = self.phi_e
        sigma_e = self.sigma_e

        all_arrays = [lam_0_e, lam_1_e, delta_0_e, delta_1_e, mu_e, phi_e,
                      sigma_e]

        count = 0
        for struct in all_arrays:
            count += ma.count_masked(struct)

        return count

    def _size_checks(self):
        """
        Run size checks on parameter arrays
        """
        dim = self.neqs * self.k_ar + self.latent
        assert np.shape(self.lam_0_e) == (dim, 1), "Shape of lam_0_e incorrect"
        assert np.shape(self.lam_1_e) == (dim, dim), \
                "Shape of lam_1_e incorrect"

        assert np.shape(self.delta_1_e) == (dim, 1), "Shape of delta_1_e" \
            "incorrect"
        assert np.shape(self.mu_e) == (dim, 1), "Shape of mu incorrect"
        assert np.shape(self.phi_e) == (dim, dim), \
                "Shape of phi_e incorrect"
        assert np.shape(self.sigma_e) == (dim, dim), \
                "Shape of sig_e incorrect"

    def _gen_bounds(self, lowerbounds, upperbounds):
        if lowerbounds or upperbounds:
            bounds = []
            for bix in range(max(len(lowerbounds), len(upperbounds))):
                tbound = []
                if lowerbounds:
                    tbound.append(lowerbounds[bix])
                else:
                    tbound.append(-np.inf)
                if upperbounds:
                    tbound.append(upperbounds[bix])
                else:
                    tbound.append(np.inf)
                bounds.append(tuple(tbound))
        else:
            return None

    def _init_extend(self):
        pass

class AffineML(Affine, LikelihoodModel):
    """
    Estimation class in the case of Direct Maximimum Likelihood
    """
    def loglike(self, params):
        """
        Returns float
        Loglikelihood used in latent factor models

        Parameters
        ----------
        params : list
            Values of parameters to pass into masked elements of array

        Returns
        -------
        loglikelihood : float
        """

        latent = self.latent
        per = self.periods
        var_data_vert = self.var_data_vert
        var_data_vertm1 = self.var_data_vertm1

        lam_0, lam_1, delta_0, delta_1, mu, phi, \
            sigma = self.params_to_array(params)

        if self.fast_gen_pred:
            solve_a, solve_b = self.opt_gen_pred_coef(lam_0, lam_1, delta_0,
                                                      delta_1, mu, phi, sigma)

        else:
            solve_a, solve_b = self.gen_pred_coef(lam_0, lam_1, delta_0,
                                                  delta_1, mu, phi, sigma)

        # first solve for unknown part of information vector
        lat_ser, jacob, yield_errs  = self._solve_unobs(a_in=solve_a,
                                                           b_in=solve_b)

        # here is the likelihood that needs to be used
        # use two matrices to take the difference
        var_data_use = var_data_vert.join(lat_ser)[1:]
        var_data_usem1 = var_data_vertm1.join(lat_ser.shift())[1:]

        errors = var_data_use.values.T - mu - np.dot(phi,
                                                     var_data_usem1.values.T)
        sign, j_logdt = nla.slogdet(jacob)
        j_slogdt = sign * j_logdt

        sign, sigma_logdt = nla.slogdet(np.dot(sigma, sigma.T))
        sigma_slogdt = sign * sigma_logdt

        var_yields_errs = np.var(yield_errs, axis=1)

        like = -(per - 1) * j_slogdt - (per - 1) * 1.0 / 2 * sigma_slogdt - \
               1.0 / 2 * np.sum(np.dot(np.dot(errors.T, \
               la.inv(np.dot(sigma, sigma.T))), errors)) - (per - 1) / 2.0 * \
               np.log(np.sum(var_yields_errs)) - 1.0 / 2 * \
               np.sum(yield_errs**2/var_yields_errs[None].T)

        return like

    def nloglike(self, params):
        """
        Return negative loglikelihood
        Negative Loglikelihood used in latent factor models
        """
        like = self.loglike(params)
        return -like

    def score(self, params):
        """
        Return the gradient of the loglike at params

        Parameters
        ----------
        params : list

        Notes
        -----
        Return numerical gradient
        """
        loglike = self.loglike
        return approx_fprime(params, loglike, epsilon=1e-8)

    def hessian(self, params):
        """
        Returns numerical hessian.
        """
        loglike = self.loglike
        return approx_hess(params, loglike)

    def std_errs(self, params):
        """
        Return standard errors
        """
        hessian = self.hessian(params)
        std_err = np.sqrt(-np.diag(la.inv(hessian)))
        return std_err

    def _solve_unobs(self, a_in, b_in):
        """
        Solves for unknown factors

        Parameters
        ----------
        a_in : list of floats (periods)
            List of elements for A constant in factors -> yields
            relationship
        b_in : array (periods, neqs * k_ar + latent)
            Array of elements for B coefficients in factors -> yields
            relationship

        Returns
        -------
        var_data_c : DataFrame
            VAR data including unobserved factors
        jacob : array (neqs * k_ar + num_yields)**2
            Jacobian used in likelihood
        yield_errs : array (num_yields - latent, periods)
            The errors for the yields estimated with error
        """
        yc_data = self.yc_data
        var_data_vert = self.var_data_vert
        yc_names = self.yc_names
        num_yields = self.num_yields
        names = self.names
        k_ar = self.k_ar
        neqs = self.neqs
        latent = self.latent
        no_err = self.no_err
        err = self.err
        no_err_mat = self.no_err_mat
        err_mat = self.err_mat
        noerr_cols = self.noerr_cols
        err_cols = self.err_cols

        yc_data_names = yc_names.tolist()
        no_err_num = len(noerr_cols)
        err_num = len(err_cols)

        # need to combine the two matrices
        # these matrices will collect the final values
        a_all = np.zeros([num_yields, 1])
        b_all_obs = np.zeros([num_yields, neqs * k_ar])
        b_all_unobs = np.zeros([num_yields, latent])

        a_sel = np.zeros([no_err_num, 1])
        b_sel_obs = np.zeros([no_err_num, neqs * k_ar])
        b_sel_unobs = np.zeros([no_err_num, latent])
        for ix, y_pos in enumerate(no_err):
            a_sel[ix, 0] = a_in[no_err_mat[ix] - 1]
            b_sel_obs[ix, :] = b_in[no_err_mat[ix] - 1, :neqs * k_ar]
            b_sel_unobs[ix, :] = b_in[no_err_mat[ix] - 1, neqs * k_ar:]

            a_all[y_pos, 0] = a_in[no_err_mat[ix] - 1]
            b_all_obs[y_pos, :] = b_in[no_err_mat[ix] - 1][:neqs * k_ar]
            b_all_unobs[y_pos, :] = b_in[no_err_mat[ix] - 1][neqs * k_ar:]

        # now solve for unknown factors using long arrays
        unobs = np.dot(la.inv(b_sel_unobs),
                    yc_data.filter(items=noerr_cols).values.T - a_sel - \
                    np.dot(b_sel_obs, var_data_vert.T))

        # re-initialize a_sel, b_sel_obs, and b_sel_obs
        a_sel = np.zeros([err_num, 1])
        b_sel_obs = np.zeros([err_num, neqs * k_ar])
        b_sel_unobs = np.zeros([err_num, latent])
        for ix, y_pos in enumerate(err):
            a_all[y_pos, 0] =  a_sel[ix, 0] = a_in[err_mat[ix] - 1]
            b_all_obs[y_pos, :] = b_sel_obs[ix, :] = \
                    b_in[err_mat[ix] - 1][:neqs * k_ar]
            b_all_unobs[y_pos, :] = b_sel_unobs[ix, :] = \
                    b_in[err_mat[ix] - 1][neqs * k_ar:]

        yield_errs = yc_data.filter(items=err_cols).values.T - a_sel - \
                        np.dot(b_sel_obs, var_data_vert.T) - \
                        np.dot(b_sel_unobs, unobs)

        lat_ser = pa.DataFrame(index=var_data_vert.index)
        for factor in range(latent):
            lat_ser["latent_" + str(factor)] = unobs[factor, :]
        meas_mat = np.zeros((num_yields, err_num))

        for col_index, col in enumerate(err_cols):
            row_index = yc_data_names.index(col)
            meas_mat[row_index, col_index] = 1

        jacob = self._construct_J(b_obs=b_all_obs, b_unobs=b_all_unobs,
                                  meas_mat=meas_mat)


        return lat_ser, jacob, yield_errs

class AffineKalman(Affine, MLEModel):
    """
    Estimation class in the case of Kalman Filter Maximimum Likelihood
    """
    def _init_extend(self):
        """
        Extends default __init_ method with additional information needed for
        State Space MLEModel attributes and methods
        """
        yc_data = self.yc_data
        latent = self.latent
        mats_ix = self.mats_ix

        k_states = latent
        obs_cov = np.identity(len(mats_ix))
        selection = np.identity(latent)

        # initialization of State Space MLEModel object
        MLEModel.__init__(self, endog=yc_data, k_states=latent,
                       obs_cov=obs_cov, selection=selection)

        self.initialize_approximate_diffuse()

        latent_iter = range(latent)
        self._names = ['unobs' + str(ix + 1) for ix in latent_iter]
        self._latex_names = ['$X_{u' + str(ix + 1) + '}$' for ix in latent_iter]

    def update(self, params, *args, **kwargs):
        """
        Update variance and covariance
        """

        mats_ix = self.mats_ix
        latent = self.latent
        var_data_vert_T = self.var_data_vert_T

        # calcualte appropriate arrays, along with affine relationships
        # A and B
        lam_0, lam_1, delta_0, delta_1, mu, phi, sigma = \
            self.params_to_array(params=params)
        if self.fast_gen_pred:
            solve_a, solve_b = self.opt_gen_pred_coef(lam_0, lam_1, delta_0,
                                                      delta_1, mu, phi, sigma)
        else:
            solve_a, solve_b = self.gen_pred_coef(lam_0, lam_1, delta_0,
                                                  delta_1, mu, phi, sigma)

        self.design = solve_b[mats_ix, -latent:]
        # NOTE: because the observed factors are known, can we just work them
        # into the intercept term in the observation equation
        # NOTE: working on trying to make this robust to only latent factors
        # pricing the model
        if var_data_vert_T:
            self.obs_intercept = solve_a[mats_ix] + np.dot(solve_b[mats_ix,
                                                                   :-latent],
                                                           var_data_vert_T)
        else:
            self.obs_intercept = solve_a[mats_ix]
        #print(self.obs_intercept)

        # NOTE: in the case of only observed facotrs
        self.transition = phi[-latent:, -latent:]
        self.state_intercept = mu[-latent:, 0]
        self.state_cov = sigma[-latent:, -latent:]

        #import ipdb;ipdb.set_trace()

    # def transform_params(self, unconstrained):
    #     # Parameters must all be positive for likelihood evaluation.
    #     # This transforms parameters from unconstrained parameters
    #     # returned by the optimizer to ones that can be used in the model.
    #     return unconstrained**2

    # def untransform_params(self, constrained):
    #     # This transforms parameters from constrained parameters used
    #     # in the model to those used by the optimizer
    #     return constrained**0.5

    def _get_model_names(self, latex=False):
        return self._latex_names if latex else self._names

class AffineResult(LikelihoodModelResults, Affine):
    """
    Results class for estimated affine model
    """
    def __init__(self, model, params):
        """
        """
        super(AffineResult, self).__init__(model, params)
        lam_0, lam_1, delta_0, delta_1, mu, phi, sigma = \
            self.model.params_to_array(params)

        self.lam_0 = lam_0
        self.lam_1 = lam_1
        self.delta_0 = delta_0
        self.delta_1 = delta_1
        self.mu = mu
        self.phi = phi
        self.sigma = sigma

    @cache_readonly
    def predicted_yields(self):
        """
        Returns DataFrame of predicted yields for each observed maturity
        """
        mats = self.model.mats
        yc_data = self.model.yc_data
        var_data_vert = self.model.var_data_vert

        a_rsk, b_rsk = self.model.gen_pred_coef(lam_0=self.lam_0,
                                                lam_1=self.lam_1,
                                                delta_0=self.delta_0,
                                                delta_1=self.delta_1,
                                                mu=self.mu, phi=self.phi,
                                                sigma=self.sigma)

        yc_pred = pa.DataFrame(index=yc_data.index)
        for mat in mats:
            yc_pred[str(mat) + '_pred'] = a_rsk[mat - 1] + \
                    np.dot(b_rsk[mat - 1], var_data_vert.values.T)

        return yc_pred

    @cache_readonly
    def risk_neutral_yields(self):
        """
        Return DataFrame of risk neutral predicted yields for each
        observed maturity
        """
        mats = self.model.mats
        yc_data = self.model.yc_data
        var_data_vert = self.model.var_data_vert

        lam_0_nr = np.zeros_like(self.lam_0)
        lam_1_nr = np.zeros_like(self.lam_1)

        yc_rn_pred = pa.DataFrame(index=yc_data.index)

        a_rn, b_rn = self.model.gen_pred_coef(lam_0=lam_0_nr, lam_1=lam_1_nr,
                                        delta_0=self.delta_0,
                                        delta_1=self.delta_1, mu=self.mu,
                                        phi=self.phi, sigma=self.sigma)

        for mat in mats:
            yc_rn_pred[str(mat) + '_risk_neutral'] = a_rn[mat - 1] + \
                    np.dot(b_rn[mat - 1], var_data_vert.values.T)

        return yc_rn_pred

    @cache_readonly
    def term_premia(self):
        """
        Return DataFrame of implied term premia for each observed
        maturity. Calculated by taking the difference between predicted yield
        and risk neutral yield.
        """
        mats = self.model.mats

        pred = self.predicted_yields
        risk_neutral = self.risk_neutral_yields

        tp = pa.DataFrame(index=pred.index)

        for mat in mats:
            tp[str(mat) + '_tp'] = pred[str(mat) + '_pred'] - \
                    risk_neutral[str(mat) + '_risk_neutral']

        return tp

    def generate_yields(self, var_data, adjusted=False):
        """
        Returns DataFrame of yields given var_data.

        Assumes that var_data begins with last lag elemnt of VAR

        Parameters
        ----------
        var_data : DataFrame
            data for var model
        adjusted : boolean
            Indicates whether data for VAR has already been transformed into
            VAR(1)
        """
        assert len(var_data) == len(var_data.dropna()), \
            "Null values in var_data"

        neqs = self.model.neqs
        k_ar = self.model.k_ar
        mats = self.model.mats
        # generate affine prediction coefficients
        a_rsk, b_rsk = self.model.gen_pred_coef(lam_0=self.lam_0,
                                                lam_1=self.lam_1,
                                                delta_0=self.delta_0,
                                                delta_1=self.delta_1,
                                                mu=self.mu, phi=self.phi,
                                                sigma=self.sigma)
        if not adjusted:
            var_data = transform_var1(var_data, k_ar)
        var_data_vert = var_data[var_data.columns[:-neqs]]
        yields = pa.DataFrame(index=var_data_vert.index)
        for mat in mats:
            yields[str(mat) + '_pred'] = a_rsk[mat - 1] + \
                    np.dot(b_rsk[mat - 1], var_data_vert.values.T)
        return yields
