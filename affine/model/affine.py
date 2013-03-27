"""
This defines the class objection Affine, intended to solve affine models of the
term structure
This class inherits from statsmodels LikelihoodModel class
"""

import numpy as np
import statsmodels.api as sm
import pandas as px
import scipy.linalg as la
import re

from numpy import linalg as nla
from numpy import ma
from statsmodels.tsa.api import VAR
from statsmodels.base.model import LikelihoodModel
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.numdiff import approx_hess, approx_fprime
from operator import itemgetter
from scipy import optimize
from util import retry

import ipdb

#C extension
try:
    import _C_extensions
    fast_gen_pred = True
except:
    fast_gen_pred = False

#############################################
# Create affine class system                #
#############################################

class Affine(LikelihoodModel):
    """
    This class defines an affine model of the term structure
    """
    def __init__(self, yc_data, var_data, rf_rate=None, maxlags=4, freq='M',
                 latent=False, no_err=None, lam_0_e=None, lam_1_e=None,
                 delta_0_e=None, delta_1_e=None, mu_e=None, phi_e=None,
                 sigma_e=None, mths=None):
        """
        Attempts to solve affine model
        yc_data : DataFrame
            yield curve data
        var_data : DataFrame
            data for var model
        rf_rate : DataFrame
            rf_rate for short_rate, used in latent factor case
        max_lags: int
            number of lags for VAR system
        freq : string
            frequency of data
        no_err : list of ints
            list of the column indexes of yields to be measured without error
            ex: [0, 3, 4]
            (1st, 4th, and 5th columns in yc_data to be estimatd without error)

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
        self.rf_rate = rf_rate
        self.yc_names = yc_data.columns
        self.num_yields = len(yc_data.columns)
        self.names = names = var_data.columns
        k_ar = self.k_ar = maxlags
        neqs = self.neqs = len(names)
        self.freq = freq
        self.latent = latent
        self.no_err = no_err

        self.lam_0_e = lam_0_e
        self.lam_1_e = lam_1_e
        self.delta_0_e = delta_0_e
        self.delta_1_e = delta_1_e
        self.mu_e = mu_e
        self.phi_e = phi_e
        self.sigma_e = sigma_e

        #generates mths: list of mths in yield curve data
        #only works for data labels matching regular expression
        #should probably be phased out
        if mths is None:
            mths = self._mths_list()
        self.mths = mths
        self.max_mth = max(mths)

        assert len(yc_data.dropna(axis=0)) == len(var_data.dropna(axis=0)) \
                                                - k_ar + 1, \
            "Number of non-null values unequal in VAR and yield curve data"

        if latent:
            #assertions for correction passed in parameters
            lat = self.lat = len(no_err)

            self.err = list(set(range(len(mths))).difference(no_err))

            self.no_err_mth, self.err_mth = self._gen_mth_list()
            #gen position list for processing list input to solver
            self.noerr_cols, self.err_cols = self._gen_col_names()
            #set to unconditional mean of short_rate
            #self.delta_0 = np.mean(rf_rate)

        #with all observed factors, mu, phi, and sigma are directly generated
        #from OLS VAR one step estimation
        else:
            self.delta_0 = 0
            self.lat = 0
            delta_1 = np.zeros([neqs*k_ar, 1])
            #delta_1 is vector of zeros, with one grabbing fed_funds rate
            #this will need to be removed, is now specified in model setup
            delta_1[np.argmax(var_data.columns == 'fed_funds')] = 1
            self.delta_1_nolat = delta_1

        #self.mu_ols, self.phi_ols, self.sigma_ols = self._gen_OLS_res()

        #maybe this should be done in setup script...
        #get VAR input data ready
        x_t_na = var_data.copy()
        for lag in range(k_ar-1):
            for var in var_data.columns:
                x_t_na[var + '_m' + str(lag + 1)] = px.Series(var_data[var].
                        values[:-(lag+1)], index=var_data.index[lag + 1:])

        var_data_vert = self.var_data_vert = x_t_na.dropna(axis=0)
        self.periods = len(self.var_data)
        self.guess_length = self._gen_guess_length()

        #final size checks
        self._size_checks()

        super(Affine, self).__init__(var_data_vert)

    def solve(self, guess_params,  method="ls", alg="newton", attempts=5,
              maxfev=10000, maxiter=10000, ftol=1e-100, xtol=1e-100,
              full_output=False):
        """
        Attempt to solve affine model

        guess_params : list
            List of starting values for parameters to be estimated
            In row-order and ordered as masked arrays

        method : string
            solution method
            ls = linear least squares
            nls = nonlinear least squares
            ml = maximum likelihood
            angpiazml = ang and piazzesi multi-step ML
        alg : str {'newton','nm','bfgs','powell','cg', or 'ncg'}
            algorithm used for numerical approximation
            Method can be 'newton' for Newton-Raphson, 'nm' for Nelder-Mead,
            'bfgs' for Broyden-Fletcher-Goldfarb-Shanno, 'powell' for modified
            Powell's method, 'cg' for conjugate gradient, or 'ncg' for Newton-
            conjugate gradient. `method` determines which solver from
            scipy.optimize is used.  The explicit arguments in `fit` are passed
            to the solver.  Each solver has several optional arguments that are
            not the same across solvers.  See the notes section below (or
            scipy.optimize) for the available arguments.
        attempts : int
            Number of attempts to retry solving if singular matrix Exception
            raised by Numpy

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
        """
        k_ar = self.k_ar
        neqs = self.neqs
        lat = self.lat
        yc_data = self.yc_data
        var_data_vert = self.var_data_vert

        if method == "ls":
            func = self._affine_nsum_errs
            solver = retry(optimize.leastsq, attempts)
            reslt = solver(func, guess_params, maxfev=maxfev, xtol=xtol,
                           full_output=full_output)
            solve_params = reslt[0]
            output = reslt[1:]

        elif method == "nls":
            func = self._affine_pred
            #need to stack
            yield_stack = self._stack_yields(yc_data)
            #run optmization
            solver = retry(optimize.curve_fit, attempts)
            reslt = solver(func, var_data_vert, yield_stack, p0=guess_params,
                           maxfev=maxfev, xtol=xtol, full_output=True)
            solve_params = reslt[0]
            solv_cov = reslt[1]

        elif method == "ml":
            solver = retry(self.fit, attempts)
            solve = solver(start_params=guess_params, method=alg,
                           maxiter=maxiter, maxfun=maxfev, xtol=xtol)
            solve_params = solve.params
            tvalues = solve.tvalues

        elif method == "angpiazml":
            solve = solver(start_params=params, method=alg, maxiter=maxiter,
                    maxfun=maxfev, xtol=xtol)
            solve_params = solve.params
            tvalues = solve.tvalues

        lam_0, lam_1, delta_0, delta_1, mu, phi, sigma = \
                self._params_to_array(solve_params)

        a_solve, b_solve = self.gen_pred_coef(lam_0, lam_1, delta_0, delta_1,
                                              mu, phi, sigma)

        #This will need to be refactored
        #if full_output:
            #return lam_0, lam_1, delta_0, delta_1, phi, sigma, a_solve, b_solve, output
        if method == "nls":
            return lam_0, lam_1, delta_0, delta_1, mu, phi, sigma, a_solve, \
                   b_solve, solv_cov
        elif method == "ls":
            return lam_0, lam_1, delta_0, delta_1, mu, phi, sigma, a_solve, \
                   b_solve, output
        elif method == "ml":
            return lam_0, lam_1, delta_0, delta_1, mu, phi, sigma, a_solve, \
                   b_solve, tvalues

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
        #would be nice to have additional arguments here
        loglike = self.loglike
        return approx_fprime(params, loglike, epsilon=1e-8)

    def hessian(self, params):
        """
        Returns numerical hessian.
        """
        #would be nice to have additional arguments here
        loglike = self.loglike
        return approx_hess(params, loglike)

    def loglike(self, params):
        """
        Loglikelihood used in latent factor models
        """
        lat = self.lat
        per = self.periods

        #all of the params don't seem to be moving
        #only seems to be for certain solution methods

        lam_0, lam_1, delta_0, delta_1, mu, phi, sigma = self._params_to_array(params)

        if fast_gen_pred:
            solve_a, solve_b = self.opt_gen_pred_coef(lam_0, lam_1, delta_0,
                                                      delta_1, mu, phi, sigma)

        else:
            solve_a, solve_b = self.gen_pred_coef(lam_0, lam_1, delta_0,
                                                  delta_1, mu, phi, sigma)

        #first solve for unknown part of information vector
        var_data_c, jacob, yield_errs  = self._solve_unobs(a_in=solve_a,
                                                           b_in=solve_b)

        # here is the likelihood that needs to be used
        # sigma is implied VAR sigma
        # use two matrices to take the difference

        errors = var_data_c.values.T[:, 1:] - mu - np.dot(phi,
                var_data_c.values.T[:, :-1])

        sign, j_logdt = nla.slogdet(jacob)
        j_slogdt = sign * j_logdt

        sign, sigma_logdt = nla.slogdet(np.dot(sigma, sigma.T))
        sigma_slogdt = sign * sigma_logdt

        like = -(per - 1) * j_slogdt - (per - 1) * 1.0 / 2 * sigma_slogdt - \
               1.0 / 2 * np.sum(np.dot(np.dot(errors.T, \
               la.inv(np.dot(sigma, sigma.T))), errors)) - (per - 1) / 2.0 * \
               np.log(np.sum(np.var(yield_errs, axis=1))) - 1.0 / 2 * \
               np.sum(yield_errs**2/np.var(yield_errs, axis=1)[None].T)

        return like

    def gen_pred_coef(self, lam_0, lam_1, delta_0, delta_1, mu, phi, sigma):
        """
        Generates prediction coefficient vectors A and B
        lam_0 : array
        lam_1 : array
        delta_0 : array
        delta_1 : array
        phi : array
        sigma : array
        """
        #Thiu should be passed to a C function, it is really slow right now
        #Should probably set this so its not recalculated every run
        max_mth = self.max_mth
        #generate predictions
        a_pre = np.zeros((max_mth, 1))
        a_pre[0] = -delta_0
        b_pre = []
        b_pre.append(-delta_1)

        for mth in range(max_mth-1):
            a_pre[mth+1] = (a_pre[mth] + np.dot(b_pre[mth].T, \
                            (mu - np.dot(sigma, lam_0))) + \
                            (1.0/2)*np.dot(np.dot(np.dot(b_pre[mth].T, sigma), \
                            sigma.T), b_pre[mth]) - delta_0)[0][0]
            b_pre.append(np.dot((phi - np.dot(sigma, lam_1)).T, \
                                b_pre[mth]) - delta_1)
        n_inv = 1.0/np.add(range(max_mth), 1).reshape((max_mth, 1))
        a_solve = -(a_pre*n_inv)
        b_solve = np.zeros_like(b_pre)
        for mth in range(max_mth):
            b_solve[mth] = np.multiply(-b_pre[mth], n_inv[mth])
        return a_solve, b_solve

    def opt_gen_pred_coef(self, lam_0, lam_1, delta_0, delta_1, mu, phi,
                          sigma):
        """
        Generation prediction coefficient vectors A and B in fast C function
        lam_0 : array
        lam_1 : array
        delta_0 : array
        delta_1 : array
        phi : array
        sigma : array
        """
        max_mth = self.max_mth

        #should probably do some checking here

        return _C_extensions.gen_pred_coef(lam_0, lam_1, delta_0, delta_1, mu,
                                           phi, sigma, max_mth)

    def _affine_nsum_errs(self, params):
        """
        This function generates the sum of the prediction errors
        """
        #This function is slow
        lat = self.lat
        mths = self.mths
        yc_data = self.yc_data
        var_data_vert = self.var_data_vert

        lam_0, lam_1, delta_0, delta_1, mu, phi, sigma = \
            self._params_to_array(params=params)

        if fast_gen_pred:
            solve_a, solve_b = self.opt_gen_pred_coef(lam_0, lam_1, delta_0,
                                                      delta_1, mu, phi, sigma)

        else:
            solve_a, solve_b = self.gen_pred_coef(lam_0, lam_1, delta_0,
                                                  delta_1, mu, phi, sigma)
        errs = []

        yc_data_val = yc_data.values

        for ix, mth in enumerate(mths):
            act = yc_data_val[:, ix]
            pred = a_solve[mth - 1] + np.dot(b_solve[mth - 1].T,
                                             var_data_vert.T)[0]
            errs = errs + (act - pred).tolist()
        return errs

    def _solve_unobs(self, a_in, b_in):
        """
        Solves for unknown factors

        Parameters
        ----------
        a_in : list of floats (periods)
            List of elements for A constant in factors -> yields relationship
        b_in : array (periods, neqs * k_ar + lat)
            Array of elements for B coefficients in factors -> yields
            relationship

        Returns
        -------
        var_data_c : DataFrame
            VAR data including unobserved factors
        jacob : array (neqs * k_ar + num_yields)**2
            Jacobian used in likelihood
        yield_errs : array (num_yields - lat, periods)
            The errors for the yields estimated with error
        """
        yc_data = self.yc_data
        var_data_vert = self.var_data_vert
        yc_names = self.yc_names
        num_yields = self.num_yields
        names = self.names
        k_ar = self.k_ar
        neqs = self.neqs
        lat = self.lat
        no_err = self.no_err
        err = self.err
        no_err_mth = self.no_err_mth
        err_mth = self.err_mth
        noerr_cols = self.noerr_cols
        err_cols = self.err_cols

        yc_data_names = yc_names.tolist()
        no_err_num = len(noerr_cols)
        err_num = len(err_cols)

        #need to combine the two matrices
        #these matrices will collect the final values
        a_all = np.zeros([num_yields, 1])
        b_all_obs = np.zeros([num_yields, neqs * k_ar])
        b_all_unobs = np.zeros([num_yields, lat])

        a_sel = np.zeros([no_err_num, 1])
        b_sel_obs = np.zeros([no_err_num, neqs * k_ar])
        b_sel_unobs = np.zeros([no_err_num, lat])
        for ix, y_pos in enumerate(no_err):
            a_sel[ix, 0] = a_in[no_err_mth[ix] - 1]
            b_sel_obs[ix, :, None] = b_in[no_err_mth[ix] - 1][:neqs * k_ar]
            b_sel_unobs[ix, :, None] = b_in[no_err_mth[ix] - 1][neqs * k_ar:]

            a_all[y_pos, 0] = a_in[no_err_mth[ix] - 1]
            b_all_obs[y_pos, :, None] = b_in[no_err_mth[ix] - 1][:neqs * k_ar]
            b_all_unobs[y_pos, :, None] = \
                    b_in[no_err_mth[ix] - 1][neqs * k_ar:]
        #now solve for unknown factors using long matrices

        unobs = np.dot(la.inv(b_sel_unobs),
                    yc_data.filter(items=noerr_cols).values.T - a_sel - \
                    np.dot(b_sel_obs, var_data_vert.values.T))

        #re-initialize a_sel, b_sel_obs, and b_sel_obs
        a_sel = np.zeros([err_num, 1])
        b_sel_obs = np.zeros([err_num, neqs * k_ar])
        b_sel_unobs = np.zeros([err_num, lat])
        for ix, y_pos in enumerate(err):
            a_all[y_pos, 0] =  a_sel[ix, 0] = a_in[err_mth[ix] - 1]
            b_all_obs[y_pos, :, None] = b_sel_obs[ix, :, None] = \
                    b_in[err_mth[ix] - 1][:neqs * k_ar]
            b_all_unobs[y_pos, :, None] = b_sel_unobs[ix, :, None] = \
                    b_in[err_mth[ix] - 1][neqs * k_ar:]

        yield_errs = yc_data.filter(items=err_cols).values.T - a_sel - \
                        np.dot(b_sel_obs, var_data_vert.values.T) - \
                        np.dot(b_sel_unobs, unobs)

        var_data_c = var_data_vert.copy()
        for factor in range(lat):
            var_data_c["latent_" + str(factor)] = unobs[factor, :]
        meas_mat = np.zeros((num_yields, err_num))

        for col_index, col in enumerate(err_cols):
            row_index = yc_data_names.index(col)
            meas_mat[row_index, col_index] = 1

        jacob = self._construct_J(b_obs=b_all_obs,
                                    b_unobs=b_all_unobs, meas_mat=meas_mat)

        return var_data_c, jacob, yield_errs

    def _mths_list(self):
        """
        This function just grabs the mths of yield curve points and return
        a list of them
        """
        mths = []
        columns = self.yc_names
        matcher = re.compile(r"(.*?)([0-9]+)$")
        for column in columns:
            mths.append(int(re.match(matcher, column).group(2)))
        return mths

    def _params_to_array(self, params):
        """
        Process params input into appropriate arrays

        Parameters
        ----------
        params : list
            guess parameters
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

        return tuple(all_arrays)

    def _affine_pred(self, data, *params):
        """
        Function based on lambda and data that generates predicted yields
        data : DataFrame
        params : tuple of floats
            parameter guess
        """
        mths = self.mths
        yc_data = self.yc_data

        lam_0, lam_1, delta_0, delta_1, mu, phi, sigma \
                = self._params_to_array(params)

        if fast_gen_pred:
            solve_a, solve_b = self.opt_gen_pred_coef(lam_0, lam_1, delta_0,
                                                      delta_1, mu, phi, sigma)

        else:
            solve_a, solve_b = self.gen_pred_coef(lam_0, lam_1, delta_0,
                                                  delta_1, mu, phi, sigma)

        pred = px.DataFrame(index=yc_data.index)

        for i in mths:
            pred["l_tr_m" + str(i)] = a_test[i-1] + np.dot(b_test[i-1].T,
                                      data.T).T[:,0]

        pred = self._stack_yields(pred)

        return pred

    def _stack_yields(self, orig):
        """
        Stacks yields into single column ndarray
        """
        mths = self.mths
        obs = len(orig)
        new = np.zeros((len(mths) * obs))
        for col, mth in enumerate(orig.columns):
            new[col*obs:(col+1)*obs] = orig[mth].values
        return new

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

    def _gen_mth_list(self):
        """
        Generate list of mths measured with and wihout error
        """
        yc_names = self.yc_names
        no_err = self.no_err
        mths = self.mths
        err = self.err

        no_err_mth = []
        err_mth = []

        for index in no_err:
            no_err_mth.append(mths[index])
        for index in err:
            err_mth.append(mths[index])

        return no_err_mth, err_mth

    def _construct_J(self, b_obs, b_unobs, meas_mat):
        """
        Consruct jacobian matrix
        meas_mat : array
        """
        k_ar = self.k_ar
        neqs = self.neqs
        lat = self.lat
        num_yields = self.num_yields
        num_obsrv = neqs * k_ar

        #now construct Jacobian
        msize = neqs * k_ar + num_yields
        jacob = np.zeros([msize, msize])
        jacob[:num_obsrv, :num_obsrv] = np.identity(neqs*k_ar)

        jacob[num_obsrv:, :num_obsrv] = b_obs
        jacob[num_obsrv:, num_obsrv:num_obsrv + lat] = b_unobs
        jacob[num_obsrv:, num_obsrv + lat:] = meas_mat

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
        dim = self.neqs * self.k_ar + self.lat
        assert np.shape(self.lam_0_e) == (dim, 1), "Shape of lam_0_e incorrect"
        assert np.shape(self.lam_1_e) == (dim, dim), \
                "Shape of lam_1_e incorrect"

        if self.latent:
            assert np.shape(self.delta_1_e) == (dim, 1), "Shape of delta_1_e" \
                "incorrect"
            assert np.shape(self.mu_e) == (dim, 1), "Shape of mu incorrect"
            assert np.shape(self.phi_e) == (dim, dim), \
                    "Shape of phi_e incorrect"
            assert np.shape(self.sigma_e) == (dim, dim), \
                    "Shape of sig_e incorrect"
