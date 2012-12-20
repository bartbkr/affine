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
from statsmodels.tsa.api import VAR
from statsmodels.base.model import LikelihoodModel
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.numdiff import (approx_hess, approx_fprime)
from operator import itemgetter
from scipy import optimize
from util import flatten, select_rows, retry

#debugging
import pdb

#############################################
# Create affine class system                   #
#############################################

class Affine(LikelihoodModel):
    """
    This class defines an affine model of the term structure
    """
    def __init__(self, yc_data, var_data, rf_rate=None, maxlags=4, freq='M',
                    latent=False, no_err=None):
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

        #generates mths: list of mths in yield curve data
        mths = self._mths_list()
        self.mths = mths

        assert len(yc_data.dropna(axis=0)) == len(var_data.dropna(axis=0)) \
                                                - k_ar + 1, \
            "Number of non-null values unequal in VAR and yield curve data"

        if latent:

            lat = self.lat = len(no_err)

            self.err = list(set(range(len(mths))).difference(no_err))

            self.pos_list = self._gen_pos_list()

            self.no_err_mth, self.err_mth = self._gen_mth_list()
            #gen position list for processing list input to solver
            self.noerr_cols, self.err_cols = self._gen_col_names()
            #set to unconditional mean of short_rate
            self.delta_0 = np.mean(rf_rate)

        #with all observed factors, mu, phi, and sigma are directly generated
        #from OLS VAR one step estimation
        else:
            self.delta_0 = 0
            self.lat = 0
            delta_1 = np.zeros([neqs*k_ar, 1])
            #delta_1 is vector of zeros, with one grabbing fed_funds rate
            delta_1[np.argmax(var_data.columns == 'fed_funds')] = 1
            self.delta_1_nolat = delta_1

        self.mu_ols, self.phi_ols, self.sigma_ols = self._gen_OLS_res()

        #maybe this should be done in setup script...
        #get VAR input data ready
        x_t_na = var_data.copy()
        for lag in range(k_ar-1):
            for var in var_data.columns:
                x_t_na[var + '_m' + str(lag + 1)] = px.Series(var_data[var].
                        values[:-(lag+1)], index=var_data.index[lag + 1:])

        var_data_vert = self.var_data_vert = x_t_na.dropna(axis=0)
        self.periods = len(self.var_data)

        super(Affine, self).__init__(var_data_vert)

    def solve(self, lam_0_g=None, lam_1_g=None, delta_1_g=None, mu_g=None,
              phi_g=None, sigma_g=None, method="ls", alg="newton",
              attempts=5, maxfev=10000, maxiter=10000, ftol=1e-100, 
              xtol=1e-100, full_output=False):
        """
        Attempt to solve affine model

        lam_0_g : array (neqs * k_ar + lat, 1)
            guess for elements of lambda_0
        lam_1_g : array (neqs * k_ar + lat, neqs * k_ar + lat)
            guess for elements of lambda_1
        delta_1_g : array (neqs * k_ar + lat, 1)
            guess for elements of delta_1
        mu_g : array (neqs * k_ar + lat, 1)
            guess for elements of mu
        phi_g : array (neqs * k_ar + lat, neqs * k_ar + lat)
            guess for elements of phi
        sigma_g : array (neqs * k_ar + lat, neqs * k_ar + lat)
            guess for elements of sigma
        method : string
            solution method
            ls = linear least squares
            nls = nonlinear least squares
            ml = maximum likelihood
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

        scipy.optimize.leastsq params
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
        #Notes for you:
        #remember that Ang and Piazzesi treat observed and unobserved factors
        #as orthogonal
        #observed factor parameters can thus be estimated using OLS
        k_ar = self.k_ar
        neqs = self.neqs
        lat = self.lat
        yc_data = self.yc_data
        var_data_vert = self.var_data_vert

        dim = neqs * k_ar + lat
        if lam_0_g is not None:
            assert np.shape(lam_0_g) == (dim, 1), "Shape of lam_0_g incorrect"
        if lam_1_g is not None:
            assert np.shape(lam_1_g) == (dim, dim), "Shape of lam_1_g incorrect"

        #creates single input vector for params to solve
        if lat:
            assert np.shape(delta_1_g) == (dim, 1), "Shape of delta_1_g" \
                "incorrect"
            assert np.shape(mu_g) == (dim, 1), "Shape of mu incorrect"
            assert np.shape(phi_g) == (dim, dim), "Shape of phi_g incorrect"
            assert np.shape(sigma_g) == (dim, dim), "Shape of sig_g incorrect"
            delta_1_g, mu_g, phi_g, sigma_g = \
                    self._pass_ols(delta_1=delta_1_g, mu=mu_g, phi=phi_g,
                                   sigma=sigma_g)

            params = self._params_to_list(lam_0=lam_0_g, lam_1=lam_1_g, 
                    delta_1=delta_1_g, mu=mu_g, phi=phi_g, sigma=sigma_g)

        else:
            params = self._params_to_list(lam_0=lam_0_g, lam_1=lam_1_g)

        if method == "ls":
            func = self._affine_nsum_errs
            solver = retry(optimize.leastsq, attempts)
            reslt = solver(func, params, maxfev=maxfev, xtol=xtol, full_output=full_output)
            solv_params = reslt[0]
            output = reslt[1:]

        elif method == "nls":
            func = self._affine_pred
            #need to stack
            yield_stack = self._stack_yields(yc_data)
            #run optmization
            solver = retry(optimize.curve_fit, attempts)
            reslt = solver(func, var_data_vert, yield_stack, p0=params,
                    maxfev=maxfev, xtol=xtol, full_output=full_output)
            solv_params = reslt[0]
            solv_cov = reslt[1]

        elif method == "ml":
            solver = retry(solf.fit, attempts)
            solve = solver(start_params=params, method=alg, maxiter=maxiter,
                    maxfun=maxfev, xtol=xtol, fargs=(lam_0_g, lam_1_g,
                        delta_1_g, mu_g, phi_g, sigma_g))

            solv_params = solve.params
            tvalues = solve.tvalues

        # elif method = "ml_angpiaz":
        #     func = self.something

        lam_0, lam_1, delta_1, mu, phi, sigma = \
                self._param_to_array(params=solv_params, delta_1=delta_1_g,
                                      mu=mu_g, phi=phi_g, sigma=sigma_g)

        a_solve, b_solve = self.gen_pred_coef(lam_0=lam_0, lam_1=lam_1,
                                              delta_1=delta_1, mu=mu, phi=phi,
                                              sigma=sigma)

        #This will need to be refactored
        #if full_output:
            #return lam_0, lam_1, delta_1, phi, sigma, a_solve, b_solve, output 
        if method == "nls":
            return lam_0, lam_1, delta_1, mu, phi, sigma, a_solve, b_solve, \
                    solv_cov
        elif method == "ls":
            return lam_0, lam_1, delta_1, mu, phi, sigma, a_solve, b_solve, \
                    output
        elif method == "ml":
            return lam_0, lam_1, delta_1, mu, phi, sigma, a_solve, b_solve, \
                    tvalues

    def score(self, params, *args):
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
        return approx_fprime(params, loglike, epsilon=1e-8, args=(args))

    def hessian(self, params, *args):
        """
        Returns numerical hessian.
        """
        #would be nice to have additional arguments here
        loglike = self.loglike
        my_stuff = args
        return approx_hess(params, loglike, args=(args))

    def loglike(self, params, lam_0, lam_1, delta_1, mu, phi, sigma):
        """
        Loglikelihood used in latent factor models
        """
        lat = self.lat
        per = self.periods

        #HERE all of the params don't seem to be moving
        #only seems to be for certain solution methods

        lam_0, lam_1, delta_1, mu, phi, sigma \
            = self._param_to_array(params=params, delta_1=delta_1, mu=mu, \
                                   phi=phi, sigma=sigma)

        #print lam_0

        solve_a, solve_b = self.gen_pred_coef(lam_0=lam_0, lam_1=lam_1, \
                                delta_1=delta_1, mu=mu, phi=phi, sigma=sigma)

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

    def gen_pred_coef(self, lam_0, lam_1, delta_1, mu, phi, sigma):
        """
        Generates prediction coefficient vectors A and B
        lam_0 : array
        lam_1 : array
        delta_1 : array
        phi : array
        sigma : array
        """
        mths = self.mths
        delta_0 = self.delta_0
        max_mth = max(mths)
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

    def _affine_nsum_errs(self, params):
        """
        This function generates the sum of the prediction errors
        """
        lat = self.lat
        mths = self.mths
        yc_data = self.yc_data
        var_data_vert = self.var_data_vert

        lam_0, lam_1, delta_1, mu, phi, sigma = self._param_to_array(params=params)

        a_solve, b_solve = self.gen_pred_coef(lam_0=lam_0, lam_1=lam_1,
                                              delta_1=delta_1, mu=mu,phi=phi,
                                              sigma=sigma)

        errs = []

        yc_data_val = yc_data.values
        
        for ix, mth in enumerate(mths):
            act = np.flipud(yc_data_val[:, ix])
            pred = a_solve[mth - 1] + np.dot(b_solve[mth - 1].T, 
                                        np.fliplr(var_data_vert.T))[0]
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

    def _param_to_array(self, params, delta_1=None, mu=None, phi=None,
            sigma=None):
        """
        Process params input into appropriate arrays

        Parameters
        ----------
        delta_1 : array (neqs * k_ar + lat, 1)
            delta_1 prior to complete model solve
        mu : array (neqs * k_ar + lat, 1)
            mu prior to complete model solve
        phi : array (neqs * k_ar + lat, neqs * k_ar + lat)
            phi prior to complete model solve
        sigma : array (neqs * k_ar + lat, neqs * k_ar + lat)
            sigma prior to complete model solve

        """
        lat = self.lat
        neqs = self.neqs
        k_ar = self.k_ar

        if lat:

            pos_list = self.pos_list

            lam_0_est = params[:pos_list[0]]
            lam_1_est = params[pos_list[0]:pos_list[1]]
            delta_1_g = params[pos_list[1]:pos_list[2]]
            mu_g = params[pos_list[2]:pos_list[3]]
            phi_g = params[pos_list[3]:pos_list[4]]
            sigma_g = params[pos_list[4]:]

            #lambda_0

            lam_0 = np.zeros([k_ar * neqs + lat, 1])
            lam_0[:neqs, 0] = np.asarray(lam_0_est[:neqs]).T
            lam_0[-lat:, 0] = np.asarray(lam_0_est[-lat:]).T

            #lambda_1

            lam_1 = np.zeros([k_ar * neqs + lat, k_ar * neqs + lat])
            lam_1[:neqs, :neqs] = np.reshape(lam_1_est[:neqs**2], (neqs, neqs))
            nxt = neqs*lat
            lam_1[:neqs, -lat:] = np.reshape(lam_1_est[neqs**2:\
                                            neqs**2 + nxt],(neqs,lat))
            nxt = nxt + neqs**2
            lam_1[-lat:, :neqs] = np.reshape(lam_1_est[nxt: \
                                            nxt+lat*neqs], (lat, neqs))
            nxt = nxt + lat * neqs
            lam_1[-lat:, -lat:] = np.reshape(lam_1_est[nxt: \
                                            nxt + lat**2], (lat, lat))

            #delta_1

            delta_1[-lat:, 0] = np.asarray(delta_1_g)

            #mu

            mu[-lat:, 0] = np.asarray(mu_g)

            #phi

            phi[-lat:, -lat:] = np.reshape(phi_g, (lat, lat))

            #sigma

            sigma[-lat:, -lat:] = np.reshape(sigma_g, (lat, lat))

        else:

            lam_0_est = params[:neqs]
            lam_1_est = params[neqs:]

            lam_0 = np.zeros([k_ar*neqs, 1])
            lam_0[:neqs] = np.asarray([lam_0_est]).T

            lam_1 = np.zeros([k_ar*neqs, k_ar*neqs])
            lam_1[:neqs, :neqs] = np.reshape(lam_1_est, (neqs, neqs))

            delta_1 = self.delta_1_nolat
            mu = self.mu_ols
            phi = self.phi_ols
            sigma = self.sigma_ols

        return lam_0, lam_1, delta_1, mu, phi, sigma

    def _affine_pred(self, data, *params):
        """
        Function based on lambda and data that generates predicted yields
        data : DataFrame
        params : tuple of floats
            parameter guess
        """
        mths = self.mths
        yc_data = self.yc_data

        lam_0, lam_1, delta_1, mu, phi, sigma = self._param_to_array(params)

        a_test, b_test = self.gen_pred_coef(lam_0, lam_1, delta_1, mu, phi,
                                            sigma)

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
    
    def _params_to_list(self, lam_0=None, lam_1=None, delta_1=None, mu=None,
                        phi=None, sigma=None):
        """
        Creates a single list of params from guess arrays that is passed into
        solver
        lam_0 : array (neqs * k_ar + lat, 1)
            guess for elements of lambda_0
        lam_1 : array (neqs * k_ar + lat, neqs * k_ar + lat)
            guess for elements of lambda_1
        delta_1 : array (neqs * k_ar + lat, 1)
            guess for elements of delta_1
        mu : array (neqs * k_ar + lat, 1)
            guess for elements of mu
        phi : array (neqs * k_ar + lat, neqs * k_ar + lat)
            guess for elements of phi
        sigma : array (neqs * k_ar + lat, neqs * k_ar + lat)
            guess for elements of sigma
        """
        #we will integrate standard assumptions
        #these could be changed later, but need to think of a standard way of
        #bring them in

        lat = self.lat
        neqs = self.neqs
        guess_list = []
        #we assume that those params corresponding to lags are set to zero
        if lat: 
            #we are assuming independence between macro factors and latent
            #factors
            guess_list.append(flatten(lam_0[:neqs]))
            guess_list.append(flatten(lam_0[-lat:]))
            guess_list.append(flatten(lam_1[:neqs, :neqs]))
            guess_list.append(flatten(lam_1[:neqs, -lat:]))
            guess_list.append(flatten(lam_1[-lat:, :neqs]))
            guess_list.append(flatten(lam_1[-lat:, -lat:]))
            guess_list.append(flatten(delta_1[-lat:, 0]))
            guess_list.append(flatten(mu[-lat:, 0]))
            guess_list.append(flatten(phi[-lat:, -lat:]))
            guess_list.append(flatten(sigma[-lat:, -lat:]))
        else:
            guess_list.append(flatten(lam_0[:neqs]))
            guess_list.append(flatten(lam_1[:neqs, :neqs]))

        #flatten this list into one dimension
        flatg_list = [item for sublist in guess_list for item in sublist]
        return flatg_list
    
    def _gen_OLS_res(self):
        """
        Runs VAR on macro data and retrieves parameters
        """
        #run VAR to generate parameters for known 
        var_data = self.var_data
        k_ar = self.k_ar
        neqs = self.neqs
        lat = self.lat
        freq = self.freq

        var_fit = VAR(var_data, freq=freq).fit(maxlags=k_ar)

        coefs = var_fit.params.values
        sigma_u = var_fit.sigma_u

        obs_var = neqs * k_ar

        mu = np.zeros([k_ar*neqs, 1])
        mu[:neqs] = coefs[0, None].T

        phi = np.zeros([k_ar * neqs, k_ar * neqs])
        phi[:neqs] = coefs[1:].T
        phi[neqs:obs_var, :(k_ar - 1) * neqs] = np.identity((k_ar - 1) * neqs)

        sigma = np.zeros([k_ar * neqs, k_ar * neqs])
        sigma[:neqs, :neqs] = sigma_u
        sigma[neqs:obs_var, neqs:obs_var] = np.identity((k_ar - 1) * neqs)
        
        return mu, phi, sigma

    def _pass_ols(self, delta_1, mu, phi, sigma):
        """
        Inserts estimated OLS parameters into appropriate matrices

        delta_1 : array (neqs * k_ar + lat, 1)
            guess for elements of delta_1
        mu : array (neqs * k_ar + lat, 1)
            guess for elements of mu
        phi : array (neqs * k_ar + lat, neqs * k_ar + lat)
            guess for elements of phi
        sig : array (neqs * k_ar + lat, neqs * k_ar + lat)
            guess for elements of sigma
        """
        k_ar = self.k_ar
        neqs = self.neqs
        macro = self.var_data.copy()[k_ar - 1:]

        macro["constant"] = 1
        delta_1[:neqs] = OLS(self.rf_rate,
                             macro).fit().params[1:].values[None].T
        mu[:neqs * k_ar, 0, None] = self.mu_ols[None]
        phi[:neqs * k_ar, :neqs * k_ar] = self.phi_ols[None]
        sigma[:neqs * k_ar, :neqs * k_ar] = self.sigma_ols[None]

        return delta_1, mu, phi, sigma

    def _ml_meth(self, params, lam_0_g, lam_1_g, delta_1_g, mu_g, phi_g, sigma_g):
        """
        This is a wrapper for the simple maximum likelihodd solution method
        """
        lam_0_g
        #solve_a = self.gen_pred_coef(lam_0=lam_0_g, lam_1_g=lam_1_g

    def _gen_pos_list(self):
        """
        Generates list of positions from draw parameters from list
        Notes: this is only lengths of parameters that we are solving for using
        numerical maximization
        """
        neqs = self.neqs
        k_ar = self.k_ar
        lat = self.lat

        pos_list = []
        pos = 0
        len_lam_0 = neqs + lat
        len_lam_1 = neqs**2 + (neqs * lat) + (lat * neqs) + lat**2
        len_delta_1 = lat
        len_mu = lat
        len_phi = lat * lat
        len_sig = lat * lat
        length_list = [len_lam_0, len_lam_1, len_delta_1, len_mu, len_phi,
                       len_sig]

        for length in length_list:
            pos_list.append(length + pos)
            pos += length

        return pos_list

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
        LEFT OFF here 
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
