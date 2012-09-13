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

from statsmodels.tsa.api import VAR
from statsmodels.base.model import LikelihoodModel
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.numdiff import (approx_hess, approx_fprime)
from operator import itemgetter
from scipy import optimize
from util import flatten, select_rows

#debugging
import pdb

#############################################
# Create affine class system                   #
#############################################

class Affine(LikelihoodModel):
    """
    This class defines an affine model of the term structure
    """
    def __init__(self, yc_data, var_data, rf_rate=None, maxlags=4,
                 freq='M', latent=0, no_err=None):
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
        latent : int
            # number of latent variables
        no_err : list of ints
            list of the yields by number of periods that are estimated without error
            ex: [1, 6, 12] 
            (1, 6, and 12 period yields measured without error)
        """
        self.yc_data = yc_data
        self.var_data = var_data
        self.num_yields = len(yc_data.columns)
        self.names = names = var_data.columns
        self.k_ar = maxlags
        self.neqs = len(names)
        self.no_err = no_err
        self.freq = freq
        lat = self.latent = latent

        #generates mths: list of mths in yield curve data
        pre, mths = self._mths_list()
        self.mths = mths

        self.mu_ols, self.phi_ols, self.sigma_ols = self._gen_OLS_res()

        if lat:
            assert len(no_err) >= lat, "One yield estimated with no err" \
                                        + "for each latent variable"
            #gen position list for processing list input to solver
            self.pos_list = self._gen_pos_list()

            pdb.set_trace()

            self.noerr_indx = list(set(mths).intersection(no_err)).sort()
            self.err_indx = list(set(mths).difference(no_err)).sort()
            self.noerr_cols, self.err_cols = self._gen_col_names(pre)
            #set to unconditional mean of short_rate
            self.delta_0 = np.mean(rf_rate)

        #with all observed factors, mu, phi, and sigma are directly generated
        #from OLS VAR one step estimation
        else:
            self.delta_0 = 0
            delta_1 = np.zeros([neqs*k_ar, 1])
            #delta_1 is vector of zeros, with one grabbing fed_funds rate
            delta_1[np.argmax(var_data.columns == 'fed_funds')] = 1
            self.delta_1_nolat = delta_1

        #maybe this should be done in setup script...
        #get VAR input data ready
        x_t_na = var_data.copy()
        for lag in range(k_ar-1):
            for var in var_data.columns:
                x_t_na[var + '_m' + str(lag+1)] = px.Series(var_data[var].
                        values[:-(lag+1)], index=var_data.index[lag+1:])

        self.var_data_vert = x_t_na.dropna(axis=0)
        self.periods = len(self.var_data)

        super(Affine, self).__init__(var_data)

    def solve(self, lam_0_g=None, lam_1_g=None, delta_1_g=None, mu_g=None,
            phi_g=None, sigma_g=None, method="ls", maxfev=10000, ftol=1e-100,
            xtol=1e-100, full_output=False):
        """
        Attempt to solve affine model

        method : string
            ls = linear least squares
            cf = nonlinear least squares
            ml = maximum likelihood
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
        sig_g : array (neqs * k_ar + lat, neqs * k_ar + lat)
            guess for elements of sigma

        scipy.optimize.leastsq params
        maxfev : int
            maximum number of calls to the function for solution alg
        ftol : float
            relative error desired in sum of squares
        xtol : float
            relative error desired in the approximate solution
        full_output : bool
            non_zero to return all optional outputs
        """
        #Notes for you Bart:
        #remember that Ang and Piazzesi treat observed and unobserved factors
        #as orthogonal
        #observed factor parameters can thus be estimated using OLS
        k_ar = self.k_ar
        neqs = self.neqs
        lat = self.latent
        yc_data = self.yc_data

        dim = neqs * k_ar + lat

        #creates single input vector for params to solve
        assert np.shape(lam_0_g) == (dim, 1), "Shape of lam_0_g incorrect"
        assert np.shape(lam_1_g) == (dim, dim), "Shape of lam_1_g incorrect"
        if lat:
            assert np.shape(delta_1_g) == (dim, 1), "Shape of delta_1_g" \
                "incorrect"
            assert np.shape(mu_g) == (dim, 1), "Shape of mu incorrect"
            assert np.shape(phi_g) == (dim, dim), "Shape of phi_g incorrect"
            assert np.shape(sig_g) == (dim, dim), "Shape of sig_g incorrect"
            delta_1_g, mu_g, phi_g, sigma_g = \
                    self._pass_ols(delta_1=delta_1_g, mu=mu_g, phi=phi_g,
                                   sigma=sigma_g)

            params = self._params_to_list(lam_0=lam_0_g, lam_1=lam_1_g, \
                    delta_1=delta_1_g, mu=mu_g, phi=phi_g, sigma=sig_g)

        else:
            params = []
            params = self._params_to_list(lam_0=lam_0_g, lam_1=lam_1_g)

        #this should be specified in function call
        if method == "ls":
            func = self._affine_nsum_errs
            reslt = optimize.leastsq(func, params, maxfev=maxfev,
                                xtol=xtol, full_output=full_output)
            lam_solv = reslt[0]
            output = reslt[1:]

        elif method == "cf":
            func = self._affine_pred
            #need to stack
            yield_stack = self._stack_yields(yc_data)
            #run optmization
            reslt = optimize.curve_fit(func, var_data_vert, yield_stack, p0=params,
                                       maxfev=maxfev, xtol=xtol,
                                       full_output=full_output)
            lam_solv = reslt[0]
            lam_cov = reslt[1]

        elif method == "ml":
            #reslt = optmize.ne(func, var_data_vert, yi)
            lam_solv = super(Affine).fit(start_params=params, method=solver,
                    maxiter=maxiter, maxfun=maxfev, xtol=xtol,
                    fargs=(lam_0_g, lam_1_g, delta_1_g, mu_g, phi_g, sigma_g))

        # elif method = "mlangpiz":
        #     func = self.something

        lam_0, lam_1, delta_1, phi, sigma = \
                self._params_to_array(delta_1=delta_1_g, mu=mu_g, phi=phi_g,
                        sigma=sigma_g, *lam_solv)

        a_solve, b_solve = self.gen_pred_coef(lam_0=lam_0, lam_1=lam_1,
                                              delta_1=delta_1, mu=mu, phi=phi,
                                              sigma=sigma)

        #if full_output:
            #return lam_0, lam_1, delta_1, phi, sigma, a_solve, b_solve, output 
        if method == "cf":
            return lam_0, lam_1, delta_1, phi, sigma, a_solve, b_solve, lam_cov
        elif method == "ls":
            return lam_0, lam_1, delta_1, phi, sigma, a_solve, b_solve, output

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
        loglike = self._affine_nsum_errs
        return approx_hess(params, loglike)[0]

    def loglike(self, params, lam_0, lam_1, delta_1, mu, phi, sigma):
        """
        Loglikelihood used in latent factor models
        """
        lat = self.latent
        per = self.periods

        delta_1, mu, phi, sigma = _params_to_array(params=params, \
                                    delta_1=delta_1, mu=mu, phi=phi, \
                                    sigma=sigma)

        solve_a, solve_b = self.gen_pred_coef(lam_0=lam_0, lam_1=lam_1, \
                                delta_1=delta_1, mu=mu, phi=phi, sigma=sigma)
        #first solve for unknown part of information vector
        var_data_c, jacob, yield_errs  = self._solve_unobs(solve_a=solve_a,
                                                solve_b=solve_b)


        # here is the likelihood that needs to be used
        # sigma is implied VAR sigma
        # use two matrices to take the difference
        errors = var_data_c.values.T[2:] - mu - np.dot(phi,
                var_data_c.values.T[:-1])


        like = -(per - 1) * np.logdet(jacob) - (per - 1) * 1.0 / 2 * \
               np.logdet(np.dot(sigma, sigma.T)) - 1.0 / 2 * \
               np.sum(np.dot(np.dot(errors.T, np.inv(np.dot(sigma, sigma.T))),\
                             errors)) - (per - 1) / 2.0 * \
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
        lat = self.latent
        mths = self.mths
        yc_data = self.yc_data
        x_t = self.var_data_vert

        lam_0, lam_1, delta_1, mu, phi, sigma = self._params_to_array(*params)

        a_solve, b_solve = self.gen_pred_coef(lam_0, lam_1, delta_1, phi, sigma)

        errs = []
        
        for i in mths:
            act = np.flipud(yc_data['l_tr_m' + str(i)].values)
            pred = a_solve[i-1] + np.dot(b_solve[i-1].T, np.fliplr(x_t.T))[0]
            errs = errs + (act-pred).tolist()
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
        var_data = self.var_data_vert
        names = self.names
        k_ar = self.k_ar
        neqs = self.neqs
        no_err = self.no_err
        lat = self.latent
        noerr_cols = self.noerr_cols
        err_cols = self.err_cols

        names = var_data.columns
        no_err_num = len(noerr_cols)
        err_num = len(err_cols)

        a_sel = np.zeros([no_err_num, 1])
        b_sel_obs = np.zeros([no_err_num, neqs*k_ar])
        b_sel_unobs = np.zeros([no_err_num, lat])
        for indx, period in enumerate(no_err):
            a_sel[indx, 0] = a_in[period-1]
            b_sel_obs[indx, :] = b_in[period-1][:neqs * k_ar]
            b_sel_unobs = b_in[period-1][neqs * k_ar:]
        #now solve for unknown factors using long matrices
        unobs = np.dot(la.inv(b_sel_unobs), \
                    (ycdata.filter(items=noerr_cols).values.T - a_sel - \
                    np.dot(b_sel_obs, var_data.values.T)))

        yield_errs = ycdata.filter(items=err_cols).values.T - a_sel - \
                        np.dot(b_sel_obs, var_data.values.T) - \
                        np.dot(b_sel_unobs, unobs)

        var_data_c = var_data.copy()
        for factor in range(lat):
            var_data_c["latent_" + str(factor)] = unobs[factor, :]
        meas_mat = np.zeros(neqs * k_ar + lat, err_num)
        for col_index, col in enumerate(err_cols):
            row_index = names.index(col)
            meas_mat[row_index, col_index] = 1

        jacob = self._construct_J(b_sel_obs=b_sel_obs, \
                                    b_sel_unobs=b_sel_unobs, meas_mat=meas_mat)
        
        return var_data_c, jacob, yield_errs 

    def _mths_list(self):
        """
        This function just grabs the mths of yield curve points and return
        a list of them
        """
        mths = []
        columns = self.yc_data.columns
        matcher = re.compile(r"(.*)([0-9]+)$")
        for column in columns:
            pre = re.match(matcher, column).group(1)
            print pre
            mths.append(re.match(matcher, column).group(2))
            print mths
        return pre, mths

    def _params_to_array(self, params=None, delta_1=None, mu=None, phi=None,
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
        lat = self.latent
        neqs = self.neqs
        k_ar = self.k_ar

        if lat:

            pos_lst = self.pos_lst

            lam_0_est = params[:pos_lst[0]]
            lam_1_est = params[pos_lst[0]:pos_lst[1]]
            delta_1_g = params[pos_lst[1]:pos_lst[2]]
            mu_g = params[pos_lst[2]:pos_lst[3]]
            phi_g = params[pos_lst[3]:pos_lst[4]]
            sigma_g = params[pos_lst[4]:]

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

    def _affine_pred(self, x_t, *params):
        """
        Function based on lambda and x_t that generates predicted yields
        x_t : X_inforionat
        """
        mths = self.mths
        yc_data = self.yc_data

        lam_0, lam_1, delta_1, phi, sigma = self._params_to_array(*params)

        a_test, b_test = self.gen_pred_coef(lam_0, lam_1, delta_1, phi, sigma)

        pred = px.DataFrame(index=yc_data.index)

        for i in mths:
            pred["l_tr_m" + str(i)] = a_test[i-1] + np.dot(b_test[i-1].T,
                                      x_t.T).T[:,0]

        pred = self._stack_yields(pred)

        return pred

    def _stack_yields(self, orig):
        """
        Stacks yields into single column ndarray
        """
        mths = self.mths
        obs = len(orig)
        new = np.zeros((len(mths)*obs))
        for col, mth in enumerate(orig.columns):
            new[col*obs:(col+1)*obs] = orig[mth].values
        return new
    
    def _params_to_list(lam_0=None, lam_1=None, delta_1=None, mu=None,
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
        #see _params_to_array

        lat = self.latent
        neqs = self.neqs
        guess_list = []
        #we assume that those params corresponding to lags are set to zero
        if lat >= 1:
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
        lat = self.latent
        freq = self.freq

        var_fit = VAR(var_data, freq=freq).fit(maxlags=k_ar)

        coefs = var_fit.params.values
        sigma_u = var_fit.sigma_u

        mu = np.zeros([k_ar*neqs+lat, 1])
        mu[:neqs] = coefs[0, None].T

        phi = np.zeros([k_ar*neqs, k_ar*neqs])
        phi[:neqs] = coefs[1:].T
        phi[neqs:, :(k_ar-1)*neqs] = np.identity((k_ar-1)*neqs)

        sigma = np.zeros([k_ar*neqs, k_ar*neqs])
        sigma[:neqs, :neqs] = sigma_u
        
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
        macro = self.var_data
        macro["constant"] = 1
        delta_1[:neqs] = OLS(self.rf_rate, macro).fit().params[1:].values
        mu[:neqs*k_ar, 0] = self.mu_ols
        phi[:neqs*k_ar, :neqs] = self.phi_ols
        sigma[:neqs, :neqs] = self.sigma_ols

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
        lat = self.latent

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

    def _gen_col_names(self, pre_name):
        """
        Generate column names for err and noerr
        """
        noerr_indx = self.noerr_indx
        err_indx = self.err_indx
        noerr_cols = []
        err_cols = []
        for col in noerr_indx:
            noerr_cols.append(pre_name + str(col))
        for col in err_indx:
            err_cols.append(pre_name + str(col))
        return noerr_cols, err_cols

    def _construct_J(self, b_sel_obs, b_sel_unobs, meas_mat): 
        """
        Consruct jacobian matrix
        meas_mat : array 
        LEFT OFF here 
        """
        k_ar = self.k_ar
        neqs = self.neqs
        lat = self.latent
        num_yields = self.num_yields
        num_obs = neqs * k_ar

        #now construct Jacobian
        msize = neqs * k_ar + num_yields
        jacob = np.zeros([msize, msize])
        jacob[:num_bs, :num_obs] = np.identity(neqs*k_ar)
        jacob[num_obs:, :num_obs] = b_sel_obs
        jacob[num_obs:, num_obs:num_obs + lat] = b_sel_unobs
        jacob[num_obs:, num_obs + lat:] = meas_mat

        return jacob
