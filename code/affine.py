"""
This defines the class objection Affine, intended to solve affine models of the
term structure
This class inherits from statsmodels LikelihoodModel class
"""

import numpy as np
import statsmodels.api as sm
import pandas as px
import scipy.linalg as la

from statsmodels.tsa.api import VAR
from statsmodels.base.model import LikelihoodModel
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
        no_err : list of strings
            list of the yields that are estimated without error
        """
        self.yc_data = yc_data
        self.var_data = var_data
        self.k_ar = max_lags
        self.neqs = len(var_data.columns)
        self.no_err = no_err
        self.freq = freq
        lat = self.latent = latent

        #generates mths and mth_only
        self._proc_to_mth()

        self.mu, self.phi, self.sigma = self._gen_OLS_res()

        if lat:
            assert len(no_err) >= lat, "One yield estimated with no err" \
                                        + "for each latent variable"
            yc_data_cols = yc_data.columns.tolist()
            self.noerr_indx = list(set(yc_data_cols).intersection(no_err))
            self.err_indx = list(set(yc_data_cols).difference(no_err))
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

        self.var_data = x_t_na.dropna(axis=0)

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
        mth_only = self.mth_only

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

            lam = self._params_to_list(lam_0=lam_0_g, lam_1=lam_1_g, \
                    delta_1=delta_1_g, mu=mu_g, phi=phi_g, sigma=sig_g)

        else:
            lam = []
            self._params_to_list(lam_0=lam_0_g, lam_1=lam_1_g)
            lam_0_list = flatten(lam_0_g[:neqs])
            lam_1_list = flatten(lam_1_g[:neqs, :neqs])
            for param in lam_0_list:
                lam.append(param)
            for param in lam_1_list:
                lam.append(param)

        #this should be specified in function call
        if method == "ls":
            func = self._affine_nsum_errs
            reslt = optimize.leastsq(func, lam, maxfev=maxfev,
                                xtol=xtol, full_output=full_output)
            lam_solv = reslt[0]
            output = reslt[1:]

        elif method == "cf":
            func = self._affine_pred
            #need to stack
            yield_stack = self._stack_yields(mth_only)
            #run optmization
            reslt = optimize.curve_fit(func, var_data, yield_stack, p0=lam,
                                       maxfev=maxfev, xtol=xtol,
                                       full_output=full_output)

            lam_solv = reslt[0]
            lam_cov = reslt[1]
        elif method = "ml":
            funct = self.

        lam_0, lam_1, delta_1, phi, sigma = self._proc_lam(*lam_solv)

        a_solve, b_solve = self.gen_pred_coef(lam_0, lam_1, delta_1, phi, sigma)

        #if full_output:
            #return lam_0, lam_1, delta_1, phi, sigma, a_solve, b_solve, output 
        if method == "cf":
            return lam_0, lam_1, delta_1, phi, sigma, a_solve, b_solve, lam_cov
        elif method == "ls":
            return lam_0, lam_1, delta_1, phi, sigma, a_solve, b_solve, output

    def score(self, lam):
        """
        Return the gradient of the loglike at AB_mask.

        Parameters
        ----------
        AB_mask : unknown values of A and B matrix concatenated

        Notes
        -----
        Return numerical gradient
        """
        loglike = self._affine_nsum_errs
        return approx_fprime(lam, loglike, epsilon=1e-8)

    def hessian(self, lam):
        """
        Returns numerical hessian.
        """
        loglike = self._affine_nsum_errs
        return approx_hess(lam, loglike)[0]

    #def loglike(self, params):
    #    """
    #    Loglikelihood used in latent factor models
    #    """
    #    # here is the likelihood that needs to be used
    #    # sigma is implied VAR sigma
    #    # use two matrices to take the difference
    #    like = -(T - 1) * np.logdet(J) - (T - 1) * 1.0 / 2 * \
    #            np.logdet(np.dot(sigma, sigma.T)) - 1.0 / 2 * \
    #            np.sum(np.dot(np.dot(errors.T, np.inv(np.dot(sigma, sigma.T))),\
    #                          err)) - (T - 1) / 2.0 * \
    #            np.log(np.sum(np.var(meas_err, axis=1))) - 1.0 / 2 * \
    #            np.sum(meas_err/np.var(meas_err, axis=1))

    def gen_pred_coef(self, lam_0_ab, lam_1_ab, delta_1, phi, sigma):
        """
        Generates prediction coefficient vectors A and B
        lam_0_ab : array
        lam_1_ab : array
        delta_1 : array
        phi : array
        sigma : array
        """
        mths = self.mths
        delta_0 = self.delta_0
        mu = self.mu
        max_mth = max(mths)
        #generate predictions
        a_pre = np.zeros((max_mth, 1))
        a_pre[0] = -delta_0
        b_pre = []
        b_pre.append(-delta_1)
        for mth in range(max_mth-1):
            a_pre[mth+1] = (a_pre[mth] + np.dot(b_pre[mth].T, \
                            (mu - np.dot(sigma, lam_0_ab))) + \
                            (1.0/2)*np.dot(np.dot(np.dot(b_pre[mth].T, sigma), \
                            sigma.T), b_pre[mth]) - delta_0)[0][0]
            b_pre.append(np.dot((phi - np.dot(sigma, lam_1_ab)).T, \
                                b_pre[mth]) - delta_1)
        n_inv = 1.0/np.add(range(max_mth), 1).reshape((max_mth, 1))
        a_solve = -(a_pre*n_inv)
        b_solve = np.zeros_like(b_pre)
        for mths in range(max_mth):
            b_solve[mths] = np.multiply(-b_pre[mths], n_inv[mths])
        return a_solve, b_solve

    def _affine_nsum_errs(self, lam):
        """
        This function generates the sum of the prediction errors
        """
        lat = self.latent
        mths = self.mths
        mth_only = self.mth_only
        x_t = self.var_data

        lam_0, lam_1, delta_1, phi, sigma = self._proc_lam(*lam)

        a_solve, b_solve = self.gen_pred_coef(lam_0, lam_1, delta_1, phi, sigma)

        #this is explosive

        if lat:
            x_t = self._solve_x_t_unkn(a_solve, b_solve)

        errs = []
        
        for i in mths:
            act = np.flipud(mth_only['l_tr_m' + str(i)].values)
            pred = a_solve[i-1] + np.dot(b_solve[i-1].T, np.fliplr(x_t.T))[0]
            errs = errs + (act-pred).tolist()
        return errs

    def _solve_x_t_unkn(self, a_in, b_in, x_t = None):
        """
        This is still under development
        It should solve for the unobserved factors in the x_t VAR data
        """
        lat = self.latent
        no_err = self.no_err
        mth_only = self.mth_only
        yc_data = self.yc_data
        x_t_new = np.append(x_t, np.zeros((x_t.shape[0], lat)), axis=1)
        errors = x_t[1:] - mu - np.dot(phi, x_t[:-1])
        if x_t is None:
            x_t = self.var_data
        T = x_t.shape[0]

        # solve for unknown factors
        noerr_indx = self.noerr_indx
        a_noerr = select_rows(noerr_indx, a_in)
        b_0_noerr = select_rows(noerr_indx, b_in)
        # this is the right hand for solving for the unobserved latent 
        # factors
        r_hs = yc_data[no_err] - a_noerr[None].T - np.dot(b_0_noerr, x_t)
        lat = la.solve(b_u, r_hs)

        #solve for pricing error on other yields
        err_indx = self.err_indx
        a_err = select_rows(err_indx, a_in)
        b_0_err = select_rows(err_indx, b_in)
        r_hs = yc_data[no_err] - a_noerr[None].T - np.dot(b_0_noerr, x_t)
        meas_err = la.solve(b_m, r_hs)

        #create Jacobian (J) here
        
        #this taken out for test run, need to be added back in
        #J = 

    def _proc_to_mth(self):
        """
        This function transforms the yield curve data so that the names are all
        in months
        (not sure if this is necessary)
        """
        frame = self.yc_data
        mths = []
        fnd = 0
        n_cols = len(frame.columns)
        for col in frame.columns:
            if 'm' in col:
                mths.append(int(col[6]))
                if fnd == 0:
                    mth_only = px.DataFrame(frame[col],
                            columns = [col],
                            index=frame.index)
                    fnd = 1
                else:
                    mth_only[col] = frame[col]
            elif 'y' in col:
                mth = int(col[6:])*12
                mths.append(mth)
                mth_only[('l_tr_m' + str(mth))] = frame[col]
        col_dict = dict([( mth_only.columns[x], mths[x]) for x in
                    range(n_cols)])
        cols = np.asarray(sorted(col_dict.iteritems(),
                        key=itemgetter(1)))[:,0].tolist()
        mth_only = mth_only.reindex(columns = cols)
        mths.sort()
        self.mths = mths
        self.mth_only = mth_only

    #def _unk_likl(self):
    #    likl = -(T-1)*np.logdet(J) - (T-1)*1.0/2*np.logdet(np.dot(sigma,\
    #            sigma.T)) - 1.0/2*

    def _proc_lam(self, *lam):
        """
        Process lam input into appropriate parameters
        """
        lat = self.latent
        neqs = self.neqs
        k_ar = self.k_ar

        if lat:

            pos_lst = self.pos_lst

            lam_0_est = lam[:pos_lst[0]]
            lam_1_est = lam[pos_lst[0]:pos_lst[1]]
            delt_1_g = lam[pos_lst[1]:pos_lst[2]]
            phi_g = lam[pos_lst[2]:pos_lst[3]]
            sig_g = lam[pos_lst[3]:]

            lam_0 = np.zeros([k_ar*neqs+lat, 1])
            lam_0[:neqs, 0] = np.asarray(lam_0_est[:neqs]).T
            lam_0[-lat:, 0] = np.asarray(lam_0_est[-lat:]).T

            lam_1 = np.zeros([k_ar*neqs+lat, k_ar*neqs+lat])
            lam_1[:neqs, :neqs] = np.reshape(lam_1_est[:neqs**2], (neqs, neqs))
            nxt = neqs*lat
            lam_1[:neqs, -lat:] = np.reshape(lam_1_est[neqs**2:\
                                            neqs**2 + nxt],(neqs,lat))
            nxt = nxt + neqs**2
            lam_1[-lat:, :neqs] = np.reshape(lam_1_est[nxt: \
                                            nxt+lat*neqs], (lat, neqs))
            nxt = nxt + lat*neqs
            lam_1[-lat:, -lat:] = np.reshape(lam_1_est[nxt: \
                                            nxt + lat**2], (lat, lat))
            delta_1 = self.delta_1.copy()
            delta_1[-lat:, 0] = np.asarray(delt_1_g)

            #add rows/columns for unk params
            phi_n = self.phi.copy()
            add = np.zeros([lat, np.shape(phi_n)[1]])
            phi_n = np.append(phi_n, add, axis=0)
            add = np.zeros([np.shape(phi_n)[0], lat])
            phi = np.append(phi_n, add, axis=1)
            #fill in parm guesses
            phi[-lat:, -lat:] = np.reshape(phi_g, (lat, lat))

            #add rows/columns for unk params
            sig_n = self.sigma.copy()
            add = np.zeros([lat, np.shape(sig_n)[1]])
            sig_n = np.append(sig_n, add, axis=0)
            add = np.zeros([np.shape(sig_n)[0], lat])
            sigma = np.append(sig_n, add, axis=1)
            sigma[-lat:, -lat:] = np.reshape(sig_g, (lat, lat))

        else:
            lam_0_est = lam[:neqs]
            lam_1_est = lam[neqs:]

            lam_0 = np.zeros([k_ar*neqs, 1])
            lam_0[:neqs] = np.asarray([lam_0_est]).T

            lam_1 = np.zeros([k_ar*neqs, k_ar*neqs])
            lam_1[:neqs, :neqs] = np.reshape(lam_1_est, (neqs, neqs))

            delta_1 = self.delta_1_nolat
            phi = self.phi_ols
            sigma = self.sigma_ols

        return lam_0, lam_1, delta_1, phi, sigma

    def _affine_pred(self, x_t, *lam):
        """
        Function based on lambda and x_t that generates predicted yields
        x_t : X_inforionat
        """
        mths = self.mths
        mth_only = self.mth_only

        lam_0, lam_1, delta_1, phi, sigma = self._proc_lam(*lam)
        delta_1, phi, sigma = 

        a_test, b_test = self.gen_pred_coef(lam_0, lam_1, delta_1, phi, sigma)

        pred = px.DataFrame(index=mth_only.index)

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
        #see _proc_lam

        #!!!LEFT off here, look into **kwargs passing

        lat = self.latent
        neqs = self.neqs
        guess_list = []
        lam_0_list = flatten(lam_0)
        lam_1_list = flatten(lam_1)
        if lat >= 1:
        #delta_1_list_t will be estimated using OLS
            delta_1_list_t = flatten(delta_1[:neqs, 0])
            delta_1_list_b = flatten(delta_1[:-lat, 0])
            mu_list = flatten(mu) 
            phi = flatten(phi)
    
    def _gen_OLS_res(self):
        """
        Runs VAR on macro data and retrieves parameters
        """
        #run VAR to generate parameters for known 
        var_data = self.var_data
        freq = self.freq

        var_fit = VAR(var_data, freq=freq).fit(maxlags=maxlags)

        params = var_fit.params.values
        sigma_u = var_fit.sigma_u

        mu = np.zeros([k_ar*neqs+lat, 1])
        mu[:neqs] = params[0, None].T

        phi = np.zeros([k_ar*neqs, k_ar*neqs])
        phi[:neqs] = params[1:].T
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
        delta_1[:neqs] = OLS(



