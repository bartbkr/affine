import numpy as np
import socket
import pickle
import os
import scikits.statsmodels.api as sm
from scikits.statsmodels.tsa.api import VAR
from scikits.statsmodels.tsa.stattools import adfuller
from scikits.statsmodels.base.model import LikelihoodModel
from scikits.statsmodels.sandbox.regression.numdiff import (approx_hess,
                                                        approx_fprime)
from scikits.statsmodels.tsa.filters import hpfilter

import pandas as px
import pandas.core.datetools as dt
from operator import itemgetter
from scipy import optimize
import scipy.linalg as la

import matplotlib.pyplot as plt

import itertools as it

#identify computer
#identify computer
comp = socket.gethostname()
if comp == "BBAKER":
    path_pre = "C:\\Documents and Settings\\bbaker"
if comp == "bart-Inspiron-1525":
    path_pre = "/home/bart"
if comp == "linux-econ6":
    path_pre = "/home/labuser"

########################################
# Get data                             #
########################################

mthdata = px.read_csv(path_pre + "/Documents/Econ_630/data/VARbernankedata.txt",
                      na_values="M", index_col = 0, parse_dates=True)
#pdata = px.read_csv(path_pre + "/Documents/Econ_630/data/prices.txt",
#                      na_values="M", index_col = 0, parse_dates=True)

########################################
# Investigate Prices                   #
########################################

#pdata_new = pdata.diff(periods=12)/pdata
#pdata_new = pdata_new.dropna(axis=0)
#pdata_new['PCE no'] = pdata_new['PCE no']*100
#pdata_new['Food'] = pdata_new['Food']*100
#pdata_new['Energy'] = pdata_new['Energy']*10
#pdata_new.plot()
#plt.legend(loc='best')
#plt.show()

########################################
# Test that HP filter/empl gap correct #
########################################
mthdata['tr_empl_gap'], mthdata['hp_ch'] = hpfilter(mthdata['Total_Nonfarm_employment'], lamb=129600)
mthdata['tr_empl_gap_perc'] = mthdata['tr_empl_gap']/mthdata['hp_ch']

#may have used wrong lambda before
#going to stick with new estimate
#plt.figure()
#mthdata['tr_empl_gap'].plot()
#plt.show()

########################################
# Test that inflation                  #
########################################

mthdata['act_infl'] = mthdata['Pers_Cons_P'].diff(periods=12)/mthdata['Pers_Cons_P']*100

#Looks good

#mthdata.reindex(columns = ['act_infl', 'inflexp_1yr_mean']).plot(subplots=True)
#plt.show()

#Eurodollar rate is clearly upward trending, so lets difference it
mthdata['ed_diff'] = mthdata['one_year_ED']
#mthdata['ed_diff'] = mthdata['one_year_ED'].diff(periods=1)

#############################################
# Run VAR on Bernanke, Sack, Reinhart Model #
#############################################

mod_data = mthdata.reindex(columns=['tr_empl_gap_perc',
                                    'act_infl',
                                    'inflexp_1yr_mean',
                                    'fed_funds',
                                    'ed_diff']).dropna(axis=0)
data = mod_data.values
names = mod_data.columns
dates = mod_data.index

mod_data_ap = mthdata.reindex(columns=['tr_empl_gap_perc',
                                    'act_infl']).dropna(axis=0)

#mod = VAR(data, names=names, dates=dates, freq='monthly')
mod = VAR(mod_data, freq='M')
vreg = mod.fit(maxlags=4)
irsf = vreg.irf(periods=50)
#irsf.plot(orth=True, stderr_type='mc', impulse='tr_empl_gap_perc',repl=1000)
#plt.savefig("../output/VAR_empl_shk.png")
#plt.show()





########################################
# Opt lag tests                        #
########################################

#print 'AIC'
#print VAR(data).fit(maxlags=12, ic='aic').k_ar
#print 'FPE'
#print VAR(data).fit(maxlags=12, ic='fpe').k_ar
#print 'HQIC'
#print VAR(data).fit(maxlags=12, ic='hqic').k_ar
#print 'BIC'
#print VAR(data).fit(maxlags=12, ic='bic').k_ar

#########################################
# Set up BSR affine model               #
#########################################
k_ar = vreg.k_ar

#create BSR X_t
x_t_na = mod_data.copy()
for t in range(k_ar-1):
    for var in mod_data.columns:
        x_t_na[var + '_m' + str(t+1)] = px.Series(mod_data[var].values[:-(t+1)],
                                            index=mod_data.index[t+1:])
#remove missing values
X_t = x_t_na.dropna(axis=0)


#############################################
# Grab yield curve data                     #
#############################################

ycdata = px.read_csv(path_pre + "/Documents/Econ_630/data/yield_curve.csv",
                     na_values = "M", index_col=0, parse_dates=True)

mod_yc_data_nodp = ycdata.reindex(columns=['l_tr_m3', 'l_tr_m6',
                                      'l_tr_y1', 'l_tr_y2',
                                      'l_tr_y3', 'l_tr_y5',
                                      'l_tr_y7', 'l_tr_y10'])
mod_yc_data = mod_yc_data_nodp.dropna(axis=0)
mod_yc_data = mod_yc_data.join(X_t['fed_funds'], how='right')
mod_yc_data = mod_yc_data.rename(columns = {'fed_funds' : 'l_tr_m1'})
mod_yc_data = mod_yc_data.drop(['l_tr_m1'], axis=1)

#drop those obsevations not needed

################################################
# Generate predictions of cumulative shortrate #
################################################

#10 yr = 120 mths
mths_pred = 120
pred = np.zeros((vreg.neqs, len(mod_data)-vreg.k_ar))
k_ar = vreg.k_ar
neqs = vreg.neqs
params = vreg.params.values.copy()
avg_12 = []
for t in range(k_ar, len(mod_data)):
    s_data = mod_data[t-k_ar:t].values
    for i in range(120):
        new_dat = np.zeros((1,neqs))
        for p in range(neqs):
            new_dat[0,p] = params[0,p] + np.dot(np.flipud(s_data)[:k_ar].flatten()[None], params[1:,p,None])[0]
        s_data = np.append(s_data, new_dat, axis=0)
    avg_12.append(np.mean(s_data[:,3]))

#implied term premium
var_tp = px.DataFrame(index=mod_yc_data.index[1:], data=np.asarray(avg_12), columns=["pred_12"])
var_tp['act_12'] = mod_yc_data['l_tr_y10']
var_tp['VAR term premium'] = var_tp['act_12'] - var_tp['pred_12']

#############################################
# Create affine class system                   #
#############################################


class BSR(LikelihoodModel):
    def __init__(self, yc_data, var_data, rf_rate=None, maxlags=4,
                 freq='M', latent=0, no_err=None):
        """
        Attemps to solve BSR model
        """
        self.yc_data = yc_data

        #gen VAR instance
        data = var_data.values
        names = var_data.columns
        dates = var_data.index

        #mod = VAR(data, names=names, dates=dates, freq='monthly')
        mod = VAR(var_data, freq=freq)
        vreg = mod.fit(maxlags=maxlags)

        #generates mths and mth_only
        self._proc_to_mth()

        #number of latent variables to include
        lat = self.latent = latent
        self.no_err = no_err

        self.k_ar = k_ar = vreg.k_ar
        self.neqs = neqs = vreg.neqs
        self.params = params = vreg.params.values
        sigma_u = vreg.sigma_u

        if lat:
            assert len(no_err) >= lat, "One yield estimated with no err"\
                                        + "for each latent variable"

            len_lst = [neqs+lat, (neqs + lat)**2, lat, lat**2, lat**2]
            pos_lst = []
            acc = 0
            for x in len_lst:
                pos_lst.append(x+acc)
                acc += x
            self.pos_lst = pos_lst

        mu = np.zeros([k_ar*neqs+lat,1])
        mu[:neqs] = params[0,None].T
        self.mu = mu

        phi = np.zeros([k_ar*neqs, k_ar*neqs])
        phi[:neqs] = params[1:].T
        phi[neqs:,:(k_ar-1)*neqs] = np.identity((k_ar-1)*neqs)
        self.phi = phi

        sig = np.zeros([k_ar*neqs, k_ar*neqs])
        sig[:neqs, :neqs] = sigma_u
        self.sig = sig

        if lat == 0:
            self.delta_0 = 0
            delta_1 = np.zeros([neqs*k_ar,1])
            #delta_1 is vector of zeros, with one grabbing fed_funds rate
            delta_1[np.argmax(mod_data.columns == 'fed_funds')] = 1
            self.delta_1 = delta_1

        else:
            #this is the method outlined by Ang and Piazzesi (2003)
            reg_data = var_data.copy()
            reg_data['intercept'] = 1
            par = sm.OLS(rf_rate, reg_data).fit().params
            self.delta_0 = par.values[-1]
            delta_1 = np.zeros([neqs*k_ar+lat,1])
            delta_1[:neqs,0] = par.values[:neqs]
            self.delta_1 = delta_1

        #get VAR input data ready
        x_t_na = var_data.copy()
        for t in range(k_ar-1):
            for var in var_data.columns:
                x_t_na[var + '_m' + str(t+1)] = px.Series(var_data[var].
                        values[:-(t+1)], index=var_data.index[t+1:])
        self.var_data = x_t_na.dropna(axis=0)

        super(BSR, self).__init__(var_data)


    def solve(self, lam_0_g, lam_1_g, delt_1_g=None, phi_g=None,
            sig_g=None, maxfev=10000, xtol=1e-100, 
              full_output=False):
        lat = self.latent
        neqs = self.neqs
        k_ar = self.k_ar

        assert len(lam_0_g) == neqs + lat, "Length of lam_0_g not correct"
        assert len(lam_1_g) == (neqs + lat)**2, "Length of lam_1_g not correct"
        if lat:
            assert len(delt_1_g) == lat, "Length of delt_1_g not correct"
            assert len(phi_g) == lat**2, "Length of phi_g not correct"
            assert len(sig_g) == lat**2, "Length of sig_g not correct"
            lam = np.asarray(lam_0_g + lam_1_g + delt_1_g + phi_g + sig_g)
        else:
            lam = np.asarray(lam_0_g + lam_1_g)

        func = self._BSR_nsum_errs
        reslt = optimize.leastsq(func, lam, maxfev=maxfev,
                                xtol=xtol, full_output=full_output)
        lam_solv = reslt[0]
        output = reslt[1:]

        lam_0, lam_1, delta_1, phi, sig = self._proc_lam(lam_solv)

        a, b = self.gen_pred_coef(lam_0, lam_1, delta_1, phi, sig)

        if full_output:
            return lam_0, lam_1, delta_1, phi, sig, a, b, output
        else:
            return lam_0, lam_1, delta_1, phi, sig, a, b

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
        loglike = self._BSR_nsum_errs
        return approx_fprime(lam, loglike, epsilon=1e-8)

    def hessian(self, lam):
        """
        Returns numerical hessian.
        """
        loglike = self._BSR_nsum_errs
        return approx_hess(lam, loglike)[0]

    def gen_pred_coef(self, lam_0_ab, lam_1_ab, delta_1, phi, sig):
        lat = self.latent
        mths = self.mths
        delta_0 = self.delta_0
        mu = self.mu
        max_mth = max(mths)
        #generate predictions
        A = np.zeros((max_mth, 1))
        A[0] = -delta_0
        B = []
        B.append(-delta_1)
        for x in range(max_mth-1):
            A[x+1] = (A[x] + np.dot(B[x].T, (mu - np.dot(sig,  \
            lam_0_ab))) + (1.0/2)*np.dot(np.dot(np.dot(B[x].T, \
            sig), sig.T), B[x]) - delta_0)[0][0]
            B.append(np.dot((phi - np.dot(sig, lam_1_ab)).T, B[x]) - delta_1)
        n_inv = 1.0/np.add(range(max_mth), 1).reshape((max_mth,1))
        a = -(A*n_inv)
        b = np.zeros_like(B)
        for x in range(max_mth):
            b[x] = np.multiply(-B[x], n_inv[x])
        return a, b

    def _BSR_nsum_errs(self, lam):
        """
        This function generates the sum of the prediction errors
        """
        lat = self.latent
        no_err = self.no_err
        neqs = self.neqs
        k_ar = self.k_ar
        mths = self.mths
        mth_only = self.mth_only
        X_t = self.var_data

        lam_0, lam_1, delta_1, phi, sig = self._proc_lam(lam)

        a, b = self.gen_pred_coef(lam_0, lam_1, delta_1, phi, sig)

        #this is explosive

        if lat:
            X_t = self._solve_X_t_unkn(a, b, X_t)

        errs = []

        for i in mths:
            act = np.flipud(mth_only['l_tr_m' + str(i)].values)
            pred = a[i-1] + np.dot(b[i-1].T, np.fliplr(X_t.T))[0]
            errs = errs + (act-pred).tolist()
        return errs

    def _solve_X_t_unkn(self, a, b, X_t):
        lat = self.latent
        no_err = self.no_err
        mth_only = self.mth_only
        X_t_new = np.append(X_t, np.zeros((X_t.shape[0],lat)), axis=1)
        for t in range(X_t.shape[0]):
            #mx = c
            m = np.zeros((lat,lat))
            c = np.zeros(lat)
            for x, i in enumerate(no_err):
                act = mth_only['l_tr_m' + str(i)].values[t]
                #print "i"
                #print i
                #print "act"
                #print act
                #print "a[i-1]"
                #print a[i-1]
                #print "b[i-1].T[:,:-lat]"
                #print b[i-1].T[:,:-lat]
                #print "X_t.values.T[:,t,None]"
                #print X_t.values.T[:,t,None]
                c[x] = act - a[i-1] - np.dot(b[i-1].T[:,:-lat],\
                        X_t.values.T[:,t,None])
                m[x] = b[i-1].T[:,-lat:]
            X_t_new[t,-lat:] = la.solve(m,c)
        return X_t_new

    def _proc_to_mth(self):
        frame = self.yc_data
        mths = []
        fnd = 0
        n_cols = len(frame.columns)
        for x in frame.columns:
            srt_ord = []
            if 'm' in x:
                mths.append(int(x[6]))
                if fnd == 0:
                    mth_only = px.DataFrame(frame[x],
                            columns = [x],
                            index=frame.index)
                    fnd = 1
                else:
                    mth_only[x] = frame[x]
            elif 'y' in x:
                mth = int(x[6:])*12
                mths.append(mth)
                mth_only[('l_tr_m' + str(mth))] = frame[x]
        col_dict = dict([( mth_only.columns[x], mths[x]) for x in
                    range(n_cols)])
        cols = np.asarray(sorted(col_dict.iteritems(),
                        key=itemgetter(1)))[:,0].tolist()
        mth_only = mth_only.reindex(columns = cols)
        mths.sort()
        self.mths = mths
        self.mth_only = mth_only

    def _proc_lam(self, lam):
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

            lam_0 = np.zeros([k_ar*neqs+lat,1])
            lam_0[:neqs,0] = np.asarray(lam_0_est[:neqs]).T
            lam_0[-lat:,0] = np.asarray(lam_0_est[-lat:]).T

            lam_1 = np.zeros([k_ar*neqs+lat, k_ar*neqs+lat])
            lam_1[:neqs,:neqs] = np.reshape(lam_1_est[:neqs**2], (neqs,neqs))
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
            delta_1[-lat:,0] = np.asarray(delt_1_g)

            #add rows/columns for unk params
            phi_n = self.phi.copy()
            add = np.zeros([lat, np.shape(phi_n)[1]])
            phi_n = np.append(phi_n, add, axis=0)
            add = np.zeros([np.shape(phi_n)[0], lat])
            phi = np.append(phi_n, add, axis=1)
            #fill in parm guesses
            phi[-lat:, -lat:] = np.reshape(phi_g, (lat,lat))

            #add rows/columns for unk params
            sig_n = self.sig.copy()
            add = np.zeros([lat, np.shape(sig_n)[1]])
            sig_n = np.append(sig_n, add, axis=0)
            add = np.zeros([np.shape(sig_n)[0], lat])
            sig = np.append(sig_n, add, axis=1)
            sig[-lat:, -lat:] = np.reshape(sig_g, (lat,lat))

        else:
            lam_0_est = lam[:neqs]
            lam_1_est = lam[neqs:]

            lam_0 = np.zeros([k_ar*neqs,1])
            lam_0[:neqs] = np.asarray([lam_0_est]).T

            lam_1 = np.zeros([k_ar*neqs, k_ar*neqs])
            lam_1[:neqs,:neqs] = np.reshape(lam_1_est, (neqs,neqs))

            delta_1 = self.delta_1
            phi = self.phi
            sig = self.sig

        return lam_0, lam_1, delta_1, phi, sig


#############################################
# Testing                                   #
#############################################

#anl_mths, mth_only_data = proc_to_mth(mod_yc_data)
bsr = BSR(yc_data = mod_yc_data, var_data = mod_data)
neqs = bsr.neqs
k_ar = bsr.k_ar

#test sum_sqr_pe
lam_0_t = [0.03,0.1,0.2,-0.21,0.32]
lam_0_nr = np.zeros([5*4, 1])

lam_1_t = []
lam_1_nr = np.zeros([5*4, 5*4])
for x in range(neqs):
    lam_1_t = lam_1_t + (np.asarray([[0.03,0.1,0.2,0.21,0.32]]) \
                            *np.random.random())[0].tolist()
    #lam_2_t[x, :neqs] = np.asarray([[2.5e-90,1e-87,9.5e-75,
    #                                1.21e-93,-0.5e-88]])

#rerun
a_nrsk, b_nrsk = bsr.gen_pred_coef(lam_0_nr, lam_1_nr, bsr.delta_1,
                bsr.phi, bsr.sig)
#out_bsr = bsr.solve(lam_0_t, lam_1_t, xtol=1e-140, maxfev=1000000,
                #full_output=True)

#init pkl
#pkl_file = open("out_bsr1.pkl", 'wb')

#save rerun
#pickle.dump(out_bsr, pkl_file)

#load instead of rerun
pkl_file = open("out_bsr1.pkl", 'rb')
out_bsr_ld = pickle.load(pkl_file)

lam_0_n, lam_1_n, delta_1_n, phi_n, sig_n, a, b, output_n = out_bsr_ld

#gen BSR predicted
X_t = bsr.var_data
per = bsr.mth_only.index
act_pred = px.DataFrame(index=per)
for i in bsr.mths:
    act_pred[str(i) + '_mth_act'] = bsr.mth_only['l_tr_m' + str(i)]
    act_pred[str(i) + '_mth_pred'] = a[i-1] + \
                                    np.dot(b[i-1].T, X_t.values.T)[0]
    act_pred[str(i) + '_mth_nrsk'] = a_nrsk[i-1] + \
                                    np.dot(b_nrsk[i-1].T, X_t.values.T)[0]
#plot act 10-year plot
#thirty_yr = act_pred.reindex(columns = filter(lambda x: '360' in x, act_pred))
ten_yr = act_pred.reindex(columns = filter(lambda x: '120' in x, act_pred))
seven_yr = act_pred.reindex(columns = filter(lambda x: '84' in x, act_pred))
five_yr = act_pred.reindex(columns = filter(lambda x: '60' in x,act_pred))
three_yr = act_pred.reindex(columns = filter(lambda x: '36' in x,act_pred))
two_yr = act_pred.reindex(columns = filter(lambda x: '24' in x,act_pred))
one_yr = act_pred.reindex(columns = ['12_mth_act',
                                     '12_mth_pred',
                                     '12_mth_nrsk'])
six_mth = act_pred.reindex(columns = ['6_mth_act',
                                      '6_mth_pred',
                                      '6_mth_nrsk'])

#plot the term premium
ten_yr['rsk_prem'] = ten_yr['120_mth_pred'] - ten_yr['120_mth_nrsk']
var_tp['BSR term premium'] = ten_yr['rsk_prem']
#ten_yr['rsk_prem'].plot()

#plot the slope of the yield curve
ten_yr_diff_act = (mod_yc_data['l_tr_y10'] - X_t['fed_funds']).dropna()
var_tp['Slope term premium'] = ten_yr_diff_act
#ten_yr_diff_act.plot()
#plt.show()

#plot 10_mth - 3_mth
t_struct_move = px.DataFrame(index=mod_yc_data_nodp.index)
t_struct_move['10 Yr Yield - 3 Mth Yield'] = mod_yc_data_nodp['l_tr_y10'] - mod_yc_data_nodp['l_tr_m3']
t_struct_move = t_struct_move.dropna(axis=0)
t_struct_move.plot()
rec = px.DataFrame(index = t_struct_move.index)
#identify recession periods
rec['periods'] = 0
rec['periods'][:11] = 1
rec['periods'][101:111] = 1
rec['periods'][230:239] = 1
rec['periods'][311:330] = 1
#plt.fill_between(rec.index, rec.min().values, rec.max().values, where=rec['periods']==1, facecolor='grey', alpha=0.5)
plt.fill_between(rec.index, -1, 5, where=rec['periods']==1, facecolor='grey', alpha=0.5)
plt.axis('tight')
plt.savefig("../output/slope_ts.png")
#plt.show()
#plt.close()

key_rate = px.DataFrame(index=mod_yc_data_nodp.index)
key_rate['10 Yr Yield'] = mod_yc_data_nodp['l_tr_y10']
key_rate['3 Mth Yield'] = mod_yc_data_nodp['l_tr_m3']
key_rate['Fed Funds Rate'] = mthdata['fed_funds']
key_rate = key_rate.dropna(axis=0)
key_rate.plot()
plt.fill_between(rec.index, 0, 15, where=rec['periods']==1, facecolor='grey', alpha=0.5)
plt.axis('tight')
plt.savefig("../output/key_rates.png")
#plt.show()
#plt.close()

var_tp_fin = var_tp.reindex(columns=['Slope term premium', 'VAR term premium', 'BSR term premium'])
#var_tp_fin.plot()
#plt.fill_between(rec.index[60:], -1.5, 12, where=rec['periods'][60:]==1, facecolor='grey', alpha=0.5)
#plt.axis('tight')
#plt.savefig("../output/tp_fin.png")

#new VAR system with slope
new_VAR_sys1 = mod_data.join(var_tp_fin['Slope term premium']).dropna()
new_VAR_sys1 = new_VAR_sys1.drop(['ed_diff'], axis=1)
mod_sys1 = VAR(new_VAR_sys1, freq='M')
vreg_sys1 = mod_sys1.fit(maxlags=3)
irf_sys1 = vreg_sys1.irf(periods=50)
irf_sys1.plot(orth=True, impulse="Slope term premium")
data=new_VAR_sys1 

print "Slope tp system"
print 'AIC'
print VAR(data, freq='M').fit(maxlags=12, ic='aic').k_ar
print 'FPE'
print VAR(data, freq='M').fit(maxlags=12, ic='fpe').k_ar
print 'HQIC'
print VAR(data, freq='M').fit(maxlags=12, ic='hqic').k_ar
print 'BIC'
print VAR(data, freq='M').fit(maxlags=12, ic='bic').k_ar
print "Slope caus test tp -> empl"
print vreg_sys1.test_causality("tr_empl_gap_perc", "Slope term premium", verbose=False, signif=0.10)
print "Slope caus test empl -> tp"
print vreg_sys1.test_causality( "Slope term premium", "tr_empl_gap_perc", verbose=False, signif=0.10)

#new VAR system with VAR predicted tp
new_VAR_sys2 = mod_data.join(var_tp_fin['VAR term premium']).dropna()
new_VAR_sys2 = new_VAR_sys2.drop(['ed_diff'], axis=1)
mod_sys2 = VAR(new_VAR_sys2, freq='M')
vreg_sys2 = mod_sys2.fit(maxlags=3)
irf_sys2 = vreg_sys2.irf(periods=50)
irf_sys2.plot(orth=True, impulse="VAR term premium")
data=new_VAR_sys2 

print "VAR tp system"
print 'AIC'
print VAR(data, freq='M').fit(maxlags=12, ic='aic').k_ar
print 'FPE'
print VAR(data, freq='M').fit(maxlags=12, ic='fpe').k_ar
print 'HQIC'
print VAR(data, freq='M').fit(maxlags=12, ic='hqic').k_ar
print 'BIC'
print VAR(data, freq='M').fit(maxlags=12, ic='bic').k_ar

print "VAR caus test tp -> empl"
print vreg_sys2.test_causality("tr_empl_gap_perc", "VAR term premium", verbose=False, signif=0.10)
print "VAR caus test empl -> tp"
print vreg_sys2.test_causality( "VAR term premium", "tr_empl_gap_perc", verbose=False, signif=0.10)

#new VAR system with BSR predicted tp
new_VAR_sys3 = mod_data.join(var_tp_fin['BSR term premium']).dropna()
new_VAR_sys3 = new_VAR_sys3.drop(['ed_diff'], axis=1)
mod_sys3 = VAR(new_VAR_sys3, freq='M')
vreg_sys3 = mod_sys3.fit(method='ols', maxlags=3)
irf_sys3 = vreg_sys3.irf(periods=50)
irf_sys3.plot(orth=True, impulse="BSR term premium")
data=new_VAR_sys3 

print "BSR tp system"
print 'AIC'
print VAR(data, freq='M').fit(method='ols', maxlags=12, ic='aic').k_ar
print 'FPE'
print VAR(data, freq='M').fit(method='ols', maxlags=12, ic='fpe').k_ar
print 'HQIC'
print VAR(data, freq='M').fit(method='ols', maxlags=12, ic='hqic').k_ar
print 'BIC'
print VAR(data, freq='M').fit(method='ols', maxlags=12, ic='bic').k_ar

print "BSR caus test tp -> empl"
print vreg_sys3.test_causality("tr_empl_gap_perc", "BSR term premium", verbose=False, signif=0.10)
print "BSR caus test empl -> tp"
print vreg_sys3.test_causality( "BSR term premium", "tr_empl_gap_perc", verbose=False, signif=0.10)
