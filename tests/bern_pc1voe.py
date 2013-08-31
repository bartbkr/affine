"""
This script attempts to replicate the Bernanke, Sack, and Reinhart (2004) model
"""
import numpy as np
import numpy.ma as ma
import pandas as pa
import datetime as dt
import matplotlib.pyplot as plt

import atexit
import keyring
import sys

from statsmodels.tsa.api import VAR
from statsmodels.tsa.filters import hpfilter
from statsmodels.sandbox.pca import Pca
from affine.constructors.helper import (to_mth, bsr_constructor, pass_ols)
from affine.model.affine import Affine

import ipdb

start_date = dt.datetime.now()
passwd = keyring.get_password("email_auth", "bartbkr")

########################################
# Get macro data                       #
########################################
mthdata = pa.read_csv("./data/macro_data.csv", na_values="M",
                        index_col = 0, parse_dates=True, sep=";")
nonfarm = mthdata['Total_Nonfarm_employment_seas'].dropna()

tr_empl_gap, hp_ch = hpfilter(nonfarm, lamb=129600)

mthdata['tr_empl_gap'] = pa.Series(tr_empl_gap, index=nonfarm.index)
mthdata['hp_ch'] = pa.Series(hp_ch, index=nonfarm.index)
mthdata['tr_empl_gap_perc'] = mthdata['tr_empl_gap']/mthdata['hp_ch'] * 100
mthdata['act_infl'] = \
    mthdata['PCE_seas'].diff(periods=12)/mthdata['PCE_seas']*100
mthdata['ed_fut'] = 100 - mthdata['ed4_end_mth']
mthdata['uncert'] = mthdata['gnp_gdp_top10'] - mthdata['gnp_gdp_bot10']

#normalize uncert and djialog_std to zero mean and unit variance
for var in ['uncert', 'voe_cboe']:
    mthdata[var + "_norm"] = (mthdata[var] - mthdata[var].mean()) / \
                                  mthdata[var].std()
norm_vals = mthdata.filter(regex='.*_norm').dropna()

#PC, but performed over entire period
norm_pca = Pca(norm_vals.values.T, names=norm_vals.columns.tolist())
norm_pca_data = norm_pca.project(nPCs=1)
mthdata['uncertPC1'] = pa.Series(norm_pca_data.T.tolist()[0],
                                 index=norm_vals.index)

#define final data set
mod_data = mthdata.reindex(columns=['tr_empl_gap_perc',
                                   'act_infl',
                                   'gnp_gdp_deflat_nxtyr',
                                    'fed_funds',
                                    'uncertPC1']).dropna(axis=0)

neqs = 5
k_ar = 4

#########################
# Grab yield curve data #
#########################

#create BSR x_t
x_t_na = mod_data.copy()
for t in range(k_ar-1):
    for var in mod_data.columns:
        x_t_na[var + '_m' + str(t+1)] = pa.Series(mod_data[var].values[:-(t+1)],
                                            index=mod_data.index[t+1:])
#remove missing values
x_t = x_t_na.dropna(axis=0)

ycdata = pa.read_csv("./data/yield_curve.csv", na_values = "M", index_col=0,
                     parse_dates=True, sep=";")

mod_yc_data_nodp = ycdata.reindex(columns=['trcr_m3', 'trcr_m6',
                                      'trcr_y1', 'trcr_y2',
                                      'trcr_y3', 'trcr_y5',
                                      'trcr_y7', 'trcr_y10'])
mod_yc_data = mod_yc_data_nodp.dropna(axis=0)
mod_yc_data = mod_yc_data.join(x_t['fed_funds'], how='right')
mod_yc_data.insert(0, 'trcr_m1', mod_yc_data['fed_funds'])
mod_yc_data = mod_yc_data.drop(['fed_funds'], axis=1)

mod_yc_data = to_mth(mod_yc_data)

mths = [3, 6, 12, 24, 36, 60, 84, 120]
del mod_yc_data['trcr_m1']

# Setup model
meth = "ls"
run_groups = []
atts = 25
np.random.seed(101)
collect_0 = []
collect_1 = []

#subset to range specified in BSR

var_dates = pa.date_range("5/1/1990", "5/1/2012", freq="MS").to_pydatetime()
yc_dates = pa.date_range("8/1/1990", "5/1/2012", freq="MS").to_pydatetime()

mod_data = mod_data.ix[var_dates]
mod_yc_data = mod_yc_data.ix[yc_dates]

##################################################
# Define exit message to send to email upon fail #
##################################################
#atexit.register(fail_mail, start_date, passwd)

#generate decent guesses
lam_0_e, lam_1_e, delta_0_e, delta_1_e, mu_e, phi_e, sigma_e \
    = bsr_constructor(k_ar=k_ar, neqs=neqs)

delta_0_e, delta_1_e, mu_e, phi_e, sigma_e = pass_ols(var_data=mod_data,
                                                      freq="M", lat=0,
                                                      k_ar=k_ar, neqs=neqs,
                                                      delta_0=delta_0_e,
                                                      delta_1=delta_1_e,
                                                      mu=mu_e, phi=phi_e,
                                                      sigma=sigma_e)
delta_1_e[np.argmax(mod_data.columns == 'fed_funds')] = 1

print "Initial estimation"
bsr_model = Affine(yc_data=mod_yc_data, var_data=mod_data, lam_0_e=lam_0_e,
                   lam_1_e=lam_1_e, delta_0_e=delta_0_e, delta_1_e=delta_1_e,
                   mu_e=mu_e, phi_e=phi_e, sigma_e=sigma_e, mths=mths)


guess_length = bsr_model.guess_length
guess_params = [0.0000] * guess_length

xtols = [0.1,
         0.05,
         0.03,
         0.01,
         0.009,
         0.005,
         0.001,
         0.0001
        ]

ftols = [1.49012e-8]

for xtol in xtols:
    for ftol in ftols:
        print "xtol " + str(xtol)
        print "ftol " + str(ftol)
        print "Begin " + str(yc_dates[0])
        print "End " + str(yc_dates[-1])
        print "variables " + str(list(bsr_model.names))
        out_bsr = bsr_model.solve(guess_params=guess_params, method='nls',
                                ftol=ftol, xtol=xtol, maxfev=10000000,
                                full_output=False)

        lam_0, lam_1, delta_0, delta_1, mu, phi, sigma, a_solve, \
                        b_solve, solv_cov = out_bsr

        a_rsk, b_rsk = bsr_model.gen_pred_coef(lam_0=lam_0, lam_1=lam_1,
                                            delta_0=delta_0, delta_1=delta_1,
                                            mu=mu, phi=phi, sigma=sigma)

        #generate no risk results
        lam_0_nr = np.zeros([neqs*k_ar, 1])
        lam_1_nr = np.zeros([neqs*k_ar, neqs*k_ar])
        sigma_zeros = np.zeros_like(sigma)
        a_nrsk, b_nrsk = bsr_model.gen_pred_coef(lam_0=lam_0_nr,
                                                 lam_1=lam_1_nr,
                                                 delta_0=delta_0,
                                                 delta_1=delta_1, mu=mu,
                                                 phi=phi, sigma=sigma_zeros)
        #gen BSR predicted
        X_t = bsr_model.var_data_vert
        per = bsr_model.yc_data.index
        act_pred = pa.DataFrame(index=per)
        for i in mths:
            act_pred[str(i) + '_mth_act'] = bsr_model.yc_data['trcr_m'
                                                              + str(i)]
            act_pred[str(i) + '_mth_pred'] = a_rsk[i-1] + \
                                            np.dot(b_rsk[i-1], X_t.values.T)
            act_pred[str(i) + '_mth_nrsk'] = a_nrsk[i-1] + \
                                            np.dot(b_nrsk[i-1].T, X_t.values.T)
            act_pred[str(i) + '_mth_err'] = np.abs(act_pred[str(i)
                                                            + '_mth_act']
                                                   - act_pred[str(i)
                                                              + '_mth_pred'])
        ten_yr = act_pred.reindex(columns = filter(lambda x: '120' in x,
                                                   act_pred))
        seven_yr = act_pred.reindex(columns = filter(lambda x: '84' in x,
                                                    act_pred))
        five_yr = act_pred.reindex(columns = filter(lambda x: '60' in
                                                    x,act_pred))
        three_yr = act_pred.reindex(columns = filter(lambda x: '36' in
                                                     x,act_pred))
        two_yr = act_pred.reindex(columns = filter(lambda x: '24' in
                                                   x,act_pred))
        one_yr = act_pred.reindex(columns = ['12_mth_act', '12_mth_pred',
                                            '12_mth_nrsk', '12_mth_err'])
        six_mth = act_pred.reindex(columns = ['6_mth_act', '6_mth_pred',
                                            '6_mth_nrsk', '6_mth_err'])

        #generate st dev of residuals
        yields = ['six_mth', 'one_yr', 'two_yr', 'three_yr', 'five_yr',
                  'seven_yr', 'ten_yr']
        for yld in yields:
            print yld + " & " + str(np.std(eval(yld).filter(
                                                regex= '.*err$').values,
                                                ddof=1)*100)
"""
