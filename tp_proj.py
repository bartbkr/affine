import numpy as np
import socket
import pickle
import os
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.base.model import LikelihoodModel
from statsmodels.sandbox.regression.numdiff import (approx_hess,
                                                        approx_fprime)
from statsmodels.tsa.filters import hpfilter

import pandas as px
import pandas.core.datetools as dt
from operator import itemgetter
from scipy import optimize
import scipy.linalg as la

import matplotlib.pyplot as plt

import itertools as it

from affine import BSR

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

mthdata = px.read_csv(path_pre + "/Documents/Econ_630/data/VARbernankedata.csv",
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
mthdata['ed_fut'] = np.log(mthdata['one_year_ED'])
#mthdata['ed_fut'] = mthdata['one_year_ED'].diff(periods=1)

#############################################
# Run VAR on Bernanke, Sack, Reinhart Model #
#############################################

mod_data = mthdata.reindex(columns=['tr_empl_gap_perc',
                                    'act_infl',
                                    'inflexp_1yr_mean',
                                    'fed_funds',
                                    'ed_fut']).dropna(axis=0)
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
# Testing                                   #
#############################################

# subset to pre 2005
mod_data = mod_data[:217]
mod_yc_data = mod_yc_data[:214]

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

#let's try running it on a shorter time series closer to BSR 
#original data set

out_bsr = bsr.solve(lam_0_t, lam_1_t, xtol=1e-140, maxfev=1000000,
                full_output=True)

#init pkl
pkl_file = open("out_bsr1.pkl", 'wb')

#save rerun
pickle.dump(out_bsr, pkl_file)

#load instead of rerun
#pkl_file = open("out_bsr1.pkl", 'rb')
#out_bsr_ld = pickle.load(pkl_file)

#lam_0_n, lam_1_n, delta_1_n, phi_n, sig_n, a, b, output_n = out_bsr_ld
lam_0_n, lam_1_n, delta_1_n, phi_n, sig_n, a, b, output_n = out_bsr

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
#ten_yr['rsk_prem'] = ten_yr['120_mth_pred'] - ten_yr['120_mth_nrsk']
#var_tp['BSR term premium'] = ten_yr['rsk_prem']
#ten_yr['rsk_prem'].plot()
#plt.show()

