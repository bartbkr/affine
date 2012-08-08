import numpy as np
import socket
import pickle
import os
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.base.model import LikelihoodModel
from statsmodels.tools.numdiff import (approx_hess, approx_fprime)
from statsmodels.tsa.filters import hpfilter

import pandas as px
import pandas.core.datetools as dt
from operator import itemgetter
from scipy import optimize
import scipy.linalg as la

#for sending email
import smtplib
import string

import matplotlib.pyplot as plt

import itertools as it

from affine import affine
#send email when done
import datetime as dt
d1 = dt.datetime.now()
import getpass

#attempt multi-threading
import time
import multiprocessing

#identify computer
#identify computer
comp = socket.gethostname()
global path_pre
if comp == "BBAKER":
    path_pre = "C:\\Documents and Settings\\bbaker"
if comp == "bart-Inspiron-1525":
    path_pre = "/home/bart"
if comp == "linux-econ6":
    path_pre = "/home/labuser"

passwd = getpass.getpass(prompt="Please enter email passwd: ")

##############################################################################
# Estimate model with Eurodollar futures
##############################################################################

print "Model 1 running"

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
mthdata['ed_fut'] = 100 - mthdata['one_year_ED']
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
# Set up affine affine model               #
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

def robust(mod_data, mod_yc_data, lam_0_g=None, lam_1_g=None):
    """
    Function to run model with guesses, also generating 
    mod_data : pandas DataFrame 
        model data
    mod_yc_data : pandas DataFrame
        model yield curve data
    lam_0_g : array
        Guess for lambda 0
    lam_1_g : array
        Guess for lambda 1
    """
        
    # subset to pre 2005
    mod_data = mod_data[:217]
    mod_yc_data = mod_yc_data[:214]

    #anl_mths, mth_only_data = proc_to_mth(mod_yc_data)
    bsr = affine(yc_data = mod_yc_data, var_data = mod_data)
    neqs = bsr.neqs
    k_ar = bsr.k_ar

    #test sum_sqr_pe
    if lam_0_g is None:
        lam_0_g = np.zeros([5*4, 1])
        lam_0_g[:neqs] = np.array([[0.03],[0.1],[0.2],[-0.21],[0.32]])

    #set seed for future repl

    if lam_1_g is None:
        lam_1_g = np.zeros([5*4, 5*4])
        for x in range(neqs):
            guess = [0.03,0.1,0.2,0.21,0.32]
            lam_1_g[x, :neqs] = np.array([guess])*np.random.random()

    #generate a and b for no risk 
    #a_nrsk, b_nrsk = bsr.gen_pred_coef(lam_0_nr, lam_1_nr, bsr.delta_1,
                    #bsr.phi, bsr.sig)

    out_bsr = bsr.solve(lam_0_g, lam_1_g, ftol=1e-950, xtol=1e-950,
                        maxfev=1000000000, full_output=True)

    lam_0_n, lam_1_n, delta_1_n, phi_n, sig_n, a, b, output_n = out_bsr
    return lam_0_n, lam_1_n

big_runs = 10
run_groups = []
atts = 10
np.random.seed(101)
collect = {}

#generate decent guesses
for run in range(big_runs):
    lam_0_coll = np.zeros((atts, neqs*k_ar, 1))
    lam_1_coll = np.zeros((atts, neqs*k_ar, neqs*k_ar))
    for i in range(atts):
        print (run, i)
        sin_run = robust(mod_data=mod_data, mod_yc_data=mod_yc_data)
        lam_0_coll[i] = sin_run[0]
        lam_1_coll[i] = sin_run[1]
    lam_0_mn = np.mean(lam_0_coll, axis=0)
    lam_1_mn = np.mean(lam_1_coll, axis=0)
    #add mean of these runs to run_groups
    run_groups.append((lam_0_mn, lam_1_mn))
    collect[(run,i)] = [(lam_0_coll, lam_1_coll)]

pkl_file = open("collect.pkl", "wb")
pickle.dump(collect, pkl_file)
pkl_file.close()

#now use these means as guesses for the next 10 runs
res = []
for guess in range(big_runs):
    res.append(robust(mod_data=mod_data, mod_yc_data=mod_yc_data,
        lam_0_g=run_groups[guess][0], lam_1_g=run_groups[guess][1]))

#should probably pickle the results here

pkl_file = open("out_big_run.pkl", "wb")
pickle.dump(res, pkl_file)
pkl_file.close()

# Initialize SMTP server

server=smtplib.SMTP('smtp.gmail.com:587')
server.starttls()
server.login("bartbkr",passwd)

# Send email
senddate=dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d')
subject="Your job has completed"
m="Date: %s\r\nFrom: %s\r\nTo: %s\r\nSubject: %s\r\nX-Mailer: My-Mail\r\n\r\n"\
% (senddate, "bartbkr@gmail.com", "barbkr@gmail.com", subject)
msg='''
Job has completed '''

server.sendmail("bartbkr@gmail.com", "bartbkr@gmail.com", m+msg)
server.quit()
