"""
This script attempts to replicate the Bernanke, Sack, and Reinhart (2004) model
"""
import numpy as np
import pandas as px
import datetime as dt

import atexit
import keyring
import sys

from statsmodels.tsa.api import VAR
from statsmodels.tsa.filters import hpfilter
from util import pickle_file, success_mail, fail_mail, to_mth, gen_guesses, \
                    robust

import pdb

start_date = dt.datetime.now()
passwd = keyring.get_password("email_auth", "bartbkr")

########################################
# Get macro data                       #
########################################
mthdata = px.read_csv("../data/macro_data.csv", na_values="M",
                        index_col = 0, parse_dates=True, sep=";")
nonfarm = mthdata['Total_Nonfarm_employment_seas'].dropna()

tr_empl_gap, hp_ch = hpfilter(nonfarm, lamb=129600)

mthdata['tr_empl_gap'] = px.Series(tr_empl_gap, index=nonfarm.index)
mthdata['hp_ch'] = px.Series(hp_ch, index=nonfarm.index)
mthdata['tr_empl_gap_perc'] = mthdata['tr_empl_gap']/mthdata['hp_ch'] * 100
mthdata['act_infl'] = \
    mthdata['PCE_seas'].diff(periods=12)/mthdata['PCE_seas']*100
mthdata['ed_fut'] = 100 - mthdata['ed4_end_mth']

#define final data set
mod_data = mthdata.reindex(columns=['tr_empl_gap_perc',
                                   'act_infl',
                                   'inflexp_1yr_mean',
                                    'fed_funds',
                                    'ed_fut']).dropna(axis=0)

neqs = 5
k_ar = 4

#########################
# Grab yield curve data #
#########################

#create BSR x_t
x_t_na = mod_data.copy()
for t in range(k_ar-1):
    for var in mod_data.columns:
        x_t_na[var + '_m' + str(t+1)] = px.Series(mod_data[var].values[:-(t+1)],
                                            index=mod_data.index[t+1:])
#remove missing values
x_t = x_t_na.dropna(axis=0)

ycdata = px.read_csv("../data/yield_curve.csv", na_values = "M", index_col=0,
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

# Setup model
meth = "ls"
run_groups = []
atts = 25
np.random.seed(101)
collect_0 = []
collect_1 = []

#subset to range specified in BSR

var_dates = px.date_range("3/1/1982", "12/1/2004", freq="MS").to_pydatetime()
yc_dates = px.date_range("6/1/1982", "12/1/2004", freq="MS").to_pydatetime()

mod_data = mod_data.ix[var_dates]
mod_yc_data = mod_yc_data.ix[yc_dates]

##################################################
# Define exit message to send to email upon fail #
##################################################
#atexit.register(fail_mail, start_date, passwd)

#generate decent guesses
lam_0_coll = np.zeros((atts, neqs*k_ar, 1))
lam_1_coll = np.zeros((atts, neqs*k_ar, neqs*k_ar))
collect_lam_0 = []
collect_lam_1 = []
print "Initial estimation"
for a in range(atts):
    print str(a)
    sim_run = robust(method=meth, mod_data=mod_data, mod_yc_data=mod_yc_data)
    lam_0_coll[a] = sim_run[0]
    lam_1_coll[a] = sim_run[1]

quantiles = [0, 10, 25, 50, 75, 90, 100]
for quant in quantiles:
    collect_lam_0.append((str(quant), np.percentile(lam_0_coll, quant,
                                                    axis=0)))
    collect_lam_1.append((str(quant), np.percentile(lam_1_coll, quant,
                                                    axis=0)))

pickle_file(collect_lam_0, "../temp_res/collect_lam_0_ls")
pickle_file(collect_lam_1, "../temp_res/collect_lam_1_ls")

#use medians to guess for next 50 sims
atts2 = 10
print "Second round estimation"
lam_0_all  = np.zeros((atts2, neqs*k_ar, 1))
lam_1_all  = np.zeros((atts2, neqs*k_ar, neqs*k_ar))
cov_all  = np.zeros((atts2, neqs + neqs**2, neqs + neqs**2))
collect_lam_0_ref = []
collect_lam_1_ref = []
collect_cov_ref = []
for a in range(atts2):
    print str(a)
    #third element is median
    sim_run = robust(method=meth, mod_data=mod_data, mod_yc_data=mod_yc_data,
            lam_0_g=collect_lam_0[3][1], lam_1_g=collect_lam_1[3][1],
            start_date=start_date, passwd=passwd)
    lam_0_all[a] = sim_run[0]
    lam_1_all[a] = sim_run[1]
    cov_all[a] = sim_run[2]

#These estimates are getting closer to each other throughout the entire span

for quant in quantiles:
    collect_lam_0_ref.append((str(quant), np.percentile(lam_0_all, quant,
                                                        axis=0)))
    collect_lam_1_ref.append((str(quant), np.percentile(lam_1_all, quant,
                                                        axis=0)))
    collect_cov_ref.append((str(quant), np.percentile(cov_all, quant, axis=0)))

#Collect results
pickle_file(lam_0_all, "../temp_res/lam_0_all_ls")
pickle_file(lam_1_all, "../temp_res/lam_0_all_ls")
pickle_file(cov_all, "../temp_res/cov_all_ls")
pickle_file(collect_lam_0_ref, "../temp_res/collect_lam_0_ref_ls")
pickle_file(collect_lam_1_ref, "../temp_res/collect_lam_1_ref_ls")
pickle_file(collect_cov_ref, "../temp_res/collect_cov_ref_ls")

#send success email
#success_mail(passwd)
