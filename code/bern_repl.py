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
from scipy import stats
from util import pickle_file, success_mail, fail_mail, to_mth, gen_guesses, \
                    robust

start_date = dt.datetime.now()
passwd = keyring.get_password("email_auth", "bartbkr")

########################################
# Get macro data                       #
########################################
mthdata = px.read_csv("../data/VARbernankedata.csv", na_values="M",
                        index_col = 0, parse_dates=True)

nonfarm = mthdata['Total_Nonfarm_employment'].dropna()

tr_empl_gap, hp_ch = hpfilter(nonfarm, lamb=129600)

mthdata['tr_empl_gap'] = px.Series(tr_empl_gap, index=nonfarm.index)
mthdata['hp_ch'] = px.Series(hp_ch, index=nonfarm.index)
mthdata['tr_empl_gap_perc'] = mthdata['tr_empl_gap']/mthdata['hp_ch'] * 100
mthdata['act_infl'] = \
    mthdata['Pers_Cons_P'].diff(periods=12)/mthdata['Pers_Cons_P']*100
mthdata['ed_fut'] = 100 - mthdata['one_year_ED']

#define final data set
mod_data = mthdata.reindex(columns=['tr_empl_gap_perc',
                                   'act_infl',
                                   'gnp_gdp_deflat_nxtyr',
                                    'fed_funds',
                                    'ed_fut']).dropna(axis=0)

neqs = 5
k_ar = 4

sys.exit("Stop here")

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

ycdata = px.read_csv("../data/yield_curve.csv",
                     na_values = "M", index_col=0, parse_dates=True)

mod_yc_data_nodp = ycdata.reindex(columns=['l_tr_m3', 'l_tr_m6',
                                      'l_tr_y1', 'l_tr_y2',
                                      'l_tr_y3', 'l_tr_y5',
                                      'l_tr_y7', 'l_tr_y10'])
mod_yc_data = mod_yc_data_nodp.dropna(axis=0)
mod_yc_data = mod_yc_data.join(x_t['fed_funds'], how='right')
mod_yc_data.insert(0, 'l_tr_m1', mod_yc_data['fed_funds'])
mod_yc_data = mod_yc_data.drop(['fed_funds'], axis=1)

mod_yc_data = to_mth(mod_yc_data)

# Setup model
meth = "ls"
run_groups = []
atts = 100
np.random.seed(101)
collect_0 = []
collect_1 = []

##################################################
# Define exit message to send to email upon fail #
##################################################
#atexit.register(fail_mail, start_date, passwd)

#generate decent guesses
lam_0_coll = np.zeros((atts, neqs*k_ar, 1))
lam_1_coll = np.zeros((atts, neqs*k_ar, neqs*k_ar))
print "Initial estimation"
for a in range(atts):
    print str(a)
    sim_run = robust(method=meth, mod_data=mod_data, mod_yc_data=mod_yc_data)
    lam_0_coll[a] = sim_run[0]
    lam_1_coll[a] = sim_run[1]

quant = [0, 10, 25, 50, 75, 90, 100]
for q in quant:
    collect_0.append((str(q), stats.scoreatpercentile(lam_0_coll[:], q)))
    collect_1.append((str(q), stats.scoreatpercentile(lam_1_coll[:], q)))

pickle_file(collect_0, "collect_0_curve_ls")
pickle_file(collect_1, "collect_1_curve_ls")

#use medians to guess for next 50 sims
atts2 = 50
print "Second round estimation"
lam_0_coll = np.zeros((atts2, neqs*k_ar, 1))
lam_1_coll = np.zeros((atts2, neqs*k_ar, neqs*k_ar))
cov_coll = np.zeros((atts2, neqs + neqs**2, neqs + neqs**2))
collect_0_ref = []
collect_1_ref = []
collect_cov_ref = []
for a in range(atts2):
    print str(a)
    #third element is median
    sim_run = robust(method=meth, mod_data=mod_data, mod_yc_data=mod_yc_data,
            lam_0_g=collect_0[3][1], lam_1_g=collect_1[3][1],
            start_date=start_date, passwd=passwd)
    lam_0_coll[a] = sim_run[0]
    lam_1_coll[a] = sim_run[1]
    cov_coll[a] = sim_run[2]

#These estimates are getting closer to each other throughout the entire span 

for q in quant:
    collect_0_ref.append((str(q), stats.scoreatpercentile(lam_0_coll[:], q)))
    collect_1_ref.append((str(q), stats.scoreatpercentile(lam_1_coll[:], q)))
    collect_cov_ref.append((str(q), stats.scoreatpercentile(cov_coll[:], q)))

pickle_file(collect_0_ref, "collect_0_ref_ls")
pickle_file(collect_1_ref, "collect_1_ref_ls")
pickle_file(collect_cov_ref, "collect_cov_ref_ls")

#send success email
success_mail(passwd)
