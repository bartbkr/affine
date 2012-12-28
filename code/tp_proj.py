"""
This script imports the data, gets the appropriate values, and executes the
solution algorithm
"""
import numpy as np
import pandas as px

import socket
import atexit
import keyring

from statsmodels.tsa.api import VAR
from statsmodels.tsa.filters import hpfilter
from scipy import stats
from core.util import robust, pickle_file, success_mail, to_mth

##############################################################################
# Estimate model with Eurodollar futures
##############################################################################

print "Model 1 running"

########################################
# Get data                             #
########################################

mthdata = px.read_csv("../data/VARbernankedata.csv", na_values="M",
                        index_col = 0, parse_dates=True)

########################################
# Test that HP filter/empl gap correct #
########################################
mthdata['tr_empl_gap'], mthdata['hp_ch'] = hpfilter(mthdata['Total_Nonfarm_employment'], lamb=129600)
mthdata['tr_empl_gap_perc'] = mthdata['tr_empl_gap']/mthdata['hp_ch']

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

#########################################
# Set up affine affine model               #
#########################################
k_ar = vreg.k_ar

#create BSR x_t
x_t_na = mod_data.copy()
for t in range(k_ar-1):
    for var in mod_data.columns:
        x_t_na[var + '_m' + str(t+1)] = px.Series(mod_data[var].values[:-(t+1)],
                                            index=mod_data.index[t+1:])
#remove missing values
x_t = x_t_na.dropna(axis=0)


#############################################
# Grab yield curve data                     #
#############################################

ycdata = px.read_csv("../data/yield_curve.csv",
                     na_values = "M", index_col=0, parse_dates=True)

mod_yc_data_nodp = ycdata.reindex(columns=['l_tr_m3', 'l_tr_m6',
                                      'l_tr_y1', 'l_tr_y2',
                                      'l_tr_y3', 'l_tr_y5',
                                      'l_tr_y7', 'l_tr_y10'])
mod_yc_data = mod_yc_data_nodp.dropna(axis=0)
mod_yc_data = mod_yc_data.join(x_t['fed_funds'], how='right')
mod_yc_data = mod_yc_data.rename(columns = {'fed_funds' : 'l_tr_m1'})
mod_yc_data = mod_yc_data.drop(['l_tr_m1'], axis=1)
mth_only = to_mth(mod_yc_data)


################################################
# Generate predictions of cumulative shortrate #
################################################

#10 yr = 120 mths
# mths_pred = 120
# pred = np.zeros((vreg.neqs, len(mod_data)-vreg.k_ar))
# k_ar = vreg.k_ar
# neqs = vreg.neqs
# params = vreg.params.values.copy()
# avg_12 = []
# for t in range(k_ar, len(mod_data)):
#     s_data = mod_data[t-k_ar:t].values
#     for i in range(120):
#         new_dat = np.zeros((1, neqs))
#         for p in range(neqs):
#             new_dat[0, p] = params[0, p] + np.dot(np.flipud(s_data)[:k_ar].flatten()[None], params[1:,p,None])[0]
#         s_data = np.append(s_data, new_dat, axis=0)
#     avg_12.append(np.mean(s_data[:, 3]))
# 
# #implied term premium
# var_tp = px.DataFrame(index=mth_only.index[1:], data=np.asarray(avg_12), columns=["pred_12"])
# var_tp['act_12'] = mth_only['l_tr_y10']
# var_tp['VAR term premium'] = var_tp['act_12'] - var_tp['pred_12']

##################################################
# Define exit message to send to email upon fail #
##################################################
#atexit.register(fail_mail, start_date, passwd)

#############################################
# Testing                                   #
#############################################

run_groups = []
k_ar = vreg.k_ar
neqs = vreg.neqs
atts = 2
np.random.seed(101)
collect_0 = []
collect_1 = []

meth = "ls"

#generate decent guesses
lam_0_coll = np.zeros((atts, neqs*k_ar, 1))
lam_1_coll = np.zeros((atts, neqs*k_ar, neqs*k_ar))
for a in range(atts):
    print str(a)
    sim_run = robust(method=meth, mod_data=mod_data, mod_yc_data=mth_only)
    lam_0_coll[a] = sim_run[0]
    lam_1_coll[a] = sim_run[1]

quant = [0, 10, 25, 50, 75, 90, 100]
for q in quant:
    collect_0.append((str(q), stats.scoreatpercentile(lam_0_coll[:], q)))
    collect_1.append((str(q), stats.scoreatpercentile(lam_1_coll[:], q)))

pickle_file(collect_0, "collect_0_curve")
pickle_file(collect_1, "collect_1_curve")

#use medians to guess for next 50 sims
atts2 = 1
lam_0_coll = np.zeros((atts2, neqs*k_ar, 1))
lam_1_coll = np.zeros((atts2, neqs*k_ar, neqs*k_ar))
cov_coll = np.zeros((atts2, neqs + neqs**2, neqs + neqs**2))
collect_0_ref = []
collect_1_ref = []
collect_cov_ref = []
for a in range(atts2):
    print str(a)
    #third element is median
    sim_run = robust(method=meth, mod_data=mod_data, mod_yc_data=mth_only,
            lam_0_g=collect_0[3][1], lam_1_g=collect_1[3][1], passwd=passwd)
    lam_0_coll[a] = sim_run[0]
    lam_1_coll[a] = sim_run[1]
    cov_coll[a] = sim_run[2]

#These estimates are getting closer to each other throughout the entire span 

for q in quant:
    collect_0_ref.append((str(q), stats.scoreatpercentile(lam_0_coll[:], q)))
    collect_1_ref.append((str(q), stats.scoreatpercentile(lam_1_coll[:], q)))
    collect_cov_ref.append((str(q), stats.scoreatpercentile(cov_coll[:], q)))

pickle_file(collect_0_ref, "collect_0_ref_curve")
pickle_file(collect_1_ref, "collect_1_ref_curve")
pickle_file(collect_cov_ref, "collect_cov_ref")

#send success email
success_mail(passwd)
