"""
This script attempts to solve the model with unknown variables
"""
import numpy as np
import pandas as px

import socket
import atexit
import keyring

from statsmodels.tsa.api import VAR
from statsmodels.tsa.filters import hpfilter
from scipy import stats
from util import pickle_file, success_mail, fail_mail, to_mth, gen_guesses

#identify computer
#identify computer
comp = socket.gethostname()

########################################
# Get macro data                       #
########################################
mthdata = px.read_csv("../data/VARbernankedata.csv", na_values="M",
                        index_col = 0, parse_dates=True)

mthdata['tr_empl_gap'], mthdata['hp_ch'] = \
    hpfilter(mthdata['Total_Nonfarm_employment'], lamb=129600)
mthdata['tr_empl_gap_perc'] = mthdata['tr_empl_gap']/mthdata['hp_ch']
mthdata['act_infl'] = \
    mthdata['Pers_Cons_P'].diff(periods=12)/mthdata['Pers_Cons_P']*100
mthdata['ed_fut'] = 100 - mthdata['one_year_ED']

#define final data set
mod_data = mthdata.reindex(columns=['tr_empl_gap_perc',
                                   'act_infl',
                                   'inflexp_1yr_mean',
                                    'fed_funds',
                                    'ed_fut']).dropna(axis=0)

#########################################
# Set up affine affine model            #
#########################################
k_ar = 4
neqs = len(mod_data.columns)
latent = 0

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
#align number of obs between yields and grab rf rate
mod_yc_data = mod_yc_data_nodp.dropna(axis=0)
mod_yc_data = mod_yc_data.join(x_t['fed_funds'], how='right')
mod_yc_data = mod_yc_data.rename(columns = {'fed_funds' : 'l_tr_m1'})
rf_rate = mod_yc_data["l_tr_m1"]
mod_yc_data = mod_yc_data.drop(['l_tr_m1'], axis=1)

mth_only = to_mth(mod_yc_data)

from affine import Affine

#bsr_model = Affine(yc_data=mth_only, var_data=mod_data, rf_rate=rf_rate,
                   #latent=3, no_err=[3, 36, 120])
bsr_model = Affine(yc_data=mth_only, var_data=mod_data, rf_rate=rf_rate,
                   latent=latent, no_err=[3, 36, 120])

lam_0_g, lam_1_g, delta_1_g, mu_g, phi_g, sigma_g = gen_guesses(k_ar=k_ar,
                                                                neqs=neqs,
                                                                latent=latent)
#This is for nls method, only need guesses for lam_0, lam_1
#bsr_solve = bsr_model.solve(lam_0_g=lam_0_g, lam_1_g=lam_1_g, method="nls")
bsr_solve = bsr_model.solve(lam_0_g=lam_0_g, lam_1_g=lam_1_g,
                            delta_1_g=delta_1_g, mu_g=mu_g, phi_g=phi_g,
                            sigma_g=sigma_g, method="ml", alg="newton")

lam_0 = bsr_solve[0]
lam_1 = bsr_solve[1]

print lam_0
print lam_1
