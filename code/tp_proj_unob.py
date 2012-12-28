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
from core.util import pickle_file, success_mail, to_mth, gen_guesses, \
                 ap_constructor, pass_ols

########################################
# Get macro data                       #
########################################
mthdata = px.read_csv("../data/VARbernankedata.csv", na_values="M",
                        index_col = 0, parse_dates=True)

index = mthdata['Total_Nonfarm_employment'].index
tr_empl_gap, hp_ch = hpfilter(mthdata['Total_Nonfarm_employment'], lamb=129600)

mthdata['tr_empl_gap'] = px.Series(tr_empl_gap, index=index)
mthdata['hp_ch'] = px.Series(hp_ch, index=index)

mthdata['tr_empl_gap_perc'] = mthdata['tr_empl_gap']/mthdata['hp_ch']
mthdata['act_infl'] = \
    mthdata['Pers_Cons_P'].diff(periods=12)/mthdata['Pers_Cons_P']*100
mthdata['ed_fut'] = 100 - mthdata['one_year_ED']

#define final data set
mod_data = mthdata.reindex(columns=['tr_empl_gap_perc',
                                   'act_infl', 
                                   'fed_funds']).dropna(axis=0)

#########################################
# Set up affine affine model            #
#########################################
k_ar = 4
lat = 3
latent = True

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

ycdata = px.read_csv("../data/yield_curve.csv", na_values = "M", index_col=0,
                     parse_dates=True)

mod_yc_data_nodp = ycdata.reindex(columns=['l_tr_m3', 'l_tr_m6', 'l_tr_y1',
                                           'l_tr_y2', 'l_tr_y3', 'l_tr_y5',
                                           'l_tr_y7', 'l_tr_y10'])

#align number of obs between yields and grab rf rate
mod_yc_data = mod_yc_data_nodp.dropna(axis=0)
mod_yc_data = mod_yc_data.join(x_t['fed_funds'], how='right')
mod_yc_data = mod_yc_data.rename(columns = {'fed_funds' : 'l_tr_m1'})
mod_yc_data = mod_yc_data.drop(['l_tr_m1'], axis=1).dropna()

rf_rate = mod_data["fed_funds"]

mth_only = to_mth(mod_yc_data)
yc_index = mth_only.index

#for affine model, only want two macro vars
mod_data = mod_data.reindex(columns=['tr_empl_gap_perc', 'act_infl'])
mod_index = px.date_range("10/1/1981", 
                yc_index[-1].to_pydatetime().strftime("%m/%d/%Y"), freq="MS")
mod_data = mod_data.reindex(index=mod_index)

rf_rate = rf_rate.reindex(index=yc_index)

neqs = len(mod_data.columns)

from core.affine import Affine

lam_0_e, lam_1_e, delta_1_e, mu_e, phi_e, sigma_e = ap_constructor(k_ar=k_ar,
                                                                   neqs=neqs, 
                                                                   lat=lat)

delta_1_e, mu_e, phi_e, sigma_e = pass_ols(var_data=mod_data, freq="M", lat=3,
                                           k_ar=4, neqs=2, delta_1=delta_1_e, mu=mu_e,
                                           phi=phi_e, sigma=sigma_e,
                                           rf_rate=rf_rate)

#lam_0_guess = [0.0] * (neqs + lat)
#lam_1_guess = [0.0] * ((neqs * neqs) + (neqs * lat) + (lat * neqs) + 
              #(lat * lat))
#delta_1_guess = [0.0] * (lat)
#mu_guess = [0.0] * (lat)
#phi_guess = [0.0] * (lat * lat)
#sigma_guess = [0.0] * (lat * lat)

bsr_model = Affine(yc_data=mth_only, var_data=mod_data, rf_rate=rf_rate,
                   latent=latent, no_err=[0, 4, 7], lam_0_e=lam_0_e,
                   lam_1_e=lam_1_e, delta_1_e=delta_1_e, mu_e=mu_e,
                   phi_e=phi_e, sigma_e=sigma_e)

guess_length = bsr_model.guess_length

guess_params = [0.0000] * guess_length

for numb, element in enumerate(guess_params[:30]):
    element = 0.0001
    guess_params[numb] = element * (np.random.random() - 0.5)

# #This is for nls method, only need guesses for lam_0, lam_1
# #bsr_solve = bsr_model.solve(lam_0_g=lam_0_g, lam_1_g=lam_1_g, method="nls")
bsr_solve = bsr_model.solve(guess_params=guess_params, method="ml",
                            alg="newton", maxfev=10000000, maxiter=10000000)
# 
# lam_0 = bsr_solve[0]
# lam_1 = bsr_solve[1]
# 
# lam_0, lam_1, delta_1, mu, phi, sigma, a_solve, b_solve, tvalues = bsr_solve
# 
# print "lam_0"
# print lam_0
# print "lam_1"
# print lam_1
# print "delta_1"
# print delta_1
# print "mu"
# print mu
# print "phi"
# print phi
# print "sigma"
# print sigma
# print "a_solve"
# print a_solve
# print "b_solve"
# print b_solve
# print "tvalues"
# print tvalues
# 
# #send success email
# passwd = keyring.get_password("email_auth", "bartbkr") 
# success_mail(passwd)
