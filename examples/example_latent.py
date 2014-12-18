import numpy as np
import pandas as px
import datetime as dt

import socket
import atexit

from statsmodels.tsa.api import VAR
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.sandbox.pca import Pca
from scipy import stats
from affine.model.affine import Affine
from affine.constructors.helper import (pickle_file, success_mail, to_mth,
                                        gen_guesses, ap_constructor, pass_ols)

########################################
# Get macro data                       #
########################################
mthdata = px.read_csv("./data/macro_data.csv", na_values="M", sep=";",
                      index_col = 0, parse_dates=True)

index = mthdata['Total_Nonfarm_employment_seas'].dropna().index
nonfarm = mthdata['Total_Nonfarm_employment_seas'].dropna()
tr_empl_gap, hp_ch = hpfilter(nonfarm, lamb=129600)

mthdata['tr_empl_gap'] = px.Series(tr_empl_gap, index=index)
mthdata['hp_ch'] = px.Series(hp_ch, index=index)

mthdata['tr_empl_gap_perc'] = mthdata['tr_empl_gap']/mthdata['hp_ch']

#output
output = mthdata.reindex(columns=['unemployment_seas', 'indust_prod_seas',
                                  'help_wanted_index']).dropna()
output['empl_gap'] = mthdata['tr_empl_gap_perc']

#normalize each output value to zero mean and unit variance
for var in output.columns:
    output[var + "_norm"] = (output[var] - output[var].mean()) / \
                                output[var].std()
output = output.filter(regex=".*_norm")

#retrieve first PC
output_pca = Pca(data=output.values.T, names=output.columns.tolist())
output_pca1_data = output_pca.project(nPCs=1)
output['output_pca1'] = px.Series(output_pca1_data.T.tolist()[0],
                                  index=output.index)

#prices
prices = px.DataFrame(index=index)
prices['CPI_infl'] = mthdata['CPI_seas'].diff(periods=12)/ \
       mthdata['CPI_seas'] * 100
prices['PPI_infl'] = mthdata['PPI_seas'].diff(periods=12)/ \
       mthdata['CPI_seas'] * 100

#normalize each price value to zero mean and unit variance
for var in prices.columns:
    prices[var + "_norm"] = (prices[var] - prices[var].mean()) / \
                                prices[var].std()
prices = prices.filter(regex=".*_norm").dropna()

#retrieve first PC
prices_pca = Pca(data=prices.values.T, names=prices.columns.tolist())
prices_pca1_data = prices_pca.project(nPCs=1)
prices['price_pca1'] = px.Series(prices_pca1_data.T.tolist()[0],
                                 index=prices.index)
macro_data = output.join(prices)

macro_data = macro_data.join(mthdata).reindex(columns=['price_pca1',
                                                       'output_pca1']).dropna()
macro_data = macro_data.join(mthdata['fed_funds'], how='left')
#########################################
# Set up affine affine model            #
#########################################
k_ar = 4
latent = 3

#############################################
# Grab yield curve data                     #
#############################################

ycdata = px.read_csv("./data/yield_curve.csv", na_values = "M", sep=";",
                     index_col=0, parse_dates=True)

ycdata["trb_m1"] = mthdata["fed_funds"]

yc_cols = ['trcr_y1', 'trcr_y2', 'trcr_y3', 'trcr_y5', 'trcr_y7', 'trcr_y10']
mod_yc_data_nodp = ycdata[yc_cols]
mod_yc_data_nodp.rename(columns={'trcr_y1': 'trcr_q4', 'trcr_y2':
                                 'trcr_q8', 'trcr_y3': 'trcr_q12',
                                 'trcr_y5': 'trcr_q20', 'trcr_y7':
                                 'trcr_q28', 'trcr_y10': 'trcr_q40'},
                        inplace=True)
mod_yc_data = mod_yc_data_nodp.dropna(axis=0)
mod_yc_data = mod_yc_data.join(macro_data['fed_funds'], how='right')
mod_yc_data.insert(0, 'trcr_m1', mod_yc_data['fed_funds'])
rf_rate = mod_yc_data['fed_funds']
mod_yc_data = mod_yc_data.drop(['fed_funds'], axis=1)
mod_yc_data.dropna(inplace=True)

#mod_yc_data = to_mth(mod_yc_data)

mats = [4, 8, 12, 20, 28, 40]
del mod_yc_data['trcr_m1']

macro_data_use = macro_data.dropna().reindex(columns=['price_pca1',
                                                      'output_pca1'])
macro_data_use = macro_data.ix[mod_yc_data.index]

yc_data_use = mod_yc_data.ix[macro_data_use.index[k_ar:]]
rf_rate = rf_rate.ix[macro_data_use.index]

#align number of obs between yields and grab rf rate
#mth_only = to_mth(mod_yc_data)

neqs = len(macro_data_use.columns)

#This is a constructor function for easiloy setting up the system ala Ang and
#Piazzessi 2003
lam_0_e, lam_1_e, delta_0_e, delta_1_e, mu_e, phi_e, sigma_e \
    = ap_constructor(k_ar=k_ar, neqs=neqs, lat=latent)

delta_0_e, delta_1_e, mu_e, phi_e, sigma_e = pass_ols(var_data=macro_data_use,
                                                      freq="Q", lat=latent,
                                                      k_ar=k_ar, neqs=neqs,
                                                      delta_0=delta_0_e,
                                                      delta_1=delta_1_e,
                                                      mu=mu_e, phi=phi_e,
                                                      sigma=sigma_e,
                                                      rf_rate=rf_rate)

mod_init = Affine(yc_data=yc_data_use, var_data=macro_data_use, latent=latent,
                  no_err=[0, 2, 4], lam_0_e=lam_0_e, lam_1_e=lam_1_e,
                  delta_0_e=delta_0_e, delta_1_e=delta_1_e, mu_e=mu_e,
                  phi_e=phi_e, sigma_e=sigma_e, mats=mats, lags=k_ar,
                  neqs=neqs)

guess_length = mod_init.guess_length

guess_params = [0.0000] * guess_length

np.random.seed(100)

for numb, element in enumerate(guess_params[:30]):
    element = 0.0001
    guess_params[numb] = element * (np.random.random() - 0.5)

# #This is for nls method, only need guesses for lam_0, lam_1
# #bsr_solve = mod_init.solve(lam_0_g=lam_0_g, lam_1_g=lam_1_g, method="nls")
#bsr_solve = mod_init.solve(guess_params=guess_params, method="ml",
#                            alg="newton", maxfev=10000000, maxiter=10000000)
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
