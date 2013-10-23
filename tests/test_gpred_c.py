import numpy as np
import pandas as px

import socket
import atexit
import keyring

from statsmodels.tsa.api import VAR
from statsmodels.tsa.filters import hpfilter
from statsmodels.sandbox.pca import Pca
from scipy import stats
from affine.model.affine import Affine
from affine.constructors.helper import (pickle_file, success_mail, to_mth,
                                        gen_guesses, ap_constructor, pass_ols)
import pdb

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

macro_data_ind = macro_data.index
#########################################
# Set up affine affine model            #
#########################################
k_ar = 4
lat = 3
latent = True

#create BSR x_t
x_t_na = macro_data.copy()
for t in range(k_ar-1):
    for var in macro_data.columns:
        x_t_na[var + '_m' + str(t+1)] = px.Series(macro_data[var].values[:-(t+1)],
                                            index=macro_data.index[t+1:])
#remove missing values
x_t = x_t_na.dropna(axis=0)

#############################################
# Grab yield curve data                     #
#############################################

ycdata = px.read_csv("./data/yield_curve.csv", na_values = "M", sep=";",
                     index_col=0, parse_dates=True)

ycdata["trb_m1"] = mthdata["fed_funds"]

ycdata = px.read_csv("./data/fama-bliss_formatted.csv", na_values = "M",
                     index_col=0, parse_dates=True, sep=",")

mths = [12, 24, 36, 48, 60]
final_ind = ycdata.index
yc_data_use = ycdata.reindex(index=final_ind[k_ar - 1:])

#align number of obs between yields and grab rf rate
#mth_only = to_mth(mod_yc_data)

#for affine model, only want two macro vars
macro_data_use = macro_data.reindex(index=final_ind)
rf_rate = macro_data_use["fed_funds"]

neqs = len(macro_data_use.columns)

#This is a constructor function for easiloy setting up the system ala Ang and
#Piazzessi 2003
lam_0_e, lam_1_e, delta_0_e, delta_1_e, mu_e, phi_e, sigma_e \
    = ap_constructor(k_ar=k_ar, neqs=neqs, lat=lat)

delta_0_e, delta_1_e, mu_e, phi_e, sigma_e = pass_ols(var_data=macro_data_use,
                                                      freq="M", lat=lat,
                                                      k_ar=k_ar, neqs=neqs,
                                                      delta_0=delta_0_e,
                                                      delta_1=delta_1_e,
                                                      mu=mu_e, phi=phi_e,
                                                      sigma=sigma_e,
                                                      rf_rate=rf_rate)

bsr_model = Affine(yc_data=yc_data_use, var_data=macro_data_use,
                   latent=latent, no_err=[0, 2, 4],
                   lam_0_e=lam_0_e, lam_1_e=lam_1_e, delta_0_e=delta_0_e,
                   delta_1_e=delta_1_e, mu_e=mu_e, phi_e=phi_e,
                   sigma_e=sigma_e, mths=mths)

guess_length = bsr_model.guess_length

guess_params = [0.0000] * guess_length

np.random.seed(100)

for numb, element in enumerate(guess_params[:30]):
    element = 0.0001
    guess_params[numb] = element * (np.random.random() - 0.5)

lam_0, lam_1, delta_0, delta_1, mu, phi, sigma = \
                    bsr_model._params_to_array(guess_params)

opt_a_solve, opt_b_solve = bsr_model.opt_gen_pred_coef(lam_0, lam_1, delta_0,
                                                       delta_1, mu, phi, sigma)
