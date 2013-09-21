"""
This script imports the data, gets the appropriate values, and executes the
solution algorithm
"""
import numpy as np
import pandas as px

import socket
import atexit
import keyring
import cProfile

from numpy import ma
from statsmodels.tsa.api import VAR
from statsmodels.tsa.filters import hpfilter
from scipy import stats
from core.util import robust, pickle_file, success_mail, to_mth
from core.affine import Affine

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

# subset to pre 2005
mod_data = mod_data[:217]
mod_yc_data = mod_yc_data[:214]

k_ar = 4
neqs = 5
lat = 0

lam_0_e = ma.zeros((k_ar * neqs, 1))
lam_0_e[:neqs] = ma.masked

lam_1_e = ma.zeros((k_ar * neqs, k_ar * neqs))
lam_1_e[:neqs, :neqs] = ma.masked

delta_1_e = ma.zeros([k_ar * neqs, 1])
delta_1_e[:, :] = ma.masked
delta_1_e[:, :] = ma.nomask
delta_1_e[np.argmax(mod_data.columns == 'fed_funds')] = 1

var_fit = VAR(mod_data, freq="M").fit(maxlags=k_ar)

coefs = var_fit.params.values
sigma_u = var_fit.sigma_u
obs_var = neqs * k_ar

mu_e = ma.zeros([k_ar*neqs, 1])
mu_e[:, :] = ma.masked
mu_e[:, :] = ma.nomask
mu_e[:neqs] = coefs[0, None].T

phi_e = ma.zeros([k_ar * neqs, k_ar * neqs])
phi_e[:, :] = ma.masked
phi_e[:, :] = ma.nomask
phi_e[:neqs] = coefs[1:].T
phi_e[neqs:obs_var, :(k_ar - 1) * neqs] = np.identity((k_ar - 1) * neqs)

sigma_e = ma.zeros([k_ar * neqs, k_ar * neqs])
sigma_e[:, :] = ma.masked
sigma_e[:, :] = ma.nomask
sigma_e[:neqs, :neqs] = sigma_u
sigma_e[neqs:obs_var, neqs:obs_var] = np.identity((k_ar - 1) * neqs)

method = "ls"

bsr = Affine(yc_data = mod_yc_data, var_data = mod_data, lam_0_e=lam_0_e,
         lam_1_e=lam_1_e, delta_1_e=delta_1_e, mu_e=mu_e, phi_e=phi_e,
         sigma_e=sigma_e, mths=[3, 6, 12, 24, 36, 60, 84, 120])

neqs = bsr.neqs

guess_length = bsr.guess_length

guess_params = [0.0000] * guess_length

for numb, element in enumerate(guess_params[:30]):
    element = 0.0001
    guess_params[numb] = element * (np.random.random() - 0.5)

bsr.solve(guess_params=guess_params, method=method, ftol=1e-8,
        xtol=1e-8, maxfev=100, full_output=False)
"""
def profileit(func):
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile" # Name the data file sensibly
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(datafn)
        return retval

    return wrapper

@profileit
def time_it(yc_data, var_data, lam_0_e, lam_1_e, delta_1_e, mu_e, phi_e,
            sigma_e):

    method = "ls"

    bsr = Affine(yc_data = yc_data, var_data = var_data, lam_0_e=lam_0_e,
             lam_1_e=lam_1_e, delta_1_e=delta_1_e, mu_e=mu_e, phi_e=phi_e,
             sigma_e=sigma_e)

    neqs = bsr.neqs

    guess_length = bsr.guess_length

    guess_params = [0.0000] * guess_length

    for numb, element in enumerate(guess_params[:30]):
        element = 0.0001
        guess_params[numb] = element * (np.random.random() - 0.5)

    bsr.solve(guess_params=guess_params, method=method, ftol=1e-8,
            xtol=1e-8, maxfev=100, full_output=False)

ran_it = time_it(yc_data = mod_yc_data, var_data=mod_data, lam_0_e=lam_0_e,
                 lam_1_e=lam_1_e, delta_1_e=delta_1_e, mu_e=mu_e, phi_e=phi_e,
                 sigma_e=sigma_e)
"""
