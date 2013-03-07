"""
This script generates meaningful graphs and statistical from previous run
"""
import numpy as np
import pandas as px
import datetime as dt
import matplotlib.pyplot as plt

import pickle

from affine import Affine
from statsmodels.tsa.api import VAR
from statsmodels.tsa.filters import hpfilter
from scipy import stats
from util import pickle_file, success_mail, fail_mail, to_mth, gen_guesses, \
                    robust

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
                                    'ed_fut4']).dropna(axis=0)

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

#subset to range specified in BSR

var_dates = px.date_range("3/1/1982", "12/1/2004", freq="MS").to_pydatetime()
yc_dates = px.date_range("6/1/1982", "12/1/2004", freq="MS").to_pydatetime()

mod_data = mod_data.ix[var_dates]
mod_yc_data = mod_yc_data.ix[yc_dates]

mod_data = mod_data.ix[var_dates]
mod_yc_data = mod_yc_data.ix[yc_dates]
bsr = Affine(yc_data = mod_yc_data, var_data = mod_data)

lam_0_nr = np.zeros([neqs*k_ar, 1])
lam_1_nr = np.zeros([neqs*k_ar, neqs*k_ar])
sigma_zeros = np.zeros_like(bsr.sigma_ols)

#grab previous run
pkl_file = open("../results/bern_nls/lam_0_all_nls.pkl", "rb")
final_lam_0 = pickle.load(pkl_file)[0]
pkl_file.close()

pkl_file = open("../results/bern_nls/lam_1_all_nls.pkl", "rb")
final_lam_1 = pickle.load(pkl_file)[0]
pkl_file.close()


a_rsk, b_rsk = bsr.gen_pred_coef(lam_0=final_lam_0, lam_1=final_lam_1,
                                   delta_1=bsr.delta_1_nolat, mu=bsr.mu_ols,
                                   phi=bsr.phi_ols, sigma=bsr.sigma_ols)

#generate no risk results
a_nrsk, b_nrsk = bsr.gen_pred_coef(lam_0=lam_0_nr, lam_1=lam_1_nr,
                                   delta_1=bsr.delta_1_nolat, mu=bsr.mu_ols,
                                   phi=bsr.phi_ols, sigma=sigma_zeros)
#gen BSR predicted
X_t = bsr.var_data_vert
per = bsr.yc_data.index
act_pred = px.DataFrame(index=per)
for i in bsr.mths:
    act_pred[str(i) + '_mth_act'] = bsr.yc_data['trcr_m' + str(i)]
    act_pred[str(i) + '_mth_pred'] = a_rsk[i-1] + \
                                    np.dot(b_rsk[i-1].T, X_t.values.T)[0]
    act_pred[str(i) + '_mth_nrsk'] = a_nrsk[i-1] + \
                                    np.dot(b_nrsk[i-1].T, X_t.values.T)[0]
    act_pred[str(i) + '_mth_err'] = np.abs(act_pred[str(i) + '_mth_act'] -
                                            act_pred[str(i) + '_mth_pred'])
ten_yr = act_pred.reindex(columns = filter(lambda x: '120' in x, act_pred))
seven_yr = act_pred.reindex(columns = filter(lambda x: '84' in x, act_pred))
five_yr = act_pred.reindex(columns = filter(lambda x: '60' in x,act_pred))
three_yr = act_pred.reindex(columns = filter(lambda x: '36' in x,act_pred))
two_yr = act_pred.reindex(columns = filter(lambda x: '24' in x,act_pred))
one_yr = act_pred.reindex(columns = ['12_mth_act',
                                     '12_mth_pred',
                                     '12_mth_nrsk',
                                     '12_mth_err'])
six_mth = act_pred.reindex(columns = ['6_mth_act',
                                      '6_mth_pred',
                                      '6_mth_nrsk',
                                      '6_mth_err'])
#plot the term premium
ten_yr_plot = ten_yr.reindex(columns = ['120_mth_act', '120_mth_pred',
                                        '120_mth_nrsk'])
fig = ten_yr_plot.plot(style=['-', '--', '-.'], legend=False)
x1, x2, y1, y2 = fig.axis()
fig.axis([x1, x2, 0, 14])
handles, old_labels = fig.get_legend_handles_labels()
fig.legend(handles, ('Actual', 'Predicted', 'Risk-neutral'))
plt.savefig("../../diss_writeup/figures/tenyr_rep.png")
#two year
two_yr_plot = two_yr.reindex(columns = ['24_mth_act', '24_mth_pred',
                                        '24_mth_nrsk'])
fig = two_yr_plot.plot(style=['-', '--', '-.'])
handles, old_labels = fig.get_legend_handles_labels()
fig.axis([x1, x2, 0, 14])
fig.legend(handles, ('Actual', 'Predicted', 'Risk-neutral'))
plt.savefig("../../diss_writeup/figures/twoyr_rep.png")

#generate st dev of residuals
yields = ['six_mth', 'one_yr', 'two_yr', 'three_yr', 'five_yr', 'seven_yr',
          'ten_yr']
for yld in yields:
    print yld + " & " + str(np.std(eval(yld).filter(regex='.*err$').values,
                            ddof=1))
