import numpy as np
import socket
import pickle
import os
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.base.model import LikelihoodModel
from statsmodels.tools.numdiff import approx_hess, approx_fprime
from statsmodels.tsa.filters import hpfilter

import pandas as px
from operator import itemgetter
from scipy import optimize, stats
import scipy.linalg as la

#for sending email
import smtplib
import string

import matplotlib.pyplot as plt

import itertools as it

from affine import affine
from util import robust, pickl_file
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
    text = open("test.txt")
    passwd = text.readline()[:-1]
if comp == "bart-Inspiron-1525":
    path_pre = "/home/bart"
    #passwd = getpass.getpass(prompt="Please enter email passwd: ")
if comp == "linux-econ6":
    path_pre = "/home/labuser"

##############################################################################
# Estimate model with Eurodollar futures
##############################################################################

print "Model 1 running"

########################################
# Get data                             #
########################################

mthdata = px.read_csv(path_pre + "/Documents/Econ_630/data/VARbernankedata.csv",
                      na_values="M", index_col = 0)
new_index = []
for x in mthdata.index.tolist():
    print x
    new_index.append(dt.datetime.strptime(x, "%m/%d/%Y")) 

mthdata.index = new_index
#pdata = px.read_csv(path_pre + "/Documents/Econ_630/data/prices.txt",
#                      na_values="M", index_col = 0, parse_dates=True)

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
                     na_values = "M", index_col=0)
new_index = []
for x in ycdata.index.tolist():
    new_index.append(dt.datetime.strptime(x, "%m/%d/%Y")) 
ycdata.index = new_index

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

run_groups = []
atts = 100
np.random.seed(101)
collect_0 = []
collect_1 = []

#generate decent guesses
lam_0_coll = np.zeros((atts, neqs*k_ar, 1))
lam_1_coll = np.zeros((atts, neqs*k_ar, neqs*k_ar))
for a in range(atts):
    print str(a)
    sim_run = robust(mod_data=mod_data, mod_yc_data=mod_yc_data)
    lam_0_coll[a] = sim_run[0]
    lam_1_coll[a] = sim_run[1]

quant = [0, 10, 25, 50, 75, 90, 100]
for q in quant:
    collect_0.append((str(q), stats.scoreatpercentile(lam_0_coll[:], q)))
    collect_1.append((str(q), stats.scoreatpercentile(lam_1_coll[:], q)))

pickl_file(collect_0, "collect_0_curve")
pickl_file(collect_1, "collect_1_curve")

#use medians to guess for next 50 sims
atts2 = 50
lam_0_coll = np.zeros((atts2, neqs*k_ar, 1))
lam_1_coll = np.zeros((atts2, neqs*k_ar, neqs*k_ar))
collect_0_ref = []
collect_1_ref = []
for a in range(atts2):
    print str(a)
    #third element is median
    sim_run = robust(mod_data=mod_data, mod_yc_data=mod_yc_data,
            lam_0_g=collect_0[3][1], lam_1_g=collect_1[3][1])
    lam_0_coll[a] = sim_run[0]
    lam_1_coll[a] = sim_run[1]

#These estimates are getting closer to each other throughout the entire span 

for q in quant:
    collect_0_ref.append((str(q), stats.scoreatpercentile(lam_0_coll[:], q)))
    collect_1_ref.append((str(q), stats.scoreatpercentile(lam_1_coll[:], q)))

pickl_file(collect_0_ref, "collect_0_ref_curve")
pickl_file(collect_1_ref, "collect_1_ref_curve")

# Initialize SMTP server

print "Trying to send email"
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

print "Send mail: woohoo!"
