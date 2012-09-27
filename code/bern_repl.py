"""
This script attempts to replicate the Bernanke, Sack, and Reinhart (2004) model
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
                                   'output_deflator_year_ahead',
                                    'fed_funds',
                                    'ed_fut']).dropna(axis=0)
