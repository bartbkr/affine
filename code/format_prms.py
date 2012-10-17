# coding: utf-8
import pickle
import re
import pandas as px
import numpy as np

neqs = 5

#get lambda_0 estimates
pkl_file = open("../results/bern_nls/lam_0_all_nls.pkl", "rb")
collect_0_ref = pickle.load(pkl_file)[0]
print "lambda_0 parameter estimates"
#print collect_0_ref[3][0]
out_str = str(np.round(collect_0_ref[:neqs], decimals=4))
out_str = re.sub("]", " &", out_str)
out_str = re.sub("\[+", " ", out_str)
out_str = re.sub("\n", "\n&\n", out_str)
print out_str
pkl_file.close()

#get lamda_1 estiamtes
pkl_file = open("../results/bern_nls/lam_1_all_nls.pkl", "rb")
collect_1_ref = pickle.load(pkl_file)[0]
print "lambda_1 parameter estimates"
#print collect_1_ref[3][0]
np.set_printoptions(precision=4)
out_str = str(collect_1_ref[:neqs, :neqs], decimals=4)
out_str = re.sub("]", "", out_str)
out_str = re.sub("\[+", "", out_str)
out_str = re.sub(" +", " &", out_str)
print out_str
pkl_file.close()

#get var-cov matrix
pkl_file = open("../results/bern_nls/cov_all_nls.pkl", "rb")
cov = pickle.load(pkl_file)[0]
#print cov
#variance
#print "parameter est variance"
#print np.round(np.diag(cov[3][1]), decimals=4)
#std err
print "parameter est std err"
#print np.round(np.sqrt(np.diag(cov[3][1])), decimals=4)
out_str = str(np.round(np.sqrt(np.diag(cov)), decimals=4))
out_str = re.sub("]", "", out_str)
out_str = re.sub("\[+", "", out_str)
out_str = re.sub(" +", " &", out_str)
print out_str
pkl_file.close()
