# coding: utf-8
import pickle
import re
import pandas as px
import numpy as np

#!!!need to get easier output into table

res_loc = "../results/"

#get lambda_0 estimates
pkl_file = open(res_loc + "collect_0_ref_curve.pkl", "rb")
collect_0_ref = pickle.load(pkl_file)
print "lambda_0 parameter estimates"
#print collect_0_ref[3][0]
out_str = str(np.round(collect_0_ref[3][1], decimals=4))
out_str = re.sub("]", " &", out_str)
out_str = re.sub("\[+", " ", out_str)
out_str = re.sub("\n", "\n&\n", out_str)
print out_str
pkl_file.close()

#get lamda_1 estiamtes
pkl_file = open(res_loc + "collect_1_ref_curve.pkl", "rb")
collect_1_ref = pickle.load(pkl_file)
print "lambda_1 parameter estimates"
#print collect_1_ref[3][0]
#print np.round(collect_1_ref[3][1][:5, :5], decimals=4)
out_str = str(np.round(collect_1_ref[3][1][:5, :5], decimals=4))
out_str = re.sub("]", "", out_str)
out_str = re.sub("\[+", "", out_str)
out_str = re.sub(" +", " &", out_str)
print out_str
pkl_file.close()

#get var-cov matrix
pkl_file = open(res_loc + "collect_cov_ref.pkl", "rb")
cov = pickle.load(pkl_file)
print cov[3][0]
#variance
#print "parameter est variance"
#print np.round(np.diag(cov[3][1]), decimals=4)
#std err
print "parameter est std err"
#print np.round(np.sqrt(np.diag(cov[3][1])), decimals=4)
out_str = str(np.round(np.sqrt(np.diag(cov[3][1])), decimals=4))
out_str = re.sub("]", "", out_str)
out_str = re.sub("\[+", "", out_str)
out_str = re.sub(" +", " &", out_str)
print out_str
pkl_file.close()
