# coding: utf-8
import pickle
import pandas as px
import numpy as np

#!!!need to get easier output into table

#get lambda_0 estimates
pkl_file = open("collect_0_ref_curve.pkl", "rb")
collect_0_ref = pickle.load(pkl_file)
print "lambda_0 parameter estimates"
print collect_0_ref[3][0]
print(np.round(collect_0_ref[3][1], decimals=4))
for row in np.size(collect_0_ref[3][1], axis=0):

pkl_file.close()

#get lamda_1 estiamtes
pkl_file = open("collect_1_ref_curve.pkl", "rb")
collect_1_ref = pickle.load(pkl_file)
print "lambda_1 parameter estimates"
print collect_1_ref[3][0]
print(np.round(collect_1_ref[3][1][:5, :5], decimals=4))
pkl_file.close()

#get var-cov matrix
pkl_file = open("collect_cov_ref.pkl", "rb")
cov = pickle.load(pkl_file)
print cov[3][0]
#variance
print "parameter est variance"
print np.round(np.diag(cov[3][1]), decimals=4)
#std err
print "parameter est std err"
print np.round(np.sqrt(np.diag(cov[3][1])), decimals=4)
pkl_file.close()
