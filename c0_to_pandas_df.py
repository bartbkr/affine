# coding: utf-8
import pickle
pkl_file = open("collect_0_ref.pkl")
collect_0_ref = pickle.load(pkl_file)
collect_0_ref
import pandas as px
collect_0_ref[0]
collect_0_ref[0][1]
np.size(collect_0_ref[0][1])
import numpy as np
np.size(collect_0_ref[0][1])
np.shape(collect_0_ref[0][1])
collect0_df = px.DataFrame(index=range(np.shape[0]+1))
collect0_df = px.DataFrame(index=range(np.shape(collect_0_ref)[0]+1))
np.shape(collect_0_ref)
np.shape(collect_0_ref[0])
np.shape(collect_0_ref[1])
collect_0_ref[0]
collect_0_ref[0][1[
]]
collect0_df = px.DataFrame(index=range(np.shape(collect_0_ref[0][1])[0]+1))
collect0_df
for quant in collect_0_ref[:][0]:
weogij
collect0_df = px.DataFrame(index=range(np.shape(collect_0_ref[0][1])[0]))
for i, quant in enumerate(collect_0_ref[:][0]):
    collect0_df[quant] = collect_0_ref[i][1]
    
for i, quant in enumerate(collect_0_ref[:][0]):
    print collect_0_ref[i]
    
collect_0_ref[:][0]
collect_0_ref[:]
collect_0_ref[:][0]
collect_0_ref[:,0]
collect_0_ref[:]
cols = []
for x in collect_0_ref:
    cols.append(x[0])
    
x
x
x[0]
for x in collect_0_ref:
    print x
    
for x in collect_0_ref:
    print x[0]
    
for x in collect_0_ref:
    
    
    weogj
    
cols
collect0_df = px.DataFrame(cols=cols, index=range(np.shape(collect_0_ref[0][1])[0]))
get_ipython().magic(u'pinfo px.DataFrame')
collect0_df = px.DataFrame(columns=cols, index=range(np.shape(collect_0_ref[0][1])[0]))
collect0_df
for x in cols:
woegij
data = []
data = np.zeros((range(20), len(cols))
)
range(20)
data = np.zeros((20, len(cols))
)
data
for i, x in enumerate(cols):
    data[:, i] = collect_0_ref[i][1]
    
for i, x in enumerate(cols):
    print collect_0_ref[i][1]
    
    print len(collect_0_ref[i][1])
print len(collect_0_ref[i][1])
for i, x in enumerate(cols):
    data[:, i] = collect_0_ref[i][1]
    
np.shape(data)
for i, x in enumerate(cols):
    data[:, i] = collect_0_ref[i][1]
    
for i, x in enumerate(cols):
    data[:, i] = collect_0_ref[i][1][0]
    
data
collect_0_ref
for i, x in enumerate(cols):
    print collect_0_ref[i][1]
    
for i, x in enumerate(cols):
    print collect_0_ref[i]
    
    print collect_0_ref[i][0]
for i, x in enumerate(cols):
    print collect_0_ref[i][0]
    
for i, x in enumerate(cols):
    print collect_0_ref[i][1]
    
data
data[:][0]
data[:][6]
collect0
for i, x in enumerate(cols):
    print collect_0_ref[i][1]
    
collect_0_ref[0][1]
collect_0_ref[1][1]
collect_0_ref[2][1]
collect_0_ref[3][1]
collect_0_ref[10][1]
collect_0_ref[7][1]
collect_0_ref[6][1]
np.shape(collect_0_ref[6][1])
for i, x in enumerate(cols):
    print i
    print x
    
for i, x in enumerate(cols):
    print collect_0_ref[i][0]
    print np.shape(collect_0_ref[i][1])
    
for i, x in enumerate(cols):
    print collect_0_ref[i][1]
    print np.shape(collect_0_ref[i][1])
    
for i, x in enumerate(cols):
    print np.shape(data[:,i])
    
np.shape(data[:])
np.shape(data[,i])
for i, x in enumerate(cols):
    data[:, i, None] = collect_0_ref[i][1][0]
    
data
for i, x in enumerate(cols):n
for i, x in enumerate(cols):
    data[:, i, None] = collect_0_ref[i][1]
    
data
data[:,0]
collect0_df = px.DataFrame(data=data, columns=cols, index=range(np.shape(collect_0_ref[0][1])[0]))
collect0_df
collect0_df.head()
collect0_df.head().show()
get_ipython().magic(u'pinfo collect0_df')
collect0_df.head().values
collect0_df.to_csv(path="collect0_df.csv", cols=True, index=False, sep=",")
collect0_df.to_csv("collect0_df.csv", cols=True, index=False, sep=",")
collect0_df
collect0_df.to_csv("collect0_df.csv", cols=True, index=False, sep=",")
get_ipython().magic(u'pinfo collect0_df.to_csv')
collect0_df.to_csv(path_or_buf="collect0_df.csv", index=False, sep=", header=True)