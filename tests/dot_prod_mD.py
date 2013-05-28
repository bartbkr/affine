# coding: utf-8
a = range(9)
a
len(a)
a = np.reshape(a, (3,-1))
import numpy as np
a = np.reshape(a, (3,-1))
a
b = np.reshape(range(150), (3,-1,50))
b
b.shape()
np.shape(b)
np.dot(a,b)
get_ipython().magic(u'pinfo np.multiply')
np.shape(a)
np.shape(b)
b = np.reshape(range(150), (50, 3,-1))
b
np.shape(b)
np.dot(a,b)
a
b
b[0]
np.dot(a,b[0])
c = np.dot(a,b)
np.shape(c)
c[:,0,:]