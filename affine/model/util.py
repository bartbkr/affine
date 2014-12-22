"""
These are utilies used by the affine model class
"""

import sys

import pandas as pa

from numpy.linalg import LinAlgError

def transform_var1(var_data, k_ar):
    """
    Returns Dataframe of time series into one column per variable per lag minus
    1
    """
    var_data_trans = var_data.copy()
    for lag in range(1, k_ar + 1):
        for var in var_data.columns:
            var_data_trans[str(var) + '_m' + str(lag)] = \
                pa.Series(var_data[var].values[:-(lag)],
                          index=var_data.index[lag:])
    return var_data_trans.dropna(axis=0)
