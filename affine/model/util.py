"""
These are utilies used by the affine model class
"""

import sys

import pandas as pa

from numpy.linalg import LinAlgError

def retry(func, attempts):
    """
    Decorator that attempts a function multiple times, even with exception
    """
    def inner_wrapper(*args, **kwargs):
        for attempt in xrange(attempts):
            try:
                return func(*args, **kwargs)
                break
            except LinAlgError:
                print "Trying again, maybe bad initial run"
                print "LinAlgError:", sys.exc_info()[0]
                continue
            except:
                print "Unexpected error:", sys.exc_info()[0]
                raise
    return inner_wrapper

def transform_var1(var_data, lags):
    """
    Returns Dataframe of time series into one column per variable per lag minus
    1
    """
    var_data_trans = var_data.copy()
    for lag in range(1, lags + 1):
        for var in var_data.columns:
            var_data_trans[str(var) + '_m' + str(lag)] = \
                pa.Series(var_data[var].values[:-(lag)],
                          index=var_data.index[lag:])
    return var_data_trans.dropna(axis=0)
