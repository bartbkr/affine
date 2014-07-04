# coding: utf-8
import pandas as pa
djia_daily = pa.read_csv('DJIA.csv')
djia_daily
get_ipython().set_next_input(u'djia_daily = pa.read_csv');get_ipython().magic(u'pinfo pa.read_csv')
get_ipython().set_next_input(u'djia_daily = pa.read_csv');get_ipython().magic(u'pinfo pa.read_csv')
djia_daily = pa.read_csv('./DJIA.csv', index_col='DATE', parse_dates=True)
djia_daily
djia_daily = pa.read_csv('./DJIA.csv', index_col='DATE', parse_dates=True, dtype={'VALUE':np.float64})
import numpy as np
djia_daily = pa.read_csv('./DJIA.csv', index_col='DATE', parse_dates=True, dtype={'VALUE':np.float64})
djia_daily = pa.read_csv('./DJIA.csv', index_col='DATE', parse_dates=True)
djia_daily['djia_close'] = djia_daily['VALUE'].astype('float64')
djia_daily
djia_daily.head()
djia_daily.tail()
djia_daily.head(50)
djia_daily.tail(50)
djia_daily = pa.read_csv('./DJIA.csv', index_col='DATE', parse_dates=True, na_values=['#N/A'])
djia_daily
djia_daily.groupby(lambda x: x.month).std()
djia_daily.groupby([lambda x: x.year, lambda x: x.month]).std()
djia_daily.groupby([lambda x: x.year, lambda x: x.month]).std().head()
djia_daily.groupby([lambda x: x.year, lambda x: x.month]).std().to_csv('./DJIA_mth_std.csv')
djia_mth_std = djia_daily.groupby([lambda x: x.year, lambda x: x.month]).std()
get_ipython().magic(u'pinfo djia_mth_std.rename')
djia_mth_std.rename({'VALUE': 'djia_std'}, inplace=True)
djia_mth_std
djia_mth_std.rename({'VALUE': 'djia_std'})
djia_mth_std.rename({"VALUE": 'djia_std'})
djia_mth_std.columns
djia_mth_std.rename({u"VALUE": 'djia_std'})
djia_mth_std.rename({u"VALUE": 'djia_std'}, inplace=True)
djia_mth_std
djia_mth_std.columns(['djia_std'])
djia_mth_std.columns = ['djia_std']
djia_mth_std
djia_mth_std.head()
djia_mth_std.to_csv("./DJIA_mth_std.csv")
