# coding: utf-8
import numpy as np
import pandas as pa
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick, candlestick2

data = pa.read_csv("macro_data.csv", sep=";", parse_dates=True, index_col=0, na_values='M')
ranges = data[['gnp_gdp_top10', 'gnp_gdp_bot10', 'gnp_gdpdefl_top10', 'gnp_gdpdefl_bot10']]
ranges = ranges.dropna()
ranges['month'] = ranges.index.month
ranges['disag_output'] = ranges['gnp_gdp_top10'] - ranges['gnp_gdp_bot10']
ranges['disag_infl'] = ranges['gnp_gdpdefl_top10'] - ranges['gnp_gdpdefl_bot10']

gdp_monthly = ranges.groupby('month')['disag_output'].agg(
    {'max': np.max,
     '3rd quart': lambda x: np.percentile(x, 75),
     '1st quart': lambda x: np.percentile(x, 25),
     'min': np.min,
     'mean': np.mean})
infl_monthly = ranges.groupby('month')['disag_infl'].agg(
    {'max': np.max,
     '3rd quart': lambda x: np.percentile(x, 75),
     '1st quart': lambda x: np.percentile(x, 25),
     'min': np.min})

plt.xlim([0, 12])
fig, ax = plt.subplots()

candlestick2(ax, gdp_monthly['3rd quart'], gdp_monthly['1st quart'],
            gdp_monthly['max'], gdp_monthly['min'], width=0.5)
ax.set_xticks(np.arange(0, len(gdp_monthly)))
ax.set_xticklabels(gdp_monthly.index)
ax.set_title('GDP Monthly Disagreement')

plt.show()
