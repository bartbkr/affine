# coding: utf-8
import numpy as np
import pandas as pa
import matplotlib.pyplot as plt
from matplotlib.finance import candlestick, candlestick2

data = pa.read_csv("macro_data.csv", sep=";", parse_dates=True, index_col=0,
                   na_values='M')
ranges = data[['gnp_gdp_top10_wiyear', 'gnp_gdp_bot10_wiyear',
               'gnp_gdpdefl_top10_wiyear', 'gnp_gdpdefl_bot10_wiyear']]
ranges = ranges.dropna()
ranges['month'] = ranges.index.month
ranges['disag_output'] = ranges['gnp_gdp_top10_wiyear'] - \
                         ranges['gnp_gdp_bot10_wiyear']
ranges['disag_infl'] = ranges['gnp_gdpdefl_top10_wiyear'] - \
                       ranges['gnp_gdpdefl_bot10_wiyear']

odis_saxes = ranges['disag_output'].plot()
odis_saxes.set_ylabel("90% - 10% GDP forecast")
odis_saxes.axvspan(pa.Timestamp('1987-07-01 00:00:00', tz=None),
                   pa.Timestamp('1989-01-01 00:00:00', tz=None),
                   facecolor='r', alpha=0.5)
odis_saxes.axvspan(pa.Timestamp('1989-05-01 00:00:00', tz=None),
                   pa.Timestamp('1990-07-01 00:00:00', tz=None),
                   facecolor='r', alpha=0.5)
odis_saxes.axvspan(pa.Timestamp('1992-04-01 00:00:00', tz=None),
                   pa.Timestamp('1993-04-01 00:00:00', tz=None),
                   facecolor='r', alpha=0.5)
odis_saxes.axvspan(pa.Timestamp('1996-03-01 00:00:00', tz=None),
                   pa.Timestamp('1997-04-01 00:00:00', tz=None),
                   facecolor='r', alpha=0.5)
odis_saxes.axvspan(pa.Timestamp('1999-03-01 00:00:00', tz=None),
                   pa.Timestamp('2000-10-01 00:00:00', tz=None),
                   facecolor='r', alpha=0.5)
odis_saxes.axvspan(pa.Timestamp('2002-02-01 00:00:00', tz=None),
                   pa.Timestamp('2007-05-01 00:00:00', tz=None),
                   facecolor='r', alpha=0.5)

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

fig, ax = plt.subplots()

ax.set_xticks(np.arange(0, len(gdp_monthly)))
ax.set_xticklabels(gdp_monthly.index)
ax.set_xlabel("Month")
ax.set_ylabel("90% - 10% GDP forecast")
candlestick2(ax, gdp_monthly['3rd quart'], gdp_monthly['1st quart'],
            gdp_monthly['max'], gdp_monthly['min'], width=0.5)
ax.set_xlim([-1, 12])

plt.show()
