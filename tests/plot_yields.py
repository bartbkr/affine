"""
Script for generating plots of fama-bliss
"""
import pandas as pa
import datetime as dt
import matplotlib.pyplot as plt

origdata = pa.read_csv("./data/yield_curve.csv", na_values = "M", index_col=0,
                       parse_dates=True, sep=";")[[
                           'trcr_y1', 'trcr_y2',
                           'trcr_y3', 'trcr_y5']]
origdata = origdata.rename(columns={'trcr_y1':'orig_one_year',
                                    'trcr_y2':'orig_two_year',
                                    'trcr_y3':'orig_three_year',
                                    'trcr_y5':'orig_five_year'})

fblissdata = pa.read_csv("./data/fama-bliss_formatted.csv", na_values = "M",
                         index_col=0, parse_dates=True, sep=",")[[
                         'TMYTM_1',
                         'TMYTM_2',
                         'TMYTM_3',
                         'TMYTM_5']]
fblissdata = fblissdata.rename(columns={'TMYTM_1':'fbliss_one_year',
                                        'TMYTM_2':'fbliss_two_year',
                                        'TMYTM_3':'fbliss_three_year',
                                        'TMYTM_5':'fbliss_five_year'})

fblissdata['year'] = fblissdata.index.year
fblissdata['month'] = fblissdata.index.month
fblissdata['day'] = 1
fblissdata['new_dt'] = fblissdata.apply(
    lambda row: dt.datetime(int(row['year']),
                            int(row['month']),
                            int(row['day'])), axis=1)
fblissdata.set_index('new_dt', inplace=True)

both = origdata.join(fblissdata)

#all time
plot_els = ['one', 'two', 'three', 'five']
for numb in plot_els:
    fig, ax = plt.subplots()
    plotter = both[['orig_' + numb + '_year',
                    'fbliss_' + numb + '_year']].dropna()
    ax.plot(plotter.index, plotter['orig_' + numb + '_year'], 'b--',
            label='Constant Maturity (FRED), ' + numb + ' year')
    ax.plot(plotter.index, plotter['fbliss_' + numb + '_year'], 'g-',
            label='Faba-Bliss Zero Coupon ' + numb + ' year')
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102))
    fig.savefig('./exp_figures/yields_mat' + numb + '_allavail.png')

#focused time
yc_dates = pa.date_range("8/1/1990", "5/1/2012", freq="MS").to_pydatetime()
for numb in plot_els:
    fig, ax = plt.subplots()
    plotter = both[['orig_' + numb + '_year',
                    'fbliss_' + numb + '_year']].reindex(index=yc_dates)
    ax.plot(plotter.index, plotter['orig_' + numb + '_year'], 'b--',
            label='Constant Maturity (FRED), ' + numb + ' year')
    ax.plot(plotter.index, plotter['fbliss_' + numb + '_year'], 'g-',
            label='Faba-Bliss Zero Coupon ' + numb + ' year')
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102))
    fig.savefig('./exp_figures/yields_mat' + numb + '_1990-2012.png')
