"""
Script for generating plots of fama-bliss
"""
import pandas as pa
import datetime as dt
import matplotlib.pyplot as plt

one_year_orig = pa.read_csv('./ss_results/one_year_res_orig.csv',
                             parse_dates=True, index_col=0)

two_year_orig = pa.read_csv('./ss_results/two_year_res_orig.csv',
                             parse_dates=True, index_col=0)

three_year_orig = pa.read_csv('./ss_results/five_year_res_orig.csv',
                             parse_dates=True, index_col=0)

five_year_orig = pa.read_csv('./ss_results/five_year_res_orig.csv',
                             parse_dates=True, index_col=0)







"""











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
"""
