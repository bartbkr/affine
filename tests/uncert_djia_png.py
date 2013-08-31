import pandas as px
mthdata = px.read_csv("./data/macro_data.csv", na_values="M",
                        index_col = 0, parse_dates=True, sep=";")
newguy = mthdata[['uncert', 'djialog_std']]
var_dates = px.date_range("5/1/1990", "5/1/2012", freq="MS").to_pydatetime()
newguy = newguy.ix[var_dates]
newguy.plot(secondary_y='djialog_std')
plt.show()
