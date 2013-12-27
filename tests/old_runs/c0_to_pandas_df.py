# coding: utf-8
import pickle

def to_csv(pkl_f)
"""
Create csv from pickle file results
"""
    pkl_file = open(pkl_f)
    pkl_f_imp = pickle.load(pkl_file)
    cols = []
    for x in pkl_f_imp:
        cols.append(x[0])
    data = np.zeros((20, len(cols)))
    for i, x in enumerate(cols):
        data[:, i, None] = pkl_f_imp[i][1]
    collect_df = px.DataFrame(data=data, columns=cols, index=range(np.shape(collect_0_ref[0][1])[0]))
    collect_df.to_csv(path_or_buf=pkl_f + ".csv", index=False, sep=",", header=True)
