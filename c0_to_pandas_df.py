# coding: utf-8
import pickle
pkl_file = open("collect_0_ref.pkl")
collect_0_ref = pickle.load(pkl_file)
cols = []
for x in collect_0_ref:
    cols.append(x[0])
data = np.zeros((20, len(cols))
for i, x in enumerate(cols):
    data[:, i, None] = collect_0_ref[i][1]
collect0_df = px.DataFrame(data=data, columns=cols, index=range(np.shape(collect_0_ref[0][1])[0]))
collect0_df.to_csv(path_or_buf="collect0_df.csv", index=False, sep=", header=True)
