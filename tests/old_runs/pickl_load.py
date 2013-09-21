import os
import pickle

orig_dir = os.getcwd()
loc = "../results/bern_ls"
os.chdir(loc)
files = os.listdir(os.getcwd())

for file in files:
    file_open = open(file, "rb")
    data = pickle.load(file_open)
    print file
    print data
    file_open.close()

os.chdir(orig_dir)
