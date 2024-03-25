import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_dblp_train(fold_path="../data_binary/dblp/dblp-large-tensor.txt"):
    y = []
    ind = []
    with open(fold_path, "r") as f:
        for line in f:
            items = line.strip().split("\t")
            y.append(float(items[-1]))
            ind.append([float(idx) for idx in items[0:-1]])
        ind = np.array(ind)
        y = np.array(y)
        # y[y>10]=10
        # y = y/17.427057
    return ind.astype(int), y


def load_dblp_test(fold_path="./data_binary/dblp/dblp.mat"):
    test_data = loadmat(fold_path)["data"]["test"][0, 0]
    ind_test = []
    y_test = []
    for i in range(50):
        y_test.append(test_data[0, i]["Ymiss"].flatten()[0])
        ind_test.append(test_data[0, i]["subs"].flatten()[0])
    ind_test = np.array(ind_test).reshape(-1, 3)
    y_test = np.array(y_test).flatten()
    # print(ind_test.shape)
    # print(y_test.shape)

    return ind_test.astype(int), y_test
