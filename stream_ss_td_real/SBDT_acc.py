import math
import os
import time

import data_loader
import numpy as np
import torch
import torch.nn.functional as F
from SBDT_net import PBP_net

np.random.seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load acc

ndims = [3000, 150, 30000]
ind, y = data_loader.load_acc_train(fold_path="./data_real/acc/ibm-large-tensor.txt")
ind_test, y_test = data_loader.load_acc_test_long()

ind = torch.from_numpy(ind).float().to(device=device)
y = torch.from_numpy(y).float().to(device=device)
X_test = torch.from_numpy(ind_test - 1).float().to(device=device)
y_test = torch.from_numpy(y_test).float().to(device=device)

print("loaded")

n_hidden_units = 50
n_epochs = 1
n_stream_batch = 1

# mini_batch_list = [64,128,512]
# R_list = [8]

mini_batch_list = [256]  # [64,128,512]
R_list = [3, 5, 8, 10]
avg_num = 1
dir = "./new_result"
if not os.path.exists(dir):
    os.makedirs(dir)
mode = "minibatch"  #'single' #'minibatch'

for mini_batch in mini_batch_list:
    for R in R_list:
        help_str = "acc_" + str(mini_batch) + "_" + str(R)
        mse_list = np.zeros(avg_num)
        set_start = time.time()
        time_list = np.zeros(avg_num)
        for i in range(avg_num):
            fold_start = time.time()
            # shuffel
            X_train = ind
            y_train = y
            perm = torch.randperm(X_train.size(0))
            # perm = perm[:int(0.001*perm.size)]
            X_train = X_train[perm]
            y_train = y_train[perm]

            net = PBP_net.PBP_net(
                X_train,
                y_train,
                [n_hidden_units, n_hidden_units],
                normalize=True,
                n_epochs=n_epochs,
                R=R,
                ndims=ndims,
                n_stream_batch=n_stream_batch,
                mode=mode,
                mini_batch=mini_batch,
            )

            running_performance = np.array(net.pbp_train(X_test, y_test, help_str))
            file_name = "running_result/%s.txt" % (help_str)
            np.savetxt(file_name, np.c_[running_performance])
            print("\n  saved!\n")

            total_turn = float(X_train.shape[0]) / mini_batch
            running_time = (time.time() - fold_start) * total_turn / 100
            time_list[i] = running_time

            with torch.no_grad():
                m, a, b = net.predict_deterministic(X_test)
                # We compute the test MSE
                mse = F.mse_loss(m, y_test).item()
                print("mse = %f" % (mse))
                print("a, b, mean(tau), var(tau)")
                print(
                    "take %g seconds to finish fold %d" % (time.time() - fold_start, i)
                )
                print(a, b, a / b, a / (b**2))
                mse_list[i] = mse

        print(
            "\navg of mse: %.6g , std of mse is %.6g"
            % (mse_list.mean(), mse_list.std())
        )
        print("\n take %g seconds to finish the setting" % (time.time() - set_start))

        file_name = "acc_result_v1.txt"
        file = os.path.join(dir, file_name)
        f = open(file, "a+")
        f.write("R = %d, mini_batch =%s " % (R, mini_batch))
        f.write(
            "\navg of mse: %.6g , std of mse is %.6g"
            % (mse_list.mean(), mse_list.std())
        )
        f.write("\nthe exact value is %s" % str(mse_list))
        f.write("\nmean(tau): %.5g, var(tau): %.5g" % (a / b, a / (b**2)))
        final_time = time.time() - set_start
        f.write(
            "\ntake %.4g seconds to finish the setting, avg time is %.4g "
            % (final_time, final_time / avg_num)
        )
        f.write("\n\n")
        f.close()
