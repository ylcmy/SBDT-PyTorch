import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from SBDT_net import PBP_net

np.random.seed(1)
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

# load movielens 1m
ndims = [6040, 3706]
ind = np.loadtxt("./data_real/movielens_1m/mv_train_ind.txt").astype(int)
y = np.loadtxt("./data_real/movielens_1m/mv_train_y.txt").astype(int)
ind_test = np.loadtxt("./data_real/movielens_1m/mv_test_ind.txt").astype(int)
y_test = np.loadtxt("./data_real/movielens_1m/mv_test_y.txt").astype(int)

ind = torch.from_numpy(ind).float().to(device=device)
y = torch.from_numpy(y).float().to(device=device)
ind_test = torch.from_numpy(ind_test).float().to(device=device)
y_test = torch.from_numpy(y_test).float().to(device=device)

print("loaded")

n_hidden_units = 50
n_epochs = 1
n_stream_batch = 1


# mini_batch_list = [256]
# R_list = [3,5,8,10]

mini_batch_list = [256]
R_list = [3, 5, 8, 10]
avg_num = 1
dir = "./new_result"
if not os.path.exists(dir):
    os.makedirs(dir)

mode = "minibatch"  #'single' #'minibatch'

for mini_batch in mini_batch_list:
    for R in R_list:
        help_str = "mv_" + str(mini_batch) + "_" + str(R)
        rmse_list = np.zeros(avg_num)
        set_start = time.time()
        time_list = np.zeros(avg_num)
        for i in range(avg_num):
            fold_start = time.time()
            # shuffel
            X_train = ind
            y_train = y
            perm = torch.randperm(X_train.size(0))
            # perm = perm[:1000]
            X_train = X_train[perm]
            y_train = y_train[perm]
            X_test = ind_test
            y_test = y_test

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
                device=device,
            )

            running_performance = np.array(net.pbp_train(X_test, y_test, help_str))
            file_name = "running_result/%s.txt" % (help_str)
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            np.savetxt(file_name, np.c_[running_performance])
            print("\n  saved!\n")

            total_turn = float(X_train.shape[0]) / mini_batch
            running_time = (time.time() - fold_start) * total_turn / 100
            time_list[i] = running_time

            with torch.no_grad():
                m, a, b = net.predict_deterministic(X_test)
                # We compute the test RMSE
                rmse = torch.sqrt(F.mse_loss(m, y_test)).item()
                print("rmse = %f" % (rmse))
                print("a, b, mean(tau), var(tau)")
                print(
                    "take %g seconds to finish fold %d" % (time.time() - fold_start, i)
                )
                print(a, b, a / b, a / (b**2))

                rmse_list[i] = rmse

        print(
            "\navg of rmse: %.6g , std of rmse is %.6g"
            % (rmse_list.mean(), rmse_list.std())
        )
        print("\n take %g seconds to finish the setting" % (time.time() - set_start))

        file_name = "mv1m_result_v1.txt"
        file = os.path.join(dir, file_name)
        f = open(file, "a+")
        f.write("R = %d, mini_batch =%s " % (R, mini_batch))
        f.write(
            "\navg of rmse: %.6g , std of rmse is %.6g"
            % (rmse_list.mean(), rmse_list.std())
        )
        f.write("\nthe exact value is %s" % str(rmse_list))
        f.write("\nmean(tau): %.5g, var(tau): %.5g" % (a / b, a / (b**2)))
        final_time = time.time() - set_start
        f.write(
            "\ntake %.4g seconds to finish the setting, avg time is %.4g "
            % (final_time, final_time / avg_num)
        )
        f.write("\n\n")
        f.close()
