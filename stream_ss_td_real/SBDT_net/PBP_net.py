import gzip
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad

from .pbp import PBP


class PBP_net:
    def __init__(
        self,
        X_train,
        y_train,
        n_hidden,
        n_epochs=40,
        normalize=False,
        R=3,
        ndims=[200, 100, 200],
        n_stream_batch=1,
        mini_batch=100,
        mode="single",
        device="cpu",
    ):
        self.R = R
        self.nmod = len(ndims)
        self.mean_y_train = torch.mean(y_train)
        self.std_y_train = torch.std(y_train)
        self.stream_batch = n_stream_batch
        self.mode = mode
        self.mini_batch = mini_batch
        self.y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train
        self.X_train = X_train
        self.n_epochs = n_epochs
        self.N_turns = self.X_train.shape[0] / self.mini_batch
        self.test_point = int(0.05 * self.X_train.shape[0] / self.mini_batch)

        n_units_per_layer = np.concatenate(([self.nmod * self.R], n_hidden, [1]))
        self.running_score = []

        self.pbp_instance = PBP(
            n_units_per_layer,
            self.mean_y_train,
            self.std_y_train,
            self.R,
            ndims,
            n_stream_batch,
            device,
        )

    def pbp_train(self, X_test, y_test, help_str=""):
        if self.mode == "single":
            self.pbp_instance.do_pbp(
                self.X_train, self.y_train_normalized, self.n_epochs
            )
        else:
            count = 0
            turn = 0
            mini_batch = self.mini_batch
            while count + mini_batch <= self.X_train.shape[0]:
                X_sub = self.X_train[count : count + mini_batch]
                y_sub = self.y_train_normalized[count : count + mini_batch]

                self.pbp_instance.do_pbp(X_sub, y_sub, self.n_epochs)

                count = count + mini_batch
                print("finish  %d / %d " % (count, self.X_train.shape[0]) + help_str)

                turn = turn + 1

                if turn % self.test_point == 0:
                    with torch.no_grad():
                        m, a, b = self.predict_deterministic(X_test)
                        auc = torch.sqrt(F.mse_loss(m, y_test)).item()
                        self.running_score.append(auc)
                        print(
                            "after %d th batch(%.3f), the score is %.4f"
                            % (turn, float(turn) / self.N_turns, auc)
                        )

            return self.running_score

    def re_train(self, X_train, y_train, n_epochs):
        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / np.full(
            X_train.shape, self.std_X_train
        )

        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train

        self.pbp_instance.do_pbp(X_train, y_train_normalized, n_epochs)

    def predict(self, X_test):
        X_test = np.array(X_test, ndmin=2)
        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / np.full(
            X_test.shape, self.std_X_train
        )
        m, v, v_noise = self.pbp_instance.get_predictive_mean_and_variance(X_test)
        return m, v, v_noise

    def predict_deterministic(self, X_test):
        o, var_m, var_v = self.pbp_instance.get_deterministic_output(X_test)
        return o, var_m, var_v

    def sample_weights(self):
        self.pbp_instance.sample_w()

    def save_to_file(self, filename):
        def save_object(obj, filename):
            result = pickle.dumps(obj)
            with gzip.GzipFile(filename, "wb") as dest:
                dest.write(result)
            dest.close()

        save_object(self, filename)


def load_PBP_net_from_file(filename):
    def load_object(filename):
        with gzip.GzipFile(filename, "rb") as source:
            result = source.read()
        ret = pickle.loads(result)
        source.close()
        return ret

    PBP_network = load_object(filename)
    return PBP_network
