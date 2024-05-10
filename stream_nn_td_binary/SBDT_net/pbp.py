import torch
import torch.nn.functional as F

from .network import Network
from .prior import Prior


class PBP:
    def __init__(
        self, layer_sizes, mean_y_train, std_y_train, R, ndims, n_stream_batch, device
    ):
        var_targets = 1
        self.std_y_train = std_y_train
        self.mean_y_train = mean_y_train
        self.stream_batch = n_stream_batch
        self.R = R

        # We initialize the prior
        self.prior = Prior(layer_sizes, var_targets, R, ndims, device)

        # We create the network
        params = self.prior.get_initial_params()
        self.network = Network(
            params["m_w"],
            params["v_w"],
            params["m_u"],
            params["v_u"],
            params["a"],
            params["b"],
            n_stream_batch,
            device,
        )

    def do_pbp(self, X_train, y_train, n_iterations):
        if n_iterations > 0:
            # We first do a single pass
            self.do_first_pass(X_train, y_train)

            # We refine the prior
            with torch.no_grad():
                params = self.network.get_params()
                params = self.prior.refine_prior(params)
                self.network.set_params(params)

            # print("0")

            for i in range(int(n_iterations) - 1):
                # We do one more pass
                self.do_first_pass(X_train, y_train)

                # We refine the prior
                params = self.network.get_params()
                params = self.prior.refine_prior(params)
                self.network.set_params(params)

                # print(i + 1)

    def predict_deterministic(self, test_x):
        return self.network.output_deterministic(test_x).cpu()

    def get_deterministic_output(self, X_test):
        # output = []
        # for i in range(X_test.shape[0]):
        #     output.append(self.predict_deterministic(X_test[i, :]))
        output = [self.predict_deterministic(x) for x in X_test]
        params = self.network.get_params()
        return output, params["a"], params["b"]

    def do_first_pass(self, X, y):
        counter = 0
        while counter + self.stream_batch < X.shape[0]:
            old_params = self.network.get_params()
            self.adf_update(
                X[counter : counter + self.stream_batch, :],
                y[counter : counter + self.stream_batch].flatten(),
            )
            new_params = self.network.get_params()
            self.network.remove_invalid_updates(new_params, old_params)
            self.network.set_params(new_params)
            # if counter * self.stream_batch % 1000 == 0:
            #     print(".", end="")
            counter += self.stream_batch
        # print()

    def sample_w(self):
        self.network.sample_w()

    def adf_update(self, X, y):
        self.logZ, self.a_new, self.b_new = self.network.logZ_Z1_Z2(X, y)
        self.network.generate_updates(X, self.logZ, self.a_new, self.b_new)
