import math
from re import I

import torch
import torch.distributions as dist
import torch.nn as nn


class Prior:
    def __init__(self, layer_sizes, var_targets, R, ndims, device):
        self.R = R
        self.ndims = ndims

        n_samples = 3.0
        v_observed = 1.0
        self.a_w = 2.0 * n_samples
        self.b_w = 2.0 * n_samples * v_observed

        self.rho_0 = torch.tensor(0.5)
        self.tau_0 = torch.tensor(1.0)

        self.a_u = 2.0 * n_samples
        self.b_u = 2.0 * n_samples * v_observed

        n_samples = 3.0
        a_sigma = 2.0 * n_samples
        b_sigma = 2.0 * n_samples * var_targets

        self.a_sigma_hat_nat = a_sigma - 1
        self.b_sigma_hat_nat = -b_sigma

        self.m_sigma = torch.tensor(0.0)
        self.v_sigma = torch.tensor(1.0)

        self.rnd_m_w = nn.ParameterList()
        self.m_w_hat_nat = nn.ParameterList()
        self.v_w_hat_nat = nn.ParameterList()
        self.rho_w_hat_nat = nn.ParameterList()

        for size_out, size_in in zip(layer_sizes[1:], layer_sizes[:-1]):
            self.rnd_m_w.append(
                nn.Parameter(
                    1.0
                    / math.sqrt(size_in + 1)
                    * torch.randn(
                        (size_out, size_in + 1), device=device, requires_grad=True
                    )
                )
            )
            self.m_w_hat_nat.append(
                nn.Parameter(
                    torch.zeros(
                        (size_out, size_in + 1), device=device, requires_grad=True
                    )
                )
            )
            self.v_w_hat_nat.append(
                nn.Parameter(
                    (self.a_w - 1)
                    / self.b_w
                    * torch.ones(
                        (size_out, size_in + 1), device=device, requires_grad=True
                    )
                )
            )
            self.rho_w_hat_nat.append(
                nn.Parameter(
                    torch.zeros(
                        (size_out, size_in + 1), device=device, requires_grad=True
                    )
                )
            )

        self.rnd_m_u = nn.ParameterList()
        self.m_u_hat_nat = nn.ParameterList()
        self.v_u_hat_nat = nn.ParameterList()

        # Add a core tensor like Tucker decomposition
        dim = len(ndims)
        self.rnd_m_core = nn.Parameter(
            1 / R * torch.randn((R**dim), device=device, requires_grad=True)
        )
        self.m_core_hat_nat = nn.Parameter(
            torch.zeros((R**dim), device=device, requires_grad=True)
        )
        self.v_core_hat_nat = nn.Parameter(
            (self.a_u - 1)
            / self.b_u
            * torch.ones((R**dim), device=device, requires_grad=True)
        )

        for i in range(dim):
            self.rnd_m_u.append(
                nn.Parameter(
                    1
                    / R
                    * torch.randn((ndims[i], R), device=device, requires_grad=True)
                )
            )
            self.m_u_hat_nat.append(
                nn.Parameter(
                    torch.zeros((ndims[i], R), device=device, requires_grad=True)
                )
            )
            self.v_u_hat_nat.append(
                nn.Parameter(
                    (self.a_u - 1)
                    / self.b_u
                    * torch.ones((ndims[i], R), device=device, requires_grad=True)
                )
            )

    def get_initial_params(self):
        m_w = [m for m in self.rnd_m_w]
        v_w = [1.0 / v for v in self.v_w_hat_nat]

        m_u = [m for m in self.rnd_m_u]
        v_u = [1.0 / v for v in self.v_u_hat_nat]

        return {
            "m_w": m_w,
            "v_w": v_w,
            "m_u": m_u,
            "v_u": v_u,
            "m_core": self.rnd_m_core,
            "v_core": 1.0 / self.v_core_hat_nat,
            "a": self.m_sigma,
            "b": self.v_sigma,
        }

    def gauss_pdf(self, x, mean, var):
        pdf = torch.exp(-0.5 * ((x - mean) ** 2) / var) / torch.sqrt(
            2 * math.pi * torch.abs(var)
        )
        return pdf

    def inverse_sigmoid(self, x):
        x = torch.clamp(x, min=1e-6, max=1 - 1e-6)  # 避免除零错误和log(0)
        return torch.log(x / (1 - x))

    def refine_prior(self, params):
        # for i in range(len(params["m_w"])):
        #     for j in range(params["m_w"][i].shape[0]):
        #         for k in range(params["m_w"][i].shape[1]):
        #             v_w_nat = 1.0 / params["v_w"][i][j, k]
        #             m_w_nat = params["m_w"][i][j, k] / params["v_w"][i][j, k]

        #             v_w_cav_nat = v_w_nat - self.v_w_hat_nat[i][j, k]
        #             m_w_cav_nat = m_w_nat - self.m_w_hat_nat[i][j, k]

        #             v_w_cav = 1.0 / v_w_cav_nat
        #             m_w_cav = m_w_cav_nat / v_w_cav_nat

        #             rho_star = torch.log(
        #                 self.gauss_pdf(m_w_cav, 0.0, v_w_cav + self.tau_0)
        #                 / self.gauss_pdf(m_w_cav, 0.0, v_w_cav)
        #             )

        #             v_w_til = 1 / (v_w_cav_nat + 1 / self.tau_0)
        #             m_w_til = v_w_til * (m_w_cav / v_w_cav)
        #             rho_til = rho_star + self.inverse_sigmoid(self.rho_0)

        #             m_w_new = torch.sigmoid(rho_til) * m_w_til
        #             v_w_new = torch.sigmoid(rho_til) * (
        #                 v_w_til + (1 - torch.sigmoid(rho_til)) * m_w_til**2
        #             )

        #             if (
        #                 v_w_cav > 0
        #                 and v_w_cav < 1e6
        #                 and v_w_new > 0
        #                 and v_w_new < 1e6
        #                 and ~torch.isnan(m_w_new)
        #                 and ~torch.isnan(v_w_new)
        #                 and ~torch.isinf(rho_star)
        #                 and ~torch.isinf(m_w_new)
        #             ):
        #                 v_w_new_nat = 1.0 / v_w_new
        #                 m_w_new_nat = m_w_new / v_w_new

        #                 self.m_w_hat_nat[i].data[j, k] = m_w_new_nat - m_w_cav_nat
        #                 self.v_w_hat_nat[i].data[j, k] = v_w_new_nat - v_w_cav_nat
        #                 self.rho_w_hat_nat[i].data[j, k] = rho_star

        #                 params["m_w"][i][j, k] = m_w_new
        #                 params["v_w"][i][j, k] = v_w_new

        # return params

        for i in range(len(params["m_w"])):
            v_w_nat = 1.0 / params["v_w"][i]
            m_w_nat = params["m_w"][i] / params["v_w"][i]

            v_w_cav_nat = v_w_nat - self.v_w_hat_nat[i]
            m_w_cav_nat = m_w_nat - self.m_w_hat_nat[i]

            v_w_cav = 1.0 / v_w_cav_nat
            m_w_cav = m_w_cav_nat / v_w_cav_nat

            rho_star = torch.log(
                self.gauss_pdf(m_w_cav, 0.0, v_w_cav + self.tau_0)
                / self.gauss_pdf(m_w_cav, 0.0, v_w_cav)
            )

            v_w_til = 1 / (v_w_cav_nat + 1 / self.tau_0)
            m_w_til = v_w_til * (m_w_cav / v_w_cav)
            rho_til = rho_star + self.inverse_sigmoid(self.rho_0)

            m_w_new = torch.sigmoid(rho_til) * m_w_til
            v_w_new = torch.sigmoid(rho_til) * (
                v_w_til + (1 - torch.sigmoid(rho_til)) * m_w_til**2
            )

            mask = (
                (v_w_cav > 0)
                & (v_w_cav < 1e6)
                & (v_w_new > 0)
                & (v_w_new < 1e6)
                & ~torch.isnan(m_w_new)
                & ~torch.isnan(v_w_new)
                & ~torch.isinf(rho_star)
                & ~torch.isinf(m_w_new)
            )

            v_w_new_nat = 1.0 / v_w_new
            m_w_new_nat = m_w_new / v_w_new

            self.m_w_hat_nat[i].data[mask] = m_w_new_nat[mask] - m_w_cav_nat[mask]
            self.v_w_hat_nat[i].data[mask] = v_w_new_nat[mask] - v_w_cav_nat[mask]
            self.rho_w_hat_nat[i].data[mask] = rho_star[mask]

            params["m_w"][i][mask] = m_w_new[mask]
            params["v_w"][i][mask] = v_w_new[mask]

        return params
