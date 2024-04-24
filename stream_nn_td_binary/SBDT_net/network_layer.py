import math
from turtle import forward

import torch
import torch.nn.functional as F
from torch import nn


# class NetworkLayer:
#     def __init__(self, m_w_init, v_w_init, non_linear=True, device="cpu"):
#         self.m_w = m_w_init.clone().detach()
#         self.v_w = v_w_init.clone().detach()
#         self.non_linear = non_linear
#         self.n_inputs = m_w_init.shape[1]
#         self.device = device

#     @staticmethod
#     def n_pdf(x):
#         return 1.0 / torch.sqrt(2 * torch.tensor(math.pi)) * torch.exp(-0.5 * x**2)

#     @staticmethod
#     def n_cdf(x):
#         return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

#     @staticmethod
#     def gamma(x):
#         return NetworkLayer.n_pdf(x) / NetworkLayer.n_cdf(-x)

#     @staticmethod
#     def beta(x):
#         return NetworkLayer.gamma(x) * (NetworkLayer.gamma(x) - x)

#     def output_probabilistic(self, m_w_previous, v_w_previous):
#         m_w_previous_with_bias = torch.cat([m_w_previous, torch.ones(1)], dim=0)
#         v_w_previous_with_bias = torch.cat([v_w_previous, torch.zeros(1)], dim=0)

#         m_linear = torch.dot(self.m_w, m_w_previous_with_bias) / math.sqrt(
#             self.n_inputs
#         )
#         v_linear = (
#             torch.dot(self.v_w, v_w_previous_with_bias)
#             + torch.dot(self.m_w**2, v_w_previous_with_bias)
#             + torch.dot(self.v_w, m_w_previous_with_bias**2)
#         ) / self.n_inputs

#         if self.non_linear:
#             alpha = m_linear / torch.sqrt(v_linear)
#             gamma = self.gamma(-alpha)
#             gamma_robust = -alpha - 1.0 / alpha + 2.0 / alpha**3
#             gamma_final = torch.where(alpha < -30, gamma, gamma_robust)

#             v_aux = m_linear + torch.sqrt(v_linear) * gamma_final

#             m_a = self.n_cdf(alpha) * v_aux
#             v_a = m_a * v_aux * self.n_cdf(-alpha) + self.n_cdf(alpha) * v_linear * (
#                 1 - gamma_final * (gamma_final + alpha)
#             )

#             return m_a, v_a
#         else:
#             return m_linear, v_linear

#     def output_deterministic(self, output_previous):
#         output_previous_with_bias = torch.cat(
#             [
#                 output_previous,
#                 torch.ones(1, output_previous.shape[1], device=self.device),
#             ],
#             dim=0,
#         ) / math.sqrt(self.n_inputs)

#         a = torch.mm(self.m_w, output_previous_with_bias)

#         if self.non_linear:
#             a = F.relu(a)
#             # Tanh
#             # a = torch.tanh(a)

#         return a


class NetworkLayer(nn.Module):
    def __init__(self, m_w_init, v_w_init, non_linear=True, device="cpu"):
        super(NetworkLayer, self).__init__()
        self.m_w = nn.Parameter(m_w_init.clone().detach())
        self.v_w = nn.Parameter(v_w_init.clone().detach())
        self.non_linear = non_linear
        self.n_inputs = m_w_init.shape[1]
        self.device = device

    def forward(self, x):
        output_previous_with_bias = torch.cat(
            [
                x,
                torch.ones(1, x.shape[1], device=self.device),
            ],
            dim=0,
        ) / math.sqrt(self.n_inputs)

        # a = torch.mm(self.m_w, output_previous_with_bias)
        a = self.m_w @ output_previous_with_bias

        if self.non_linear:
            a = F.relu(a)
            # Tanh
            # a = torch.tanh(a)

        return a
