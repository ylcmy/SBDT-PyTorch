import torch
import torch.nn as nn


class Embedding:

    def __init__(self, m_u_init, v_u_init):

        # self.m_u = nn.Parameter(torch.tensor(m_u_init, dtype=torch.float32))
        # self.v_u = nn.Parameter(torch.tensor(v_u_init, dtype=torch.float32))
        self.m_u = m_u_init.clone().detach().requires_grad_(True)
        self.v_u = v_u_init.clone().detach().requires_grad_(True)
