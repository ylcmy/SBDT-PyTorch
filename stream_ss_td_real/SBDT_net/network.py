import math
import torch
from torch.autograd import grad

from .embedding import Embedding
from .network_layer import NetworkLayer


class Network:
    def __init__(
        self, m_w_init, v_w_init, m_u_init, v_u_init, a_init, b_init, n_stream_batch
    ):
        self.n_stream_batch = n_stream_batch
        self.layers = []

        if len(m_w_init) > 1:
            for m_w, v_w in zip(m_w_init[:-1], v_w_init[:-1]):
                self.layers.append(NetworkLayer(m_w, v_w, True))

        self.layers.append(NetworkLayer(m_w_init[-1], v_w_init[-1], False))

        self.params_m_w = []
        self.params_v_w = []

        for layer in self.layers:
            self.params_m_w.append(layer.m_w)
            self.params_v_w.append(layer.v_w)

        self.params_embed = []

        if len(m_u_init) > 1:
            for m_u, v_u in zip(m_u_init, v_u_init):
                self.params_embed.append(Embedding(m_u, v_u))

        self.params_m_u = []
        self.params_v_u = []

        for embed in self.params_embed:
            self.params_m_u.append(embed.m_u)
            self.params_v_u.append(embed.v_u)

        self.a = torch.tensor([a_init], dtype=torch.float32, requires_grad=True)
        self.b = torch.tensor([b_init], dtype=torch.float32, requires_grad=True)

    def output_deterministic(self, x):
        x = self.get_embed(x).unsqueeze(-1)
        for layer in self.layers:
            x = layer.output_deterministic(x)
        return x[0]

    def output_probabilistic(self, m):
        v = torch.zeros_like(m)
        for layer in self.layers:
            m, v = layer.output_probabilistic(m, v)
        return m[0], v[0]

    def logZ_Z1_Z2(self, x, y):
        tau = self.a / self.b
        v = torch.tensor(0.0)
        f = self.output_deterministic(x[0])
        y = y[0]

        # for i in range(len(self.params_m_w)):
        #     prod = grad(f, self.params_m_w[i])[0] ** 2 * self.params_v_w[i]
        #     v = v + prod.sum()
        #     # self.params_m_w[i].grad.zero_()

        # for i in range(len(self.params_m_u)):
        #     prod = grad(f, self.params_m_u[i])[0] ** 2 * self.params_v_u[i]
        #     v = v + prod.sum()
        #     # self.params_m_u[i].grad.zero_()

        f.backward(retain_graph=True)
        for i in range(len(self.params_m_w)):
            prod = self.params_m_w[i].grad ** 2 * self.params_v_w[i]
            v += prod.sum()
            self.params_m_w[i].grad.zero_()
        for i in range(len(self.params_m_u)):
            prod = self.params_m_u[i].grad ** 2 * self.params_v_u[i]
            v += prod.sum()
            self.params_m_u[i].grad.zero_()

        v_final = v + 1.0 / tau
        # Gaussian PDF computation
        logZ = -0.5 * (torch.log(2 * math.pi * v_final) + (y - f) ** 2 / v_final)

        return logZ, self.a + 0.5, self.b + 0.5 * ((y - f) ** 2 + v)

    def generate_updates(self, x, logZ, a_star, b_star):
        logZ.backward()
        with torch.no_grad():
            for i in range(len(self.params_m_w)):
                grad_m_w = self.params_m_w[i].grad
                grad_v_w = self.params_v_w[i].grad

                self.params_m_w[i].add_(self.params_v_w[i] * grad_m_w)
                self.params_v_w[i].sub_(
                    self.params_v_w[i] ** 2 * (grad_m_w**2 - 2 * grad_v_w)
                )
                self.params_m_w[i].grad.zero_()
                self.params_v_w[i].grad.zero_()

            for i in range(len(self.params_m_u)):
                grad_m_u = self.params_m_u[i].grad
                grad_v_u = self.params_v_u[i].grad
                # print(grad_m_u, grad_v_u)

                self.params_m_u[i].add_(self.params_v_u[i] * grad_m_u)
                self.params_v_u[i].sub_(
                    self.params_v_u[i] ** 2 * (grad_m_u**2 - 2 * grad_v_u)
                )
                self.params_m_u[i].grad.zero_()
                self.params_v_u[i].grad.zero_()
                # print(grad_m_u, grad_v_u)
            self.a.copy_(a_star)
            self.b.copy_(b_star)

    def get_embed(self, x):
        embed_list = []
        for i in range(len(self.params_m_u)):
            indx = x[i]
            embed_list.append(self.params_m_u[i][int(indx.item())])
        return torch.cat(embed_list).flatten()

    def get_params(self):
        with torch.no_grad():
            m_w = [layer.m_w.clone() for layer in self.layers]
            v_w = [layer.v_w.clone() for layer in self.layers]
            m_u = [embed.m_u.clone() for embed in self.params_embed]
            v_u = [embed.v_u.clone() for embed in self.params_embed]

            return {
                "m_w": m_w,
                "v_w": v_w,
                "m_u": m_u,
                "v_u": v_u,
                "a": self.a.clone(),
                "b": self.b.clone(),
            }

    def set_params(self, params):
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                layer.m_w.copy_(params["m_w"][i])
                layer.v_w.copy_(params["v_w"][i])

            for i, embed in enumerate(self.params_embed):
                embed.m_u.copy_(params["m_u"][i])
                embed.v_u.copy_(params["v_u"][i])

            self.a.copy_(params["a"])
            self.b.copy_(params["b"])

    def remove_invalid_updates(self, new_params, old_params):
        m_w_new = new_params["m_w"]
        v_w_new = new_params["v_w"]
        m_w_old = old_params["m_w"]
        v_w_old = old_params["v_w"]

        a_old = old_params["a"]
        a_new = new_params["a"]
        b_old = old_params["b"]
        b_new = new_params["b"]

        m_u_new = new_params["m_u"]
        v_u_new = new_params["v_u"]
        m_u_old = old_params["m_u"]
        v_u_old = old_params["v_u"]

        for i in range(len(self.layers)):
            index1 = torch.where(v_w_new[i] <= 1e-100)
            index2 = torch.where(
                torch.logical_or(torch.isnan(m_w_new[i]), torch.isnan(v_w_new[i]))
            )

            if len(index1[0]) > 0:
                m_w_new[i][index1] = m_w_old[i][index1]
                v_w_new[i][index1] = v_w_old[i][index1]

            if len(index2[0]) > 0:
                m_w_new[i][index2] = m_w_old[i][index2]
                v_w_new[i][index2] = v_w_old[i][index2]

        if torch.isnan(a_new) or torch.isnan(b_new) or b_new <= 1e-100:
            new_params["a"] = a_old
            new_params["b"] = b_old

        for i in range(len(self.params_embed)):
            index1 = torch.where(v_u_new[i] <= 1e-100)
            index2 = torch.where(
                torch.logical_or(torch.isnan(m_u_new[i]), torch.isnan(v_u_new[i]))
            )

            if len(index1[0]) > 0:
                m_u_new[i][index1] = m_u_old[i][index1]
                v_u_new[i][index1] = v_u_old[i][index1]

            if len(index2[0]) > 0:
                m_u_new[i][index2] = m_u_old[i][index2]
                v_u_new[i][index2] = v_u_old[i][index2]
