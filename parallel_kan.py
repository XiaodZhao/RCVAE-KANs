import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random

def B_batch(x, grid, k=0, extend=True, device='cuda:0'):

    def extend_grid(grid, k_extend=0):
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
        for i in range(k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
        grid = grid.to(device)
        return grid

    if extend == True:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2).to(device)
    x = x.unsqueeze(dim=1).to(device)

    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False, device=device)
        value = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * B_km1[:, :-1] + (
                    grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * B_km1[:, 1:]
    return value


def coef2curve(x_eval, grid, coef, k, device="cuda:0"):
    if coef.dtype != x_eval.dtype:
        coef = coef.to(x_eval.dtype)
    y_eval = torch.einsum('ij,ijk->ik', coef, B_batch(x_eval, grid, k, device=device))
    return y_eval


def curve2coef(x_eval, y_eval, grid, k, device="cuda:0"):
    mat = B_batch(x_eval, grid, k, device=device).permute(0, 2, 1)
    coef = torch.linalg.lstsq(mat.to(device), y_eval.unsqueeze(dim=2).to(device),
                              driver='gelsy' if device == 'cpu' else 'gels').solution[:, :, 0]
    return coef.to(device)

class KANLayer(nn.Module):

    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, noise_scale=0.1, scale_base=1.0, scale_sp=1.0, base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, device='cuda:0'):
        super(KANLayer, self).__init__()

        self.size = size = out_dim * in_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k


        self.grid = torch.einsum('i,j->ij', torch.ones(size, device=device), torch.linspace(grid_range[0], grid_range[1], steps=num + 1, device=device))
        self.grid = torch.nn.Parameter(self.grid).requires_grad_(False)
        noises = (torch.rand(size, self.grid.shape[1]) - 1 / 2) * noise_scale / num
        noises = noises.to(device)

        self.coef = torch.nn.Parameter(curve2coef(self.grid, noises, self.grid, k, device))

        if isinstance(scale_base, float):
            self.scale_base = torch.nn.Parameter(torch.ones(size, device=device) * scale_base).requires_grad_(sb_trainable)
        else:
            self.scale_base = torch.nn.Parameter(torch.FloatTensor(scale_base).to(device)).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(size, device=device) * scale_sp).requires_grad_(sp_trainable)
        self.base_fun = base_fun

        self.mask = torch.nn.Parameter(torch.ones(size, device=device)).requires_grad_(False)
        self.grid_eps = grid_eps
        self.weight_sharing = torch.arange(size)
        self.lock_counter = 0
        self.lock_id = torch.zeros(size)
        self.device = device

    def forward(self, x):
        batch = x.shape[0]
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, device=self.device)).reshape(batch, self.size).permute(1, 0)
        preacts = x.permute(1, 0).clone().reshape(batch, self.out_dim, self.in_dim)
        base = self.base_fun(x).permute(1, 0)
        y = coef2curve(x_eval=x, grid=self.grid[self.weight_sharing], coef=self.coef[self.weight_sharing], k=self.k, device=self.device)
        y = y.permute(1, 0)
        postspline = y.clone().reshape(batch, self.out_dim, self.in_dim)
        y = self.scale_base.unsqueeze(dim=0) * base + self.scale_sp.unsqueeze(dim=0) * y
        y = self.mask[None, :] * y
        postacts = y.clone().reshape(batch, self.out_dim, self.in_dim)
        y = torch.sum(y.reshape(batch, self.out_dim, self.in_dim), dim=2)


        return y, preacts, postacts, postspline

class KAN(nn.Module):

    def __init__(self, width=None, grid=3, k=3, noise_scale=0.1, noise_scale_base=0.1, base_fun=torch.nn.SiLU(), bias_trainable=True, grid_eps=1.0, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, device='cuda:0', seed=0):
        super(KAN, self).__init__()


        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.biases = []
        self.act_fun = []
        self.depth = len(width) - 1
        self.width = width
        for l in range(self.depth):
            scale_base = 1 / np.sqrt(width[l]) + (torch.randn(width[l] * width[l + 1], ) * 2 - 1) * noise_scale_base
            sp_batch = KANLayer(in_dim=width[l], out_dim=width[l + 1], num=grid, k=k, noise_scale=noise_scale, scale_base=scale_base, scale_sp=1., base_fun=base_fun, grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable,
                                sb_trainable=sb_trainable, device=device)
            self.act_fun.append(sp_batch)

            bias = nn.Linear(width[l + 1], 1, bias=False, device=device).requires_grad_(bias_trainable)
            bias.weight.data *= 0.
            self.biases.append(bias)

        self.biases = nn.ModuleList(self.biases)
        self.act_fun = nn.ModuleList(self.act_fun)

        self.grid = grid
        self.k = k
        self.base_fun = base_fun

        self.device = device

    def forward(self, x):
        self.acts = []
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_std = []
        self.acts.append(x)

        for l in range(self.depth):

            x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x)
            x = x_numerical
            postacts = postacts_numerical

            grid_reshape = self.act_fun[l].grid.reshape(self.width[l + 1], self.width[l], -1)
            input_range = grid_reshape[:, :, -1] - grid_reshape[:, :, 0] + 1e-4
            output_range = torch.mean(torch.abs(postacts), dim=0)
            self.acts_scale.append(output_range / input_range)
            self.acts_scale_std.append(torch.std(postacts, dim=0))
            self.spline_preacts.append(preacts.detach())
            self.spline_postacts.append(postacts.detach())
            self.spline_postsplines.append(postspline.detach())

            x = x + self.biases[l].weight
            self.acts.append(x)

        return x
