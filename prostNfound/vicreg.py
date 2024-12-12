# Some of this code was adapted from the solo-learn implementation: 
# https://github.com/vturrisi/solo-learn/blob/main/solo/methods/vicreg.py
# Licencing declaration as follows: 

# ================================
# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# ================================

from torch import nn
import torch
from torch import distributed as dist
from torch.nn import functional as F


class GatherLayer(torch.autograd.Function):
    """
    Gathers tensors from all process and supports backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized():
            output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(output, x)
        else:
            output = [x]
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        if dist.is_available() and dist.is_initialized():
            all_gradients = torch.stack(grads)
            dist.all_reduce(all_gradients)
            grad_out = all_gradients[dist.get_rank()]
        else:
            grad_out = grads[0]
        return grad_out


def gather(X, dim=0):
    """Gathers tensors from all processes, supporting backward propagation."""
    return torch.cat(GatherLayer.apply(X), dim=dim)



class MLP(torch.nn.Module):
    def __init__(self, *inner_dims, dropout=0, use_batchnorm=False):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(inner_dims) - 1):
            self.layers.append(
                torch.nn.Linear(
                    in_features=inner_dims[i], out_features=inner_dims[i + 1]
                )
            )
            if use_batchnorm:
                self.layers.append(torch.nn.BatchNorm1d(inner_dims[i + 1]))
            self.layers.append(torch.nn.Dropout(p=dropout))
            self.layers.append(torch.nn.ReLU())

        self.dropout = dropout

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def variance_loss(
    z1: torch.Tensor, z2: torch.Tensor, gamma: float = 1.0
) -> torch.Tensor:
    """Computes variance loss given batch of projected features z1 from view 1 and projected
    features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: variance regularization loss.
    """

    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(gamma - std_z1)) + torch.mean(F.relu(gamma - std_z2))
    return std_loss


def covariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes covariance loss given batch of projected features z1 from view 1 and projected
    features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: covariance regularization loss.
    """

    N, D = z1.size()

    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)

    diag = torch.eye(D, device=z1.device)
    cov_loss = (
        cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D
    )
    return cov_loss


def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes mse loss given batch of projected features z1 from view 1 and projected features z2
    from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.

    Returns:
        torch.Tensor: invariance loss (mean squared error).
    """

    return F.mse_loss(z1, z2)


def vicreg_loss_func(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_loss_weight: float = 25.0,
    var_loss_weight: float = 25.0,
    cov_loss_weight: float = 1.0,
    gamma_param: float = 1.0,
):
    """Computes VICReg's loss given batch of projected features z1 from view 1 and projected
    features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        sim_loss_weight (float): invariance loss weight.
        var_loss_weight (float): variance loss weight.
        cov_loss_weight (float): covariance loss weight.

    Returns:
        torch.Tensor: VICReg loss.
    """

    sim_loss = invariance_loss(z1, z2)

    # vicreg's official code gathers the tensors here
    # https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py
    z1, z2 = gather(z1), gather(z2)

    var_loss = variance_loss(z1, z2, gamma_param)
    cov_loss = covariance_loss(z1, z2)

    loss = (
        sim_loss_weight * sim_loss
        + var_loss_weight * var_loss
        + cov_loss_weight * cov_loss
    )

    return loss


class VICRegLoss(nn.Module):
    def __init__(
        self,
        sim_loss_weight: float = 25.0,
        var_loss_weight: float = 25.0,
        cov_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        return vicreg_loss_func(
            z1,
            z2,
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )


class VICReg(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        proj_dims: list,
        features_dim: int,
        var_loss_weight: float = 25.0,
        cov_loss_weight: float = 1.0,
        inv_loss_weight: float = 25.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.projector = MLP(*[features_dim, *proj_dims])
        self.features_dim = features_dim
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight
        self.inv_loss_weight = inv_loss_weight

    def forward(self, X1, X2):
        X1 = self.backbone(X1)
        X2 = self.backbone(X2)

        X1 = self.projector(X1)
        X2 = self.projector(X2)

        loss = vicreg_loss_func(
            X1,
            X2,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
            sim_loss_weight=self.inv_loss_weight,
        )
        return loss
