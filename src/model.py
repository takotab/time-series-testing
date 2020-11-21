import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl

import data


def _get_norm(prefix, nf, ndim=2, zero=False, **kwargs):
    "Norm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    assert 1 <= ndim <= 3
    bn = getattr(nn, f"{prefix}{ndim}d")(nf, **kwargs)
    if bn.affine:
        bn.bias.data.fill_(1e-3)
        bn.weight.data.fill_(0.0 if zero else 1.0)
    return bn


def BatchNorm(nf, ndim=2, **kwargs):
    "BatchNorm layer with `nf` features and `ndim` initialized depending on `norm_type`."
    return _get_norm("BatchNorm", nf, ndim, zero=True, **kwargs)


class LinBnDrop(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"

    def __init__(self, n_in, n_out, bn=True, p=0.0, act=None, lin_first=False):
        layers = [BatchNorm(n_out if lin_first else n_in, ndim=1)] if bn else []
        if p != 0:
            layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)]
        if act is not None:
            lin.append(act)
        layers = lin + layers if lin_first else layers + lin
        super().__init__(*layers)


def store_parameter(data, trainable = False):
    return nn.Parameter(Tensor(data), trainable)   

class LitAutoEncoder(pl.LightningModule):
    def __init__(self, u_in, y_range):
        super().__init__()
        self.y_range = store_parameter(y_range)
        assert self.y_range.shape == (u_in, 2)
        self.lin_scale = LinBnDrop(u_in, u_in)

    def forward(self, x, return_scalers=False):
        scale = self.lin_scale(x)
        r, y_range = self.apply_range(x, scale)
        if return_scalers:
            return r, y_range
        return r

    def apply_range(self, x, scale):
        scale = torch.sigmoid(scale) + 0.5  # scaling the y_range with .5 to 1.5
        y_range = self.y_range[:, 0] * scale, self.y_range[:, 1] * scale

        r = (y_range[1] - y_range[0]) * torch.sigmoid(x) + y_range[0]
        return y_range

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.forward(x)
        
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)


# data
train_loader = DataLoader(data.DataScale(), batch_size=32)
val_loader = DataLoader(data.DataScale(), batch_size=32)

# model
model = LitAutoEncoder(3, [[-10,-9], [-10,-5], [10,15]])

# training
trainer = pl.Trainer(gpus=1)
trainer.fit(model, train_loader, val_loader)
