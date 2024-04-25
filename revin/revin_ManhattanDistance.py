import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """Reversible Instance Normalization for Accurate Time-Series Forecasting
           against Distribution Shift, ICLR2021.

    Parameters
    ----------
    num_features: int, the number of features or channels.
    eps: float, a value added for numerical stability, default 1e-5.
    affine: bool, if True(default), RevIN has learnable affine parameters.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError('Only modes norm and denorm are supported.')
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = [0]
        self.mean = torch.median(x)
        # print(self.mean)
        # 计算每个数据点与均值之间的曼哈顿距离
        distances = torch.sum(torch.abs(x - self.mean), dim=dim2reduce)
        # 距离的均值，即为方差的估计值
        # print(distances)
        variance = torch.mean(distances)
        # print(variance)
        self.stdev = variance + self.eps

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

if __name__ == '__main__':
    tensor_shape = (64, 6)
    x = torch.rand(tensor_shape)
    layer = RevIN(6)
    y = layer(x, mode='norm')
    z = layer(y, mode='denorm')

    print(x)
    print(y)
    print(z)