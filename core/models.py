import torch


def _activate(key, **kwargs):
    if key == 'none': return []
    if key == 'relu': return [torch.nn.ReLU(inplace=True)]
    if key == 'leakyrelu': return [torch.nn.LeakyReLU(inplace=True, **kwargs)]
    if key == 'prelu': return [torch.nn.PReLU(**kwargs)]
    if key == 'sigmoid': return [torch.nn.Sigmoid()]
    if key == 'tanh': return [torch.nn.Tanh()]

    raise RuntimeError(f"not supported activation: '{key}'")


def _init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight, mean=0., std=0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)


class _ConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, \
                 groups=1, transposed=False, output_padding=0, bn=True, activation='prelu', **kwargs):
        super(_ConvLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias) \
            if transposed else torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            *[torch.nn.BatchNorm2d(out_channels)][:bn],
            *_activate(activation, **kwargs),
        )

    def forward(self, x):
        return self.sub_module(x)


class _DLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bn=True, activation='prelu', **kwargs):
        super(_DLayer, self).__init__()

        self.sub_module = _ConvLayer(in_channels, out_channels, kernel_size, stride, padding, bn=bn, activation=activation, **kwargs)

    def forward(self, x):
        return self.sub_module(x)


class _GLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding=0, bn=True, activation='prelu', **kwargs):
        super(_GLayer, self).__init__()

        self.sub_module = _ConvLayer(in_channels, out_channels, kernel_size, stride, padding, \
                                     transposed=True, output_padding=output_padding, bn=bn, activation=activation, **kwargs)

    def forward(self, x):
        return self.sub_module(x)


class Discriminator(torch.nn.Module):

    def __init__(self, out_activate='none'):
        super(Discriminator, self).__init__()

        self.sub_module = torch.nn.Sequential(
            _DLayer(3, 64, 5, 3, 1, bn=False),
            _DLayer(64, 128, 4, 2, 1),
            _DLayer(128, 256, 4, 2, 1),
            _DLayer(256, 512, 4, 2, 1),
            _DLayer(512, 1, 4, 1, 0, bn=False, activation=out_activate),
        )

        for _m in self.modules():
            _init_weights(_m)

    def forward(self, x):
        return self.sub_module(x)


class Generator(torch.nn.Module):

    def __init__(self, sample_num):
        super(Generator, self).__init__()

        self.sub_module = torch.nn.Sequential(
            _GLayer(sample_num, 512, 4, 1, 0),
            _GLayer(512, 256, 4, 2, 1),
            _GLayer(256, 128, 4, 2, 1),
            _GLayer(128, 64, 4, 2, 1),
            _GLayer(64, 32, 5, 3, 1),
            _DLayer(32, 3, 3, 1, 1, bn=False, activation='tanh'),
        )

        for _m in self.modules():
            _init_weights(_m)

    def forward(self, x):
        return self.sub_module(x)
