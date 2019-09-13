import torch


def _activate(key):
    if key == 'none': return []
    if key == 'relu': return [torch.nn.ReLU(inplace=True)]
    if key == 'leakyrelu': return [torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)]
    if key == 'prelu': return [torch.nn.PReLU()]
    if key == 'sigmoid': return [torch.nn.Sigmoid()]
    if key == 'tanh': return [torch.nn.Tanh()]

    raise RuntimeError(f"not supported activation: '{key}'")

def _normalization(key, **kwargs):
    if key == 'none': return []
    if key == 'BatchNorm2d': return [torch.nn.BatchNorm2d(**kwargs)]
    if key == 'LayerNorm': return [torch.nn.LayerNorm(**kwargs)]

    raise RuntimeError(f"not supported normaliztion: '{key}'")


def _weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.normal_(m.weight, 0., 0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
            torch.nn.init.normal_(m.weight, 1., 0.02)
            torch.nn.init.constant_(m.bias, 0)


class _ConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=1, \
                 transposed=False, output_padding=0, activation='none', norm='none', **kwargs):
        super(_ConvLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias) \
            if transposed else torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            *_normalization(norm, **kwargs),
            *_activate(activation),
        )

    def forward(self, x):
        return self.sub_module(x)


class DCGAN_D(torch.nn.Module):

    def __init__(self, isize, nc, ndf=64, extra_layers=0, activation='leakyrelu', norm='BatchNorm2d', init_weights=True):
        super(DCGAN_D, self).__init__()
        assert isize % 16 == 0, "iszie has to be a multiple of 16"

        self.sub_module = self._make_layers(isize, nc, ndf, extra_layers, activation, norm)

        if init_weights:
            self.apply(_weights_init)

    def forward(self, x):
        return self.sub_module(x)

    @staticmethod
    def _make_layers(isize, nc, ndf, extra_layers, activation='leakyrelu', norm='BatchNorm2d'):
        layers = []

        _m = [2, 3][isize % 3 == 0]
        csize, cndf = isize // _m, ndf
        kwargs = [dict(kernel_size=5, stride=3, padding=1, activation=activation, norm='none'),
                  dict(kernel_size=4, stride=2, padding=1, activation=activation, norm='none')]
        layers.append(_ConvLayer(nc, ndf, **kwargs[_m == 2]))

        kwargs = [dict(norm=norm, num_features=cndf),
                  dict(norm=norm, normalized_shape=(cndf, csize, csize))]
        layers.extend([_ConvLayer(cndf, cndf, 3, 1, 1,
                                  activation=activation, **kwargs[norm == 'LayerNorm'])
                       for _ in range(extra_layers)])

        while csize > 4:
            kwargs = [dict(norm=norm, num_features=cndf * 2),
                      dict(norm=norm, normalized_shape=(cndf * 2, csize//2, csize//2))]
            layers.append(_ConvLayer(cndf, cndf * 2, 4, 2, 1,
                                     activation=activation, **kwargs[norm == 'LayerNorm']))
            csize, cndf = csize // 2, cndf * 2

        layers.append(_ConvLayer(cndf, 1, 4, 1, 0, activation='none', norm='none'))

        return torch.nn.Sequential(*layers)


class DCGAN_G(torch.nn.Module):

    def __init__(self, isize, nz, nc, ngf=64, extra_layers=0, activation='relu', norm='BatchNorm2d', init_weights=True):
        super(DCGAN_G, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        self.sub_module = self._make_layers(isize, nz, nc, ngf, extra_layers, activation, norm)

        if init_weights:
            self.apply(_weights_init)

    def forward(self, x):
        return self.sub_module(x)

    @staticmethod
    def _make_layers(isize, nz, nc, ngf, extra_layers=0, activation='relu', norm='BatchNorm2d'):
        layers = []

        cngf, tisize = ngf, 4
        _m = [2, 3][isize % 3 == 0]
        while tisize != isize // _m:
            cngf, tisize = cngf * 2, tisize * 2

        csize = 4

        kwargs = [dict(norm=norm, num_features=cngf),
                  dict(norm=norm, normalized_shape=(cngf, csize, csize))]
        layers.append(_ConvLayer(nz, cngf, 4, 1, 0, transposed=True,
                                 activation=activation, **kwargs[norm == 'LayerNorm']))

        while csize < isize / _m:
            kwargs = [dict(norm=norm, num_features=cngf//2),
                      dict(norm=norm, normalized_shape=(cngf//2, csize * 2, csize * 2))]
            layers.append(_ConvLayer(cngf, cngf//2, 4, 2, 1, transposed=True,
                                     activation=activation, **kwargs[norm == 'LayerNorm']))
            cngf, csize = cngf // 2, csize * 2

        kwargs = [dict(norm=norm, num_features=cngf),
                  dict(norm=norm, normalized_shape=(cngf, csize, csize))]
        layers.extend([_ConvLayer(cngf, cngf, 3, 1, 1,
                                  activation=activation, **kwargs[norm == 'LayerNorm'])
                       for _ in range(extra_layers)])

        kwargs = [dict(kernel_size=5, stride=3, padding=1, activation='tanh', norm='none'),
                  dict(kernel_size=4, stride=2, padding=1, activation='tanh', norm='none')]
        layers.append(_ConvLayer(cngf, nc, transposed=True, **kwargs[_m == 2]))

        return torch.nn.Sequential(*layers)
