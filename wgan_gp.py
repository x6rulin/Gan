import sys
import os
import torch

rootPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(rootPath)

from core.loss import EMLoss
from core.train import GanTrain


class WGan2Train(GanTrain):

    def __init__(self, gnet, dnet, sample_num, train_dataset):
        super(WGan2Train, self).__init__(sample_num, train_dataset)

        self.net = {'gnet': gnet.to(self.device), 'dnet': dnet.to(self.device)}
        self.optimizer = {'gnet': torch.optim.Adam(self.net['gnet'].parameters(), lr=self.args.lr, betas=self.args.betas),
                          'dnet': torch.optim.Adam(self.net['dnet'].parameters(), lr=self.args.lr, betas=self.args.betas)}
        self.criterion = EMLoss()

    def _critic(self, real_img):
        self._grad_enable(self.net['dnet'])

        real_img = real_img.to(self.device)
        real_out = self.net['dnet'](real_img)

        with torch.no_grad():
            sample = torch.randn(real_img.size(0), self.sample_num, 1, 1, device=self.device)
            fake_img = self.net['gnet'](sample)
        fake_out = self.net['dnet'](fake_img)

        gradient_penalty = self._gradient_penalty(real_img, fake_img, _lambda=10)

        critic_loss = self.criterion(real_scores=real_out, fake_scores=fake_out) + gradient_penalty
        self.optimizer['dnet'].zero_grad()
        critic_loss.backward()
        self.optimizer['dnet'].step()

        return critic_loss, real_out.data.mean()

    def _generator(self):
        self._no_grad(self.net['dnet'])

        sample = torch.randn(self.args.batch_size, self.sample_num, 1, 1, device=self.device)
        fake_img = self.net['gnet'](sample)
        fake_out = self.net['dnet'](fake_img)

        generator_loss = self.criterion(fake_scores=fake_out)
        self.optimizer['gnet'].zero_grad()
        generator_loss.backward()
        self.optimizer['gnet'].step()

        return generator_loss, fake_out.data.mean(), fake_img

    def _gradient_penalty(self, real_data, fake_data, _lambda=10):
        batch_size = real_data.size(0)

        alpha = torch.rand(batch_size, 1, device=self.device)
        alpha = alpha.expand_as(real_data.reshape(batch_size, -1)).reshape(real_data.size())

        interpolates = fake_data + (real_data - fake_data) * alpha
        interpolates = torch.nn.Parameter(interpolates)

        disc_interpolates = self.net['dnet'](interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones_like(disc_interpolates, device=self.device),
                                        retain_graph=True, create_graph=True, only_inputs=True)[0]
        gradients = gradients.reshape(batch_size, -1)

        gradient_penalty = ((gradients.norm(2, dim=1) -1) ** 2).mean() * _lambda
        return gradient_penalty


if __name__ == "__main__":
    import torchvision
    from core.dataset import GanData
    from local_lib.dcgan import DCGAN_D, DCGAN_G

    img_dir = r"/home/data/Cartoon_faces/faces"
    isize, nc, nz = 96, 3, 128

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(isize),
        torchvision.transforms.CenterCrop(isize),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = GanData(img_dir, transform=transform)

    _gnet = DCGAN_G(isize, nz, nc, ngf=64, extra_layers=0, activation='prelu')
    _dnet = DCGAN_D(isize, nc, ndf=64, extra_layers=1, activation='prelu', norm='LayerNorm')
    trainer = WGan2Train(_gnet, _dnet, nz, dataset)
    trainer()
