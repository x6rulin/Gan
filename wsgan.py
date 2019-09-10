import sys
import os
import torch

rootPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(rootPath)

from core.loss import WSLoss
from core.train import GanTrain


class WSGanTrain(GanTrain):

    def __init__(self, gnet, dnet, sample_num, train_dataset):
        super(WSGanTrain, self).__init__(sample_num, train_dataset)

        self.net = {'gnet': gnet.to(self.device), 'dnet': dnet.to(self.device)}
        self.optimizer = {'gnet': torch.optim.RMSprop(self.net['gnet'].parameters(), lr=5e-5),
                          'dnet': torch.optim.RMSprop(self.net['dnet'].parameters(), lr=5e-5)}
        self.criterion = WSLoss()

    def _critic(self, real_img):
        self._no_grad(self.net['gnet'])
        self._grad_enable(self.net['dnet'])

        real_img = real_img.to(self.device)
        real_out = self.net['dnet'](real_img)

        sample = torch.randn(real_img.size(0), self.sample_num, 1, 1, device=self.device)
        fake_img = self.net['gnet'](sample)
        fake_out = self.net['dnet'](fake_img)

        critic_loss = self.criterion(real_scores=real_out, fake_scores=fake_out)
        self.optimizer['dnet'].zero_grad()
        critic_loss.backward()
        self.optimizer['dnet'].step()

        for param in self.net['dnet'].parameters():
            param.data.clamp_(-0.01, 0.01)

        return critic_loss, real_out.data.mean()

    def _generator(self):
        self._grad_enable(self.net['gnet'])
        self._no_grad(self.net['dnet'])

        sample = torch.randn(self.args.batch_size, self.sample_num, 1, 1, device=self.device)
        fake_img = self.net['gnet'](sample)
        fake_out = self.net['dnet'](fake_img)

        generator_loss = self.criterion(fake_scores=fake_out)
        self.optimizer['gnet'].zero_grad()
        generator_loss.backward()
        self.optimizer['gnet'].step()

        return generator_loss, fake_out.data.mean(), fake_img


if __name__ == "__main__":
    import torchvision
    from core.dataset import GanData
    from local_lib.dcgan import Discriminator, Generator

    img_dir = r"/home/data/Cartoon_faces/faces"

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = GanData(img_dir, transform=transform)

    _n = 128
    _gnet = Generator(_n)
    _dnet = Discriminator()
    trainer = WSGanTrain(_gnet, _dnet, _n, dataset)
    trainer()
