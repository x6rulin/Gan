import sys
import os
import torch
import torchvision

rootPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(rootPath)

from local_lib.dcgan import Discriminator, Generator
from core.loss import LogDloss
from core.dataset import GanData
from core.train import GanTrain


class DcGanTrain(GanTrain):

    def __init__(self, gnet, dnet, sample_num, train_dataset):
        super(DcGanTrain, self).__init__(sample_num, train_dataset)

        self.net = {'gnet': gnet.to(self.device), 'dnet': dnet.to(self.device)}
        self.optimizer = {'gnet': torch.optim.Adam(self.net['gnet'].parameters(), lr=2e-4, betas=(0.5, 0.99)),
                          'dnet': torch.optim.Adam(self.net['dnet'].parameters(), lr=2e-4, betas=(0.5, 0.99), weight_decay=1e-3)}
        self.criterion = LogDloss()

    def _critic(self, real_img):
        self.net['gnet'].train()
        self.net['dnet'].train()
        real_img = real_img.to(self.device)
        real_out = self.net['dnet'](real_img)

        sample = torch.randn(real_img.size(0), self.sample_num, 1, 1, device=self.device)
        sample.clamp_(-3., 3.)
        fake_img = self.net['gnet'](sample)
        fake_out = self.net['dnet'](fake_img)

        critic_loss = self.criterion(real_scores=real_out, fake_scores=fake_out)
        self.optimizer['dnet'].zero_grad()
        critic_loss.backward()
        self.optimizer['dnet'].step()

        return critic_loss, real_out.data.mean()

    def _generator(self):
        self.net['gnet'].train()
        self.net['dnet'].train()
        sample = torch.randn(self.args.batch_size, self.sample_num, 1, 1, device=self.device)
        sample.clamp_(-3., 3.)
        fake_img = self.net['gnet'](sample)
        fake_out = self.net['dnet'](fake_img)

        generator_loss = self.criterion(fake_scores=fake_out)
        self.optimizer['gnet'].zero_grad()
        generator_loss.backward()
        self.optimizer['gnet'].step()

        return generator_loss, fake_out.data.mean(), fake_img


if __name__ == "__main__":

    img_dir = r"/home/data/Cartoon_faces/faces"

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = GanData(img_dir, transform=transform)

    _n = 128
    _gnet = Generator(_n)
    _dnet = Discriminator(out_activate='sigmoid')
    trainer = DcGanTrain(_gnet, _dnet, _n, dataset)
    trainer()
