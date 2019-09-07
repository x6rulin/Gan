import os
import torch
import torchvision

from local_lib.miscs import ArgParse, Trainer


class Args(ArgParse):

    def __init__(self):
        super(Args, self).__init__(description="Generative Adversarial Networks.")

        self.parser.add_argument("--nc", type=int, default=1, help="number of critic training for every mini-batch data")
        self.parser.add_argument("--ng", type=int, default=1, help="number of generator training for every mini-batch data")
        self.parser.add_argument("--img-dir", type=str, default='images', help="directory saving images generated")


class Train(Trainer):

    def __init__(self, gnet, dnet, sample_num, train_dataset, val_dataset, criterion, wgan=False):
        self.sample_num = sample_num
        self.is_wgan = wgan

        super(Train, self).__init__(train_dataset, val_dataset, args=Args())
        self.net = {'gnet': gnet.to(self.device), 'dnet': dnet.to(self.device)}
        if wgan:
            self.optimizer = {'gnet': torch.optim.RMSprop(self.net['gnet'].parameters(), lr=5e-5),
                              'dnet': torch.optim.RMSprop(self.net['dnet'].parameters(), lr=5e-5)}
        else:
            self.optimizer = {'gnet': torch.optim.Adam(self.net['gnet'].parameters(), lr=2e-4, betas=(0.5, 0.99)),
                              'dnet': torch.optim.Adam(self.net['dnet'].parameters(), lr=2e-4, betas=(0.5, 0.99), weight_decay=1e-3)}
        self.criterion = criterion

        self.epoch = 0
        self.value = 0.

    def train(self):
        print(f"epochs: {self.epoch}")
        self.net['gnet'].train()
        self.net['dnet'].train()
        for i, real_img in enumerate(self.train_loader, 1):
            for _ in range(self.args.nc):
                if self.is_wgan:
                    pass
                    # critic_loss, real_score = self._wgan_critic(real_img)
                else:
                    critic_loss, real_score = self._dcgan_critic(real_img)

            for _ in range(self.args.ng):
                generator_loss, fake_score = self._generator()

            if i % self.args.print_freq == 0:
                print(f"[epoch: {self.epoch} - {i}/{len(self.train_loader)}]Loss_dnet: {critic_loss:.6f} - "
                      f"Loss_gnet: {generator_loss:.6f} | Score_real: {real_score:.4f} - Score_fake: {fake_score:.4f}")

    def validate(self):
        if not os.path.exists(self.args.img_dir):
            os.makedirs(self.args.img_dir, 0o775)

        self.net['gnet'].eval()
        self.net['dnet'].eval()
        with torch.no_grad():
            real_scores, fake_scores = [], []
            for i, real_img in enumerate(self.val_loader, 1):
                real_img = real_img.to(self.device)
                real_out = self.net['dnet'](real_img)

                sample = torch.randn(real_img.size(0), self.sample_num, 1, 1, device=self.device)
                fake_img = self.net['gnet'](sample)
                fake_out = self.net['dnet'](fake_img)

                critic_loss = self.criterion(real_scores=real_out, fake_scores=fake_out)

                real_scores.append(real_out.data)
                fake_scores.append(fake_out.data)

                real_img = real_img.cpu().data
                fake_img = fake_img.cpu().data
                torchvision.utils.save_image(real_img, os.path.join(self.args.img_dir, f"real_{self.epoch}_{i}.png"),
                                             nrow=round(pow(self.args.batch_size, 0.5)), normalize=True, scale_each=True)
                torchvision.utils.save_image(fake_img, os.path.join(self.args.img_dir, f"real_{self.epoch}_{i}.png"),
                                             nrow=round(pow(self.args.batch_size, 0.5)), normalize=True, scale_each=True)

            real_score = torch.cat(real_scores).mean()
            fake_score = torch.cat(fake_scores).mean()
            print(f"[epochs: {self.epoch}]Score_real: {real_score:.4f} - Score_fake: {fake_score:.4f}")

        return torch.reciprocal(critic_loss.clamp_min(1e-2)) + 2 * real_score + 8 * fake_score

    def _dcgan_critic(self, real_img):
        real_img = real_img.to(self.device)
        real_out = self.net['dnet'](real_img)

        sample = torch.randn(real_img.size(0), self.sample_num, 1, 1, device=self.device)
        fake_img = self.net['gnet'](sample)
        fake_out = self.net['dnet'](fake_img)

        critic_loss = self.criterion(real_scores=real_out, fake_scores=fake_out)
        self.optimizer['dnet'].zero_grad()
        critic_loss.backward()
        self.optimizer['dnet'].step()

        return critic_loss, real_out.data.mean()

    def _wgan_critic(self, real_img):
        pass

    def _generator(self):
        sample = torch.randn(self.args.batch_size, self.sample_num, 1, 1, device=self.device)
        fake_img = self.net['gnet'](sample)
        fake_out = self.net['dnet'](fake_img)

        generator_loss = self.criterion(fake_scores=fake_out)
        self.optimizer['gnet'].zero_grad()
        generator_loss.backward()
        self.optimizer['gnet'].step()

        return generator_loss, fake_out.data.mean()
