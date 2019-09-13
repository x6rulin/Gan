import os
import torchvision

from local_lib.miscs import ArgParse, Trainer


class Args(ArgParse):

    def __init__(self):
        super(Args, self).__init__(description="Generative Adversarial Networks.")

        self.parser.add_argument("--nc", type=int, default=1, help="number of critic training for every mini-batch data")
        self.parser.add_argument("--ng", type=int, default=1, help="number of generator training for every mini-batch data")
        self.parser.add_argument("--img-dir", type=str, default='images', help="directory saving images generated")


class GanTrain(Trainer):

    def __init__(self, sample_num, train_dataset):
        self.sample_num = sample_num

        super(GanTrain, self).__init__(train_dataset, args=Args())
        self.net = {}
        self.optimizer = {}
        self.criterion = None

        self.epoch = 0
        self.value = 0.

        if not os.path.exists(self.args.img_dir):
            os.makedirs(self.args.img_dir, 0o775)

    def train(self):
        print(f"epochs: {self.epoch}")
        _ic, _pf= 1, 1
        for i, real_img in enumerate(self.train_loader, 1):
            critic_loss, real_score = self._critic(real_img)

            if _ic < self.args.nc:
                _ic += 1
            else:
                _ic = 1
                for _ in range(self.args.ng):
                    generator_loss, fake_score, fake_img = self._generator()

                if _pf < self.args.print_freq:
                    _pf += 1
                else:
                    _pf = 1
                    print(f"[epoch: {self.epoch} - {i}/{len(self.train_loader)}]Loss_dnet: {critic_loss:.6f} - "
                          f"Loss_gnet: {generator_loss:.6f} | Score_real: {real_score:.4f} - Score_fake: {fake_score:.4f}")

                    fake_img = fake_img.cpu().data
                    torchvision.utils.save_image(real_img, os.path.join(self.args.img_dir, f"real_sample_{i}.png"),
                                                 nrow=round(pow(self.args.batch_size, 0.5)), normalize=True, scale_each=True)
                    torchvision.utils.save_image(fake_img, os.path.join(self.args.img_dir, f"fake_sample_{i}.png"),
                                                 nrow=round(pow(self.args.batch_size, 0.5)), normalize=True, scale_each=True)

    def validate(self):
        return self.value

    def _critic(self, real_img):
        raise NotImplementedError

    def _generator(self):
        raise NotImplementedError

    @staticmethod
    def _no_grad(net):
        for param in net.parameters():
            if not param.requires_grad:
                param.requires_grad = False
            else:
                break

    @staticmethod
    def _grad_enable(net):
        for param in net.parameters():
            if param.requires_grad:
                break
            else:
                param.requires_grad = True
