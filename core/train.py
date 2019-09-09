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
        self.criterion = NotImplemented

        self.epoch = 0
        self.value = 0.

        self.__nc = self.args.nc
        self.__pf = self.args.print_freq
        if not os.path.exists(self.args.img_dir):
            os.makedirs(self.args.img_dir, 0o775)

    def train(self):
        print(f"epochs: {self.epoch}")
        for i, real_img in enumerate(self.train_loader, 1):
            critic_loss, real_score = self._critic(real_img)

            if self.__nc > 1:
                self.__nc -= 1
            else:
                for _ in range(self.args.ng):
                    generator_loss, fake_score, fake_img = self._generator()
                self.__nc = self.args.nc

                if self.__pf > 1:
                    self.__pf -= 1
                else:
                    print(f"[epoch: {self.epoch} - {i}/{len(self.train_loader)}]Loss_dnet: {critic_loss:.6f} - "
                          f"Loss_gnet: {generator_loss:.6f} | Score_real: {real_score:.4f} - Score_fake: {fake_score:.4f}")

                    fake_img = fake_img.cpu().data
                    torchvision.utils.save_image(real_img, os.path.join(self.args.img_dir, f"real_{i}.png"),
                                                 nrow=round(pow(self.args.batch_size, 0.5)), normalize=True, scale_each=True)
                    torchvision.utils.save_image(fake_img, os.path.join(self.args.img_dir, f"fake_{i}.png"),
                                                 nrow=round(pow(self.args.batch_size, 0.5)), normalize=True, scale_each=True)
                    self.__pf = self.args.print_freq

    def validate(self):
        return self.value

    def _critic(self, real_img):
        raise NotImplementedError

    def _generator(self):
        raise NotImplementedError
