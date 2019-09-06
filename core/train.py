import os
import torch
import torchvision
from torch.utils.data import DataLoader


class Train:

    def __init__(self, dataset, gnet, dnet, save_dir, sample_num=128, batch_size=32, num_workers=8):
        is_cuda = torch.cuda.is_available()
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=is_cuda)

        self.gnet = gnet
        self.dnet = dnet

        params_dir = os.path.join(save_dir, r'params/')
        if not os.path.exists(params_dir):
            os.makedirs(params_dir, 0o775)
        self.params_pth = {'gnet': os.path.join(params_dir, r'gnet.pth'),
                           'dnet': os.path.join(params_dir, r'dnet.pth')}

        self.img_dir = os.path.join(save_dir, r'images/')
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

        self.sample_num = sample_num
        self.device = torch.device(['cpu', 'cuda:0'][is_cuda])

        self.__resume(self.gnet, self.params_pth['gnet'])
        self.__resume(self.dnet, self.params_pth['dnet'])

    def __resume(self, model, path):
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location='cpu'))
            print(f"loading weights from {path} ...")
        model.to(self.device)

    def __call__(self, epochs=256, lr=2e-4, freq=16):
        optimizer = {'gnet': torch.optim.Adam(self.gnet.parameters(), lr=lr, betas=(0.5, 0.99)),
                     'dnet': torch.optim.Adam(self.dnet.parameters(), lr=lr, betas=(0.5, 0.99), weight_decay=1e-3)}
        criterion = torch.nn.BCELoss()

        for epoch in range(1, epochs + 1):
            print(f"epochs: {epoch}")
            self.gnet.train()
            self.dnet.train()
            for i, real_img in enumerate(self.data_loader, 1):
                real_label = torch.ones(real_img.size(0), 1, 1, 1, device=self.device)
                fake_label = torch.zeros(real_img.size(0), 1, 1, 1, device=self.device)
                real_img = real_img.to(self.device)

                real_out = self.dnet(real_img)
                loss_real = criterion(real_out, real_label)
                sample = torch.randn(real_img.size(0), self.sample_num, 1, 1, device=self.device)
                fake_img = self.gnet(sample)
                fake_out = self.dnet(fake_img)
                loss_fake = criterion(fake_out, fake_label)

                dnet_loss = loss_real + loss_fake
                optimizer['dnet'].zero_grad()
                dnet_loss.backward()
                optimizer['dnet'].step()

                sample = torch.randn(real_img.size(0), self.sample_num, 1, 1, device=self.device)
                img = self.gnet(sample)
                output = self.dnet(img)
                gnet_loss = criterion(output, real_label)

                optimizer['gnet'].zero_grad()
                gnet_loss.backward()
                optimizer['gnet'].step()

                if i % freq == 0:
                    print(f"[epoch: {epoch} - {i}/{len(self.data_loader)}]Loss_dnet: {dnet_loss:.4f} - Loss_gnet: {gnet_loss:.4f}"
                          f" | Score_real: {real_out.data.mean():.4f} - Score_fake: {output.data.mean():.4f}")

                    real_img = real_img.cpu().data
                    fake_img = img.cpu().data
                    torchvision.utils.save_image(real_img, os.path.join(self.img_dir, f"{epoch}_{i}_real_img.png"), \
                                                 nrow=round(pow(real_img.size(0), 0.5)), normalize=True, scale_each=True)
                    torchvision.utils.save_image(fake_img, os.path.join(self.img_dir, f"{epoch}_{i}_fake_img.png"), \
                                                 nrow=round(pow(real_img.size(0), 0.5)), normalize=True, scale_each=True)

                    torch.save(self.gnet.state_dict(), self.params_pth['gnet'])
                    torch.save(self.dnet.state_dict(), self.params_pth['dnet'])


if __name__ == "__main__":
    import sys
    import os
    import torchvision

    rootPath = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(rootPath)

    from models import Discriminator, Generator
    from dataset import GanData

    img_dir = r"/home/data/Cartoon_faces/faces"
    save_dir = r"/home/data/Cartoon_faces/dcgan"

    dataset = GanData(img_dir, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

    sample_num=128
    _gnet = Generator(sample_num)
    _dnet = Discriminator(out_activate='sigmoid')

    trainer = Train(dataset, _gnet, _dnet, sample_num, save_dir, batch_size=64)
    trainer(freq=32)
