import sys
import os
import torchvision

rootPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(rootPath)

if __name__ == "__main__":
    from local_lib.dcgan import Discriminator, Generator
    from core.loss import LogDloss
    from core.dataset import GanData
    from core.train import Train

    img_dir = r"/home/data/Cartoon_faces/faces"
    save_dir = r"/home/data/Cartoon_faces/dcgan"

    transform = transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = GanData(img_dir, train=True, transform=transform)
    val_dataset = GanData(img_dir, train=False, transform=transform)

    sample_num=128
    _gnet = Generator(sample_num)
    _dnet = Discriminator(out_activate='sigmoid')
    trainer = Train(_gnet, _dnet, sample_num, train_dataset, val_dataset, LogDloss())
    trainer()
