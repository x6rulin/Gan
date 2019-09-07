import os
from torch.utils.data.dataset import Dataset
from PIL import Image


class GanData(Dataset):

    def __init__(self, img_dir, train=True, transform=None):
        super(GanData, self).__init__()

        _pths = os.listdir(img_dir)
        if train:
            self.__data = [os.path.join(img_dir, pth) for pth in _pths[:int(0.9 * len(_pths))]]
        else:
            self.__data = [os.path.join(img_dir, pth) for pth in _pths[int(0.9 * len(_pths)):]]
        self.__transform = transform

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, item):
        img_data = Image.open(self.__data[item])
        if self.__transform is not None:
            img_data = self.__transform(img_data)

        return img_data
