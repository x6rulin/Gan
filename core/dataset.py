import os
from torch.utils.data.dataset import Dataset
from PIL import Image


class GanData(Dataset):

    def __init__(self, img_dir, transform=None):
        super(GanData, self).__init__()

        self.__data = [os.path.join(img_dir, pth) for pth in os.listdir(img_dir)]
        self.__transform = transform

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, item):
        img_data = Image.open(self.__data[item])
        if self.__transform is not None:
            img_data = self.__transform(img_data)

        return img_data
