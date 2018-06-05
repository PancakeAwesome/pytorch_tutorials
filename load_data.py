# import torch
# import torchvision
import torchvision.transforms as transforms
# from PIL import Image

# cifarSet = torchvision.datasets.CIFAR10(root = './data', train = True, download = True)
# # PILImage
# print(cifarSet[0])
# img, label = cifarSet[0]
# print(img)
# print(label)
# print(img.format, img.size, img.mode)
# img.show()

# mytransform = transforms.Compose([
#     transforms.ToTensor()
# ])

# # torch.utils.data.DataLoader
# cifarSet = torchvision.datasets.CIFAR10(root = "./data", train = True, download = True, transform = mytransform)
# cifarLoader = torch.utils.data.DataLoader(cifarSet, batch_size = 10, shuffle = False, num_workers = 2)

# for i, data in enumerate(cifarLoader):
#     # tensor
#     print(data[i][0])
#     # PIL
#     img = transforms.ToPILImage()(data[i][0])
#     img.show()
#     break

import os
import torch
import torch.utils.data as data
import numpy as np
import cv2
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

# 自定义数据类
class myImageFloader(data.Dataset):
    """自定义的数据类"""
    def __init__(self, root, label, transform = None, target_transform = None, loader = default_loader):
        fh = open(label)
        c = 0
        imgs = [] # 存放图片名和标签
        class_names = []
        # 读取label.txt文件
        for line in fh.readlines():
            if c == 0:
                class_names = [n.strip() for n in line.rstrip().split('    ')]
            else:
                cls = line.split()
                fn = cls.pop(0)
                if os.path.isfile(os.path.join(root, fn)):
                    imgs.append((fn, tuple([float(v) for v in cls])))
            c += 1

        self.root = root
        self.imgs = imgs
        self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, fn))
        if self.transform is not None:
            # to Tensor
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.imgs)

    def getName(self):
        return self.classes

# 实例化torch.utils.data.DataLoader
mytransform = transforms.Compose([
    transforms.ToTensor(),
])

# torch.utils.data.DataLoader
imgLoader = torch.utils.data.DataLoader(myImageFloader(root = "./data/testImages/images", label = "./data/testImages/test_images.txt", transform = mytransform), batch_size = 2, shuffle = False, num_workers = 2)

for i, data in enumerate(imgLoader):
    print(data[i][0])
    # opencv
    img2 = data[i][0].numpy() * 255
    img2 = img2.astype('uint8')
    img2 = np.transpose(img2, (1, 2, 0))
    img2 = img2[:, :, ::-1] # RGB -> BGR
    cv2.imshow('img2', img2)
    cv2.waitKey()
    break