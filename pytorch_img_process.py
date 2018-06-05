import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import PIL.Image as Image

img_path = './02.jpg'

# 转换成tensor
transform1 = transforms.Compose([
    transforms.ToTensor(), # [0, 255] -> [0.0, 1.0],[C, H, W]
    ]
)

# numpy数组
# imread返回的是python矩阵
img = cv2.imread(img_path) # 读取图像
# print(img.shape)
img1 = transform1(img) # 归一化到[0.0, 1.0]
print("img1 = ", img1)
# 将tensor转换为numpy数组，归一化到[0.0, 255.0]
img_1 = img1.numpy()*255
img_1 = img_1.astype('uint8')
img_1 = np.transpose(img_1, (1,2,0))
cv2.imshow('img_1', img_1)
cv2.waitKey()

# PIL图像
img = Image.open(img_path).convert('RGB') # 读取图像
img2 = transform1(img) # 归一化到[0.0, 1.0]
print("img2 = ", img2)
# 转化为PILImage并显示
img_2 = transforms.ToPILImage()(img2).convert('RGB')
print("img_2 = ", img_2)
img_2.show()

transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
])

# transorms.RandomCrop()
transform4 = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.RandomCrop((300, 300)),
])

img = Image.open(img_path).convert('RGB')
img3 = transform4(img)
img3.show()