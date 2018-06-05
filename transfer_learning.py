import torchvision
import torchvision.transforms as transforms
import torch
import argparse
import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim


# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('batch_size', type=int, default = 5)
parser.add_argument('test_batch_size', type=int, default = 5)
# parser.add_argument('cuda', type=bool, default = False)
parser.add_argument('log_interval', type=int, default = 200)

args = parser.parse_args()

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Scale(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.Scale(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 读取数据
trainset = torchvision.datasets.CIFAR10(root = "./data", train = True, download = True, transform = transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, shuffle = True, num_workers = 2)

testset = torchvision.datasets.CIFAR10(root = "./data", train = False, download = True, transform = transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size = args.test_batch_size, shuffle = False, num_workers = 2)

# 加载预训练模型并修改模型
# convnet
model_ft = models.resnet18(pretrained = True) # 加载预训练的模型
print(model_ft)

# 采用迁移学习的第二种方法：冻结所有参数的更新，并修改最后一层fc层参数
for _, param in enumerate(model_ft.parameters()):
    param.requires_grad = False # 冻结参数的更新

num_ftrs = model_ft.fc.in_features # 重新定义fc层，此时，会进行参数的更新
model_ft.fc = nn.Linear(num_ftrs, 10)
print(model_ft)

# 设置多GPU训练模型
# model_ft = torch.nn.DataParaller(model_ft, device_ids = 2).cuda()
# model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss() # 使用交叉熵损失函数
optimizer = optim.SGD(model_ft.fc.parameters(), lr = 0.001, momentum = 0.9)

# 训练
def train(epoch):
    model_ft.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        # if args.cuda:
        #     data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model_ft(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data[0]))

train(2000)