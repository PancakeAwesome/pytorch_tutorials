import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

class LeNetSeq(nn.Module):
    """利用torch.nn.Sequential来搭建网络"""
    def __init__(self):
        super(LeNetSeq, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
        
    def forward(self, x):
        x = self.conv(x)
        x = out.view(x.size(0), -1)
        out = self.fc(x)
        return out

net = LeNet()
print(net)

# for param in net.parameters():
    # print(type(param.data), param.size())
    # print(list(param.data))

print(net.state_dict().keys())
# 模型的参数的keys

# for key in net.state_dict():
    # print(key, 'corresponds to', list(net.state_dict()[key]))

import torch.nn.init as init
# 参数初始化
# 使用 torch.nn.init 
def initNetParams(net):
    """
    Init net params
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform(m.weight)
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):      
            init.normal(m.weight, std = 1e-3)
            if m.bias is not None:
                init.constant(m.bias, 0)

initNetParams(net)

# 保存模型
torch.save(net, 'net.ss') # 保存整个神经网络的结构和模型参数
torch.save(net.state_dict(), 'net_params.ss') # 只保存神经网络的模型参数

# 重载整个模型
model = torch.load('net.ss')
print(model)

net.load_state_dict(torch.load('net_params.pth'))
print(net)