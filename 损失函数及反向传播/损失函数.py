import torch
from torch import nn
from torch.nn import Conv2d,MaxPool2d,Flatten,Linear
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# inputs = torch.tensor([1,2,3],dtype=torch.float32)
# target = torch.tensor([1,2,5],dtype=torch.float32)

# inputs = torch.reshape(inputs,(1,1,1,3))
# targets = torch.reshape(target,(1,1,1,3))

# loss = L1Loss()
# result = loss(inputs,targets)

# loss_mse = nn.MSELoss()
# result_mse = loss_mse(inputs,targets)

# print(result)
# print(result_mse)

# x = torch.tensor([0.1,0.2,0.3])
# y = torch.tensor([1])
# x = torch.reshape(x,(1,3))
# loss_cross = nn.CrossEntropyLoss()
# result_cross = loss_cross(x,y)
# print(result_cross)



dataset = torchvision.datasets.CIFAR10(root = r'C:\Users\13394\Desktop\项目\PyTorch\dataset',train = False ,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=1,shuffle=True)

class BS(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3,32,5,padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32,32,5,padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32,64,5,padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.linear1 = Linear(1024,64)
        self.linear2 = Linear(64,10)

        # self.model1 = nn.Sequential(
        #     Conv2d(3,32,5,padding=2),
        #     MaxPool2d(2),
        #     Conv2d(32,32,5,padding=2),
        #     MaxPool2d(2),
        #     Conv2d(32,64,5,padding=2),
        #     MaxPool2d(2),
        #     Flatten(),
        #     Linear(1024,64),
        #     Linear(64,10)
        # )
    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

bs=BS()
# print(bs)
# input = torch.ones((64,3,32,32))
# output = bs(input)
# print(output.shape)


for data in dataloader:
    
    imgs,targets = data
    outputs = bs(imgs)
    # print(outputs)
    # print(targets)

loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(outputs,targets)
result_cross.backward()