#nn.conv2d
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
data_set = torchvision.datasets.CIFAR10(root = './dataset',train= True,transform=torchvision.transforms.ToTensor(),download=True)

dataloader = DataLoader(data_set,batch_size=64,shuffle=True)

class BS(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3)
    
    def forward (self,x):
        x= self.conv1(x) 
        return x
bs = BS()
writer = SummaryWriter('exp1')
step = 0
for data in dataloader:
    imgs,targets = data 
    output = bs(imgs)
    writer.add_images('imgs',imgs,step)
    output = torch.reshape(output,(-1,3,30,30))#-1自动计算chanel改变，图片数量改变(batch改变)
    writer.add_images('output',output,step)
    step+=1
writer.close()

