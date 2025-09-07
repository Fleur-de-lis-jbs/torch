#CIFAR10里面的图片size不一样，单一的linear会导致报错
import torch 
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn import Linear
dataset = torchvision.datasets.CIFAR10(root = r'C:\Users\13394\Desktop\项目\PyTorch\dataset',download = True , transform= torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size = 64,shuffle = True)

writer = SummaryWriter('exp2')

class BS(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = Linear(196608,10)
    def forward(self,input):
        output = self.linear1(input)
        return output 

bs = BS()
step = 0
for data in dataloader:
    imgs,targets = data 
    output = torch.reshape(imgs,(1,1,1,-1))
    print(output.shape)
    output = bs(output)
    print(output.shape)