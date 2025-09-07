#ReLU & Sigmoid
import torch
from torch import nn
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter
# input = torch.tensor([[1,-0.5],
#                       [-1,3]])
# input = torch.reshape(input,(-1,1,2,2))
dataset = torchvision.datasets.CIFAR10(root = r'C:\Users\13394\Desktop\项目\PyTorch\dataset',train = False , transform = torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size = 64 , shuffle = True)
writer = SummaryWriter('Sigmoid')

class BS(nn.Module):
    def __init__(self):
        super().__init__()
        self.ReLU1 = ReLU(inplace = False)#inplace = True直接替换原变量 inplace = False,原变量不替换 
        self.sigmoid1 = Sigmoid()

    def forward(self,input):
        output = self.sigmoid1(input)
        return output

bs = BS()
# output = bs(input)
step = 0
for data in dataloader:
    imgs, targets = data 
    writer.add_images('input',imgs,step)
    output = bs(imgs)
    writer.add_images('output',output,step)
    step +=1
writer.close()