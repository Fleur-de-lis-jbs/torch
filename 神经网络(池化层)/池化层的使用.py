#最大池化
import torch
from torch import nn
from torch.nn import MaxPool2d
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root = r'C:\Users\13394\Desktop\项目\PyTorch\dataset',train = False,download = True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64,shuffle = True)
# input = torch.tensor([[1,2,0,3,1],
#                      [0,1,2,3,1],
#                      [1,2,1,0,0],
#                      [5,2,3,1,1],
#                      [2,1,0,1,1]],dtype = torch.float32)
# input = torch.reshape(input,(-1,1,5,5))
# print(input.shape)

class BS(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = MaxPool2d(kernel_size = 3,ceil_mode = True)

    def forward(self,input):
        output = self.maxpool(input)
        return output
bs = BS()

writer = SummaryWriter('MaxPool')
step = 0
for data in dataloader:
    imgs,targets = data
    writer.add_images('inputs',imgs,step)
    output = bs(imgs)
    writer.add_images('outputs',output,step)
    step +=1
writer.close()