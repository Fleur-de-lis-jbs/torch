import torch
from torch import nn

class BS(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input):
        output = input +1
        return output
bs = BS()
x = torch.tensor(1.0)
output = bs(x)
print(output)