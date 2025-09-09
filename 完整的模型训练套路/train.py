#如果要在GPU上训练，要把数据（imgs和targets）,模型,以及损失函数转移到GPU上(.cuda())
import torchvision
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.nn import Conv2d,MaxPool2d,Linear,Flatten
from torch.utils.tensorboard import SummaryWriter
#训练集和测试集
train_data = torchvision.datasets.CIFAR10(root = r'C:\Users\13394\Desktop\项目\PyTorch\dataset',train = True , transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root = r'C:\Users\13394\Desktop\项目\PyTorch\dataset',train = False , transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f'训练集的长度是：{train_data_size},测试集的长度是:{test_data_size}')

train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)
#神经网络
class BS(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
    def forward(self,x):
        x = self.model1(x)
        return x

if __name__ == '__main__':
    bs = BS()
    inputs = torch.ones(64,3,32,32)    
    output = bs(inputs)
#损失函数
loss_fn = nn.CrossEntropyLoss()
#优化器
learning_rate = 0.01
optim = torch.optim.SGD(params=bs.parameters(),lr = learning_rate     )
#训练网络的参数
#记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#训练的轮数
epoch = 10
#总准确率
total_accuracy = 0
writer = SummaryWriter('exp4')

for i in range(epoch):
    print(f'第{i+1}轮训练开始')
    #训练开始
    bs.train()
    for data in train_dataloader:
        imgs,targets = data 
        outputs = bs(imgs)
        loss = loss_fn(outputs,targets)
        #优化器优化模型
        optim.zero_grad()#梯度清零
        loss.backward()#反向传播
        optim.step()#进行优化
        
        total_train_step+=1
        if total_train_step%100 == 0:
            print(f'训练次数:{total_train_step},损失值为:{loss.item()}')#loss.item()转化为真实数字类型，而不是tensor
            writer.add_scalar('train_loss',loss.item(),total_train_step)
    #测试步骤
    bs.eval()
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            outputs = bs(imgs)
            loss = loss_fn(outputs,targets)
            total_test_loss+=loss.item()
            accuracy = (outputs.argmax(1)==targets).sum()
            total_accuracy+=accuracy
        print(f'整体测试集上的损失值:{total_test_loss}')
        print(f'整体测试集上的正确率{total_accuracy/test_data_size}')
    writer.add_scalar('test_loss',total_test_loss,total_test_step)
    total_test_step+=1
    torch.save(bs,f'BS第{i}轮.pth')
writer.close()
