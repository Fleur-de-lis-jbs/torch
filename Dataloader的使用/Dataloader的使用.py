import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

data_set_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10(root='./dataset',train=True,transform=data_set_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=data_set_transform,download=True)


test_loader = DataLoader(dataset=test_set,batch_size=64,shuffle=True,num_workers=0,drop_last=True)#返回图片和target

writer = SummaryWriter('dataloader')
step = 0
# img,target = test_set[0]
# print(target) 3
for data in test_loader:
    imgs,targets = data
    writer.add_images('test_loader',imgs,step)
    step+=1
writer.close()