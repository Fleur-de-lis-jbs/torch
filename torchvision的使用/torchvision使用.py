import torchvision
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

data_set_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10(root='./dataset',train=True,transform=data_set_transform,download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset',train=False,transform=data_set_transform,download=True)

# print(test_set[0])
# img , target = test_set[0]
# img.show()

writer = SummaryWriter('exp1')
for i in range(10):
    img ,target = test_set[i]
    writer.add_image('test_set',img,i)
writer.close()
