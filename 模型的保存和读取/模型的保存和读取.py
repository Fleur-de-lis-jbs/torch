import torchvision
from torchvision.models import VGG16_Weights,vgg16
import torch


# vgg16 = torchvision.models.vgg16(weights = VGG16_Weights.DEFAULT)
# 保存方式1 保存模型结构和参数
# torch.save(vgg16,'vgg16_method1.pth')
# model = torch.load(
#     r'C:\Users\13394\Desktop\项目\PyTorch\vgg16_method1.pth',
#     weights_only=False
# )
# print(model)

#保存方式2 模型参数
# torch.save(vgg16.state_dict(),'vgg16_method2.pth')
model = vgg16(weights=VGG16_Weights.DEFAULT)
model.load_state_dict(torch.load(r'C:\Users\13394\Desktop\项目\PyTorch\vgg16_method2.pth'))
print(model)