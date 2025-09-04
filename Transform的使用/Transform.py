#数据预处理：如归一化、尺寸调整、格式转换等
#数据增强：如随机裁剪、翻转、旋转等（用于训练集，增加数据多样性）
#组合变换：通过 Compose 将多个变换操作串联执行
from torchvision import transforms
from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter('logs')
#ToTensor
img_path = r'C:\Users\13394\Desktop\项目\PyTorch\weather_recognition-main\weather_recognition-main\data\train\sleet\03.jpg'
img = Image.open(img_path)
img_cv = cv2.imread(img_path)


tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
tensor_imgcv = tensor_trans(img_cv)

writer.add_image('ToTensor',tensor_img,1)


#Normalize 标准化一个tensor图片
# print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])#传入均值，标准差
img_norm = trans_norm(tensor_img)
# print(img_norm[0][0][0])

writer.add_image('ToTensor',img_norm,2)

#Resize传入PIL图片
# print(img.size)
trans_resize = transforms.Resize((512,512))
img_resize = trans_resize(img)
img_resize = tensor_trans(img_resize)
# print(img_resize.size)

writer.add_image('ToTensor',img_resize,3)

#Compose
# PIL->PIL->Tensor
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2,tensor_trans])
img_resize2 = trans_compose(img)

writer.add_image('ToTensor',img_resize2,4)

#RandomCrop
trans_random = transforms.RandomCrop(512)
trans_compose2 = transforms.Compose([trans_random,tensor_trans])

for i in range(10):
    img_crop = trans_compose2(img)
    writer.add_image('RandomCrop',img_crop,i)

#ToPILImage


writer.close()