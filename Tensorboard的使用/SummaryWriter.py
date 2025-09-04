from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np


writer = SummaryWriter('exp1')#日志写入器，会写在指定文件夹
img_path = r'weather_recognition-main/weather_recognition-main/data/img/aridity/02.jpg'
img = Image.open(img_path)
img_array = np.array(img)#img_array 是通过 PIL 打开图像后转换的 NumPy 数组，其维度格式应为 (H, W, C)（高度 × 宽度 × 通道数），而 add_image() 默认期望 (C, H, W) 格式
writer.add_image('test',img_array,2,dataformats='HWC')#用于记录图像数据，可在 TensorBoard 中直接查看图像，适合可视化输入数据、中间特征图、生成结果等.
# y=x
for i in range(100):
    writer.add_scalar('y=x',i,i,)#用于记录标量数据（如损失值、准确率、学习率等），在 TensorBoard 中会以折线图形式展示，非常适合跟踪训练过程中的指标变化。
    #如何进入网站： 进入环境后，tensorboard --logdir=exp1 --port=xxxx 指定一个端口打开
writer.close()