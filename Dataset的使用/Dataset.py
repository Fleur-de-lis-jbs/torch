#Dataset类
from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):

    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir,self.label_dir)
        self.img_path = os.listdir(self.path)


    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label
    def __len__(self):
        return len(self.img_path)

root_dir = r'C:\Users\13394\Desktop\项目\PyTorch\weather_recognition-main\weather_recognition-main\data\train'
label_dir = 'frost'
frost_dataset = MyData(root_dir,label_dir)
print(len(frost_dataset))