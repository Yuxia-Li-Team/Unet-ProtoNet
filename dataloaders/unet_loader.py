import os
import numpy as np
from PIL import Image
import torch
import random
import torchvision.transforms.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
import torch.nn.functional as nnF
from torch.utils.data import Dataset
import PIL.ImageOps

class UnetLoader(Dataset):
    def __init__(self, ImgDir, MaskDir, FPredDir, ImgSize):
        self.ImgDir = ImgDir
        self.MaskDir = MaskDir
        self.FPredDir = FPredDir
        self.ImgSize = ImgSize
        self.Names = self.GetNames()

    def GetNames(self):
        datanames = []
        for root, dirs, files in os.walk(self.ImgDir):
            for file in files:
                if os.path.splitext(file)[1] == '.jpg':
                    datanames.append(os.path.join(os.path.splitext(file)[0]) + '.jpg')
        labelnames = []
        for root, dirs, files in os.walk(self.MaskDir):
            for file in files:
                if os.path.splitext(file)[1] == '.png':
                    labelnames.append(os.path.join(os.path.splitext(file)[0]) + '.png')
        fprednames = []
        for root, dirs, files in os.walk(self.FPredDir):
            for file in files:
                if os.path.splitext(file)[1] == '.png':
                    fprednames.append(os.path.join(os.path.splitext(file)[0]) + '.png')
        assert len(datanames) == len(labelnames) == len(fprednames)
        BigList = []
        for i in range(len(datanames)):
            datapoint = [datanames[i], labelnames[i], fprednames[i]]
            BigList.append(datapoint)
        return BigList

    def SingleDataPointRead(self, Name):
        DataPath = os.path.join(self.ImgDir, Name[0])
        LabelPath = os.path.join(self.MaskDir, Name[1])
        FPredPath = os.path.join(self.FPredDir, Name[2])
        image = Image.open(DataPath)
        label = Image.open(LabelPath)
        fpred = Image.open(FPredPath)
        fn_2 = lambda x: 0 if x <= 10 else 1
        label = label.convert('L').point(fn_2, mode='1')
        fpred = fpred.convert('L')
        fpred = PIL.ImageOps.invert(fpred)
        image = F.normalize(ToTensor()(image), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = nnF.interpolate(image.unsqueeze(dim=0), size=self.ImgSize, mode='bilinear', align_corners=True).squeeze(dim=0)
        label = nnF.interpolate(ToTensor()(label).unsqueeze(dim=0), size=self.ImgSize, mode='bilinear', align_corners=True).squeeze(dim=0)
        fpred = nnF.interpolate(ToTensor()(fpred).unsqueeze(dim=0), size=self.ImgSize, mode='bilinear', align_corners=True).squeeze(dim=0)
        return {'image': image, 'label': label, 'fpred': fpred}

    def __getitem__(self, index):
        Names = self.Names
        SingleSampleName = Names[index]
        DataPoint = self.SingleDataPointRead(SingleSampleName)
        return DataPoint

    def __len__(self):
        return len(self.Names)


if __name__ == '__main__':
    loader_instance = UnetLoader(r'C:\Users\Tim Wang\Desktop\data\train_data',
                                 r'C:\Users\Tim Wang\Desktop\data\train_label',
                                 r'C:\Users\Tim Wang\Desktop\data\cache',
                                 ImgSize=512)
    name = ['100034_sat.jpg', '100034_mask.png', '100034__fewshotpred.png']
    pic = loader_instance.SingleDataPointRead(name)['label']
    plt.imshow(pic[0], cmap='gray')
    plt.show()

    print(loader_instance.SingleDataPointRead(name)['image'].size())