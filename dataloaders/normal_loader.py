import os
import numpy as np
from PIL import Image
import torch
import random
import torchvision.transforms.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
import torch.nn.functional as nnF
import torch.cuda

class RSI(object):
    """
        Outputs:
            sup_img: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fg: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            bg: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            query_imgs: query images
                N x [B x 3 x H x W], list of tensors
            B = batchsize, H = height, W = width

            NOTE: the batchsize is determined HERE, not in torch.utils.data.Dataloader
            Please set the Dataloader's batchsize = 1 to avoid getting unexpected results
    """

    def __init__(self, base_dir, mode, n_member, n_group, batch_size, size):
        self.mode = mode  # a TO DO function, not useful for now
        self._base_dir = base_dir  # file direction
        self.n_group = n_group  # defines how many groups to choose from dataset
        self.n_member = n_member  # defines the number of data points in each group
        self.bs = batch_size  # defines the first dimension's length of the tensor
        self.size = size  # one int, the image will be resized to size * size
        self._image_dir = os.path.join(self._base_dir, 'train_data')
        self._label_dir = os.path.join(self._base_dir, 'train_label')
        self.ids = self.getImgIds()
        self.len = len(self.getImgIds())
        self.id_map = self.getRandGroup()
        self.viewer = 0

    '''get the ID of the selected group's pictures e.g. train, val, test'''
    def getImgIds(self):
        ids = []
        for root, dirs, files in os.walk(self._image_dir):
            for file in files:
                if os.path.splitext(file)[1] == '.jpg':
                    ids.append(os.path.join(os.path.splitext(file)[0][:-4]))
        if self.mode == 'train':
            return ids

        # get the image id, so we can map the serial number to real file name
        # [:-4] is used to delete the suffix _sat

    def getRandGroup(self):
        id_mapping = []
        for i in range(self.n_group):
            id_mapping.append((random.choices(range(self.len), k=self.n_member)))
        return id_mapping

        # create a random tasks, the length is determined by the input n_group
        # returns a list of list

    def getimg(self, index):
        if index <= self.len - 1:
            id_ = self.ids[index]
        else:
            id_ = self.ids[index - self.len]
        image = Image.open(os.path.join(self._image_dir, f'{id_}_sat.jpg'))
        semantic_mask = Image.open(os.path.join(self._label_dir, f'{id_}_mask.png'))
        mask1 = semantic_mask.copy()
        fn_1 = lambda x: 1 if x >= 10 else 0  # put pixel to 1 if it is not 0 (choose 10 is a dummy threshold)
        fn_2 = lambda x: 0 if x >= 10 else 1  # put pixel to 0 if it is 0
        fg = mask1.convert('L').point(fn_2, mode='1')
        bg = mask1.convert('L').point(fn_1, mode='1')
        semantic_mask = fg  # this holds when it is a 1-way task
        image1 = F.normalize(ToTensor()(image), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image1 = self.resize(image1, (1, 3, self.size, self.size), self.size)
        semantic_mask = self.resize(ToTensor()(semantic_mask), (1, self.size, self.size), self.size)
        fg = self.resize(ToTensor()(fg), (1, self.size, self.size), self.size)
        bg = self.resize(ToTensor()(bg), (1, self.size, self.size), self.size)
        sample = {'image': image1, 'label': semantic_mask,
                  'id': id_, 'fg': fg, 'bg': bg}
        image.close()
        mask1.close()  # close the PIL process to save resources, not sure if it really helps

        return sample

    def resize(self, tensor, reshape, size):
        tensor = tensor.view(1, -1, 1024, 1024)
        tensor = nnF.interpolate(tensor, size=[size, size], mode='bilinear')
        if tensor.shape == reshape:
            pass
        else:
            tensor = tensor.view(reshape)
        return tensor
    # resizing the image without going to PIL,
    # consider to do this task separately in order to speed up reading

    def pairing(self, index):
        pair_pos_init = self.id_map[index]
        samples = [self.getimg(name) for name in pair_pos_init]
        paired_images = []
        query_img_group = []
        if self.bs == 1:
            for i, member in enumerate(samples):
                img = member['image']
                if i <= self.n_member - 2:
                    paired_images.append(img)
                else:
                    query_img_group.append(img)
        else:
            for i, member in enumerate(samples):
                img = member['image']
                for j in range(1, self.bs):
                    img = torch.cat((img, self.getimg(pair_pos_init[i] + j)['image']), dim=0)
                if i <= self.n_member - 2:
                    paired_images.append(img)
                else:
                    query_img_group.append(img)

        # pick the last image in group to be the query label

        query_label_group = []
        query_label = samples[self.n_member - 1]['label']
        if self.bs != 1:
            for j in range(1, self.bs):
                query_label = torch.cat((query_label, self.getimg(pair_pos_init[self.n_member - 1] + j - 1)['label']),
                                        dim=0)
            query_label_group.append(query_label)
        else:
            query_label_group.append(query_label)

        # pick the last label in group to be the query label

        paired_bg = []
        if self.bs == 1:
            for member in samples[:-1]:
                bg = member['bg']
                paired_bg.append(bg)
        else:
            for i, member in enumerate(samples[:-1]):
                bg = member['bg']
                for j in range(1, self.bs):
                    bg = torch.cat((bg, self.getimg(pair_pos_init[i] + j)['bg']), dim=0)
                paired_bg.append(bg)
        # print('bg pairing finished')
        # bg mask is the mask which the target's mask is 0 and background is 1

        paired_fg = []
        if self.bs == 1:
            for member in samples[:-1]:
                fg = member['fg']
                paired_fg.append(fg)
        else:
            for i, member in enumerate(samples[:-1]):
                fg = member['fg']
                for j in range(1, self.bs):
                    fg = torch.cat((fg, self.getimg(pair_pos_init[i] + j)['fg']), dim=0)
                paired_fg.append(fg)
        # fg mask is the mask which the target's mask is 1 and background is 0
        # print('fg pairing finished')

        return paired_images, paired_fg, paired_bg, query_img_group, query_label_group

    def Augmentation(self, rotating=False, colour_change=False):
        pass
    # To do function, could have huge performance impact, consider to do Augmentation separately

    def __getitem__(self, index):
        img, fg, bg, query_img, query_label = self.pairing(index=index)
        sample = {'sup_img': [img], 'sup_bg': [bg], 'sup_fg': [fg], 'query_img': query_img,
                  'query_label': query_label}
        return sample

    def __len__(self):
        return self.n_group
    # return a training length to Dataloader


    # returning n-1 support image, 1 query image
    # returning normalized image, bg, fg masks, query image and binary mask

if __name__ == '__main__':
    data = RSI(base_dir=r'C:/Users/alienware/Desktop/data/', mode='train', n_group=10,
               n_member=6, batch_size=1, size=512)
    inst = data[4]
    print(inst['sup_img'][0][2][0].is_cuda)
    plt.imshow(ToPILImage()(inst['sup_img'][0][2][0]))
    plt.show()
    plt.imshow(ToPILImage()(inst['sup_img'][0][3][0].cpu()))
    plt.show()
    plt.imshow(np.asarray(inst['sup_fg'][0][2][0].cpu()), cmap='Greys')
    plt.show()
    plt.imshow(np.asarray(inst['sup_fg'][0][3][0].cpu()), cmap='Greys')
    plt.show()
    plt.imshow(np.asarray(inst['query_label'][0][0].cpu()), cmap='Greys')
    plt.show()
    plt.imshow(ToPILImage()(inst['query_img'][0][0].cpu()))
    plt.show()
