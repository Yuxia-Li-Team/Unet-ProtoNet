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


class reader(object):
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
        """

    def __init__(self, dir, iter, shot_count,batch_size, size, query_count=1):
        self.dir = dir
        self.iter = iter
        self.shot_count = shot_count
        self.batch_size = batch_size
        self.size = size
        self.query_count = query_count
        self.image_dir = os.path.join(self.dir, 'train_data')
        self.label_dir = os.path.join(self.dir, 'train_label')
        self.record_dir = os.path.join(self.dir, 'records')
        self.all_file_names = self.get_all_filenames()

    """
        procedure:
            -> get all names
            -> generate grouping using all names
            -> using grouping list to read file
            -> processing file to tensor
            -> output
    """

    def get_all_filenames(self):
        filenames = []
        for i in range(1, 5):
            with open(os.path.join(self.record_dir, 'class{}.txt'.format(i)), 'r') as f:
                filenames.append(f.readlines())
        filenames = [[name.strip('\n') for name in classes] for classes in filenames]
        return filenames

    def get_image_by_filename(self, image_name, label_name):
        """
                    Input:
                        list of filenames in the directory
                    Output:
                        tensor of image and label
        """
        image = Image.open(os.path.join(self.image_dir, image_name))
        semantic_mask = Image.open(os.path.join(self.label_dir, label_name))
        # read the image file
        fg_func = lambda x: 1 if x >= 10 else 0
        bg_func = lambda x: 1 if x <= 10 else 0
        bg = semantic_mask.convert('L').point(bg_func, mode='1')
        fg = semantic_mask.convert('L').point(fg_func, mode='1')
        # invert the fg to generate background mask
        image = F.normalize(ToTensor()(image), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = image.unsqueeze(dim=0)
        fg = ToTensor()(fg).unsqueeze(dim=0)
        bg = ToTensor()(bg).unsqueeze(dim=0)
        image = nnF.interpolate(image, size=844, mode='bilinear')
        fg = nnF.interpolate(fg, size=844, mode='bilinear')
        bg = nnF.interpolate(bg, size=844, mode='bilinear')
        image = image.squeeze(dim=0)
        fg = fg.squeeze(dim=0).squeeze(dim=0) # not needed due to protoNet model design
        bg = bg.squeeze(dim=0).squeeze(dim=0)
        sample = {'image': image, 'fg': fg, 'bg': bg, 'name': label_name}
        return sample

    def __getitem__(self, index):
        """
            Input:
                required prediction target, determined by input index
            Output:
                group of images contains both support image and query image
        """
        filenames = self.get_all_filenames() # list of lists of filenames
        current_index_class = [index + 1, None]
        list_length = []
        for class_list in filenames:
            list_length.append(len(class_list))
        if 1 <= current_index_class[0] <= list_length[0]:
            current_index_class[0] = current_index_class[0] - 1
            current_index_class[1] = 0
        elif list_length[0] < current_index_class[0] <= list_length[1] + list_length[0]:
            current_index_class[0] = current_index_class[0] - list_length[0] - 1
            current_index_class[1] = 1
        elif list_length[1] + list_length[0] < current_index_class[0] <= list_length[2] + list_length[1] + list_length[0]:
            current_index_class[0] = current_index_class[0] - list_length[1] - list_length[0] - 1
            current_index_class[1] = 2
        elif list_length[2] + list_length[1] + list_length[0] < current_index_class[0] <= list_length[3] + list_length[2] + list_length[1] + list_length[0]:
            current_index_class[0] = current_index_class[0] - list_length[2] - list_length[1] - list_length[0] - 1
            current_index_class[1] = 3
        else:
            raise IndexError('index out of bound for every class, the index now is {}'.format(current_index_class))
        # covert the index into every independent list

        group_num = current_index_class[1]
        # get current file's class
        exist_flag = False
        # a dummy flag
        while exist_flag is False:
            rand_sqnum = random.choices(range(len(self.all_file_names[group_num])), k=self.shot_count)
            # randomly choose amount files that is within the same class
            rand_shot_labels = [self.all_file_names[group_num][name] for name in rand_sqnum]
            rand_images = [self.all_file_names[group_num][name][:-8] + 'sat.jpg' for name in rand_sqnum]
            if current_index_class[0] not in rand_shot_labels:
                exist_flag = True
        # exclude the results that already contain the query image itself
        task = []
        for i in range(len(rand_shot_labels)):
            task.append(self.get_image_by_filename(rand_images[i], rand_shot_labels[i]))
        task.append(self.get_image_by_filename(self.all_file_names[current_index_class[1]][current_index_class[0]][:-8] + 'sat.jpg',
                                               self.all_file_names[current_index_class[1]][current_index_class[0]]))
        return task

    def __len__(self):
        return self.iter

if __name__ == '__main__':
    data = reader(dir=r'C:/Users/Tim Wang/Desktop/data/', iter=1,
                  shot_count=5, query_count=1, batch_size=1, size=1024)
    task = data[5816]
    figure = plt.figure()
    ax1 = figure.add_subplot(1, 2, 1)
    ax1.imshow(task[0]['fg'], cmap='gray')
    ax2 = figure.add_subplot(1, 2, 2)
    ax2.imshow(task[0]['bg'], cmap='gray')
    plt.show()
    print(task)

