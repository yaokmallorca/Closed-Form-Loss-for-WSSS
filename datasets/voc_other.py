import numpy as np
import os
from PIL import Image
import torch
import json
from torch.utils.data import Dataset
from torchvision.transforms import Compose
# home_dir = os.getcwd()
# print(home_dir)
# os.chdir(home_dir)

import cv2
import matplotlib.pyplot as plt 
import seaborn as sns

# from ..utils.transforms import OneHotEncode, OneHotEncode_smooth

def load_image(file):
    return Image.open(file)

def read_img_list(filename):
    print(filename)
    with open(filename) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return np.array(img_list)

"""
JPEGImages : normMean = [0.38009313 0.39160565 0.25242192]
JPEGImages : normstdevs = [0.12935428 0.16977909 0.1502691 ]
"""

class Voc(Dataset):

    TRAIN_LIST = "ImageSets/main/train.txt"
    VAL_LIST = "ImageSets/main/all.txt"

    def __init__(self, root, data_root, n_classes, img_transform = Compose([]),\
     label_transform=Compose([]), co_transform=Compose([]),\
      train_phase=True,split=1,labeled=True,seed=0,label_correction=False):
        np.random.seed(666)
        self.n_classes = n_classes
        self.root = root
        self.data_root = data_root
        self.images_root = os.path.join(self.data_root, 'VOCSUB', 'JPEGImages') # JPEGImages test_img
        self.trimap_root = os.path.join(self.data_root, 'VOCSUB', 'SCRLabels') #  SCRLabels test_scr
        self.eval_root = os.path.join(self.data_root, 'VOCSUB', 'EvalAnnotations')
        self.img_w = 321
        self.img_h = 321
        self.train_phase = train_phase

        self.img_list = read_img_list(os.path.join(self.data_root,'VOCSUB',self.TRAIN_LIST)) \
                        if train_phase else read_img_list(os.path.join(self.data_root,'VOCSUB',self.VAL_LIST))
        self.split = split
        self.labeled = labeled
        n_images = len(self.img_list)
        self.img_l = np.random.choice(range(n_images),int(n_images*split),replace=False) # Labeled Images
        self.img_u = np.array([idx for idx in range(n_images) if idx not in self.img_l],dtype=int) # Unlabeled Images
        if self.labeled:
            self.img_list = self.img_list[self.img_l]
        else:
            self.img_list = self.img_list[self.img_u]
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.co_transform = co_transform
        self.label_correction = label_correction
        # subset
        self.classes = ('background', 'plane', 'bike', 'person', 'car', 'dog', 'cat')
        self.palette = [[0, 0, 0], [128, 0, 0],[0, 128, 0],[192, 128, 128], 
                        [128, 128, 128],[64, 0, 128],[64, 0, 0]]
        self.palette_grayscale = [0, 38, 75, 147, 128, 33, 19]

        # full dataset
        """
        self.classes = ('plane',        'bike',     'bird',     'boat',         'bottle',
                        'bus',          'car',      'cat',      'chair',        'cow',      
                        'table',        'dog',      'horse',    'motorbike',    'person', 
                        'plant',        'sheep',    'sofa',     'train',        'tvmonitor', 
                        'background')
        self.palette = [[128, 0, 0],    [0, 128, 0],        [128, 128, 0],  [0, 0, 128],    [128, 0, 128], 
                        [0, 128, 128],  [128, 128, 128],    [64, 0, 0],     [192, 0, 0],    [64, 128, 0], 
                        [192, 128, 0],  [64, 0, 128],       [192, 0, 128],  [64, 128, 128], [192, 128, 128], 
                        [0, 64, 0],     [128, 64, 0],       [0, 192, 0],    [128, 192, 0],  [0, 64, 128], 
                        [0, 0, 0]]
        """
    def _to_grayvalue(self, rgb):
        return int(0.2989*rgb[0] + 0.5870*rgb[1] + 0.1140*rgb[2])

    """
    def decode_palette(self):
        for i in range(len(self.palette)):
            palette = self.palette[i]
            self.palette_grayscale.append(self.gray_palette(palette))
            # print("{} gray val: {}".format(self.classes[i], self._to_grayvalue(palette)))
    """

    def _decode_mask(self, mask_org):
        mask = np.zeros_like(mask_org)
        for gval in self.palette_grayscale:
            mask[mask_org == gval] = self.palette_grayscale.index(gval)
        mask[mask_org == 220] = 0
        mask[mask_org == self.palette_grayscale[0]] = 0
        # cv2.imwrite("mask_test.png", mask*10)
        # return torch.from_numpy(mask).type(torch.LongTensor)
        return mask

    def __getitem__(self, index):
        # print("name: ", self.img_list[index])
        filename = self.img_list[index]

        with open(os.path.join(self.images_root,filename+'.png'), 'rb') as f: # jpg
            # print("image: ", filename)
            image = load_image(f).convert('RGB')
            # image = image.resize((self.img_w, self.img_h))
        
        # if self.train_phase:
        # with open(os.path.join(self.trimap_root,filename+'.png'), 'rb') as f:
        #     # print("trimap: ", filename)
        #     trimap = load_image(f).convert('L')
        #     trimap_np = np.array(trimap)
        #     img_labels = np.unique(trimap_np)
        #     # trimap_out = self.__gentrimaps__(trimap_np)

        with open(os.path.join(self.eval_root,filename+'.png'), 'rb') as f:
            elabel = load_image(f).convert('L')
            elabel = self._decode_mask(np.array(elabel))
            elabel = Image.fromarray(elabel)

        image_org = image.copy()
        image, image_org, elabel = self.co_transform((image, image_org, elabel))
        elabel = self.label_transform(elabel)

        image = self.img_transform(image)
        image_org = self.label_transform(image_org)
        return np.array(image), np.array(image_org), elabel, filename

    def __len__(self):
        return len(self.img_list)


def test():
    import sys
    home_dir = "/home/yaok/software/closed-form-segmentation"
    sys.path.append(home_dir)

    from utils.transforms import RandomSizedCrop, IgnoreLabelClass, ToTensorLabel, NormalizeOwn,ZeroPadding, OneHotEncode, RandomSizedCrop3
    from torchvision.transforms import ToTensor,Compose
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    imgtr = [ToTensor(),NormalizeOwn()]
    # sigmoid 
    labtr = [ToTensorLabel(tensor_type=torch.FloatTensor)]
    # cotr = [RandomSizedCrop((320,320))] # (321,321)
    cotr = [RandomSizedCrop2((312,312))]

    dataset_dir = '/media/data/seg_dataset'
    trainset = Corrosion(home_dir, dataset_dir,img_transform=Compose(imgtr), 
                           label_transform=Compose(labtr),co_transform=Compose(cotr),
                           split=False,labeled=True)
    trainloader = DataLoader(trainset,batch_size=1,shuffle=True,num_workers=1,drop_last=True)

    for batch_id, (img, trimaps, img_org, img_names) in enumerate(trainloader):
        img, trimaps, img_org = img.numpy()[0], trimaps.numpy()[0], img_org.numpy()[0]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(img)
        ax2.imshow(trimaps)
        ax3.imshow(img_org)
        plt.show()

if __name__ == '__main__':
    test()