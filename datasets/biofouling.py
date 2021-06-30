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
    with open(filename) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return np.array(img_list)

"""
JPEGImages : normMean = [0.38009313 0.39160565 0.25242192]
JPEGImages : normstdevs = [0.12935428 0.16977909 0.1502691 ]
"""

class Biofouling(Dataset):

    TRAIN_LIST = "ImageSets/train.txt"
    VAL_LIST = "ImageSets/train.txt"

    def __init__(self, root, data_root, img_transform = Compose([]),\
     label_transform=Compose([]), co_transform=Compose([]),\
      train_phase=True,split=1,labeled=True,seed=0,label_correction=False):
        np.random.seed(666)
        self.n_class = 2
        self.root = root
        self.data_root = data_root
        self.images_root = os.path.join(self.data_root, 'BioFouling', 'JPEGImages')
        # print("images_root: ", self.images_root)
        # SemanticLabels BoxLabels SCRLabels3
        # Scribble: SCRLabelsTrue
        # Superpixel: SUPLabels
        self.trimap_root = os.path.join(self.data_root, 'BioFouling', 'SCRLabels') #  SCRLabelsTrue_20 SUPLabels_80
        # print("labels_root: ", self.labels_root)
        # self.elabels_root = os.path.join(self.data_root, 'BioFouling', 'EvaluateLabels')
        # self.clabels_root = os.path.join(self.data_root, 'BioFouling', 'EvaluateLabels')
        # self.json_root = os.path.join(self.data_root, 'corrosion', 'json')
        self.img_list = read_img_list(os.path.join(self.data_root,'BioFouling',self.TRAIN_LIST)) \
                        if train_phase else read_img_list(os.path.join(self.data_root,'BioFouling',self.VAL_LIST))
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
        self.train_phase = train_phase
        self.label_correction = label_correction

    def __getitem__(self, index):
        filename = self.img_list[index]

        with open(os.path.join(self.images_root,filename+'.png'), 'rb') as f:
            image = load_image(f).convert('RGB')
            # image.save(filename + '_org.png')
        with open(os.path.join(self.trimap_root,filename+'.png'), 'rb') as f:
            trimap = load_image(f).convert('L')
            trimap_np = np.array(trimap)

        image_org = image.copy()
        image, trimap, image_org = self.co_transform((image, trimap, image_org))
        image = self.img_transform(image)
        trimap = self.label_transform(trimap)
        image_org = self.label_transform(image_org)
        return np.array(image), np.array(trimap), np.array(image_org), filename

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
    cotr = [RandomSizedCrop3((320,320))]

    dataset_dir = '/media/data/seg_dataset'
    trainset = Biofouling(home_dir, dataset_dir,img_transform=Compose(imgtr), 
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