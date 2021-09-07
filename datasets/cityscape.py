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

class CityScape(Dataset):

    TRAIN_LIST = "ImageSets/main/train.txt"
    VAL_LIST = "ImageSets/main/val.txt"

    def __init__(self, root, data_root, n_classes, img_transform = Compose([]),\
     label_transform=Compose([]), co_transform=Compose([]),\
      train_phase=True,split=1,labeled=True,seed=0,label_correction=False):
        np.random.seed(666)
        self.n_classes = n_classes
        self.root = root
        self.data_root = data_root
        self.images_root = os.path.join(self.data_root, 'subset', 'JPEGImages') # JPEGImages test_img
        self.trimap_root = os.path.join(self.data_root, 'subset', 'SCRLabels') #  SCRLabels test_scr
        self.eval_root = os.path.join(self.data_root, 'subset', 'EvalAnnotations')
        self.img_w = 1025
        self.img_h = 512
        self.train_phase = train_phase
        self.labels = [
                #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color (bgr)         # scribble     # gray-value 128 - 163
                [       'road'                 , 1 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ], # [128,   0 ,  0]      76
                [       'sidewalk'             , 2 ,        1 , 'flat'            , 1       , False        , False        , (232, 35,244) ], # [  0, 128,   0]      149
                [       'building'             , 3 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ], # [  0,   0, 128]      29
                [       'wall'                 , 3 ,        3 , 'construction'    , 2       , False        , False        , (156,102,102) ], # [  0,   0, 128]      29
                [       'fence'                , 3 ,        4 , 'construction'    , 2       , False        , False        , (153,153,190) ], # [  0,   0, 128]      29
                [       'guard rail'           , 3 ,        4 , 'construction'    , 2       , False        , True         , (180,165,180) ], # [  0,   0, 128]      29
                [       'bridge'               , 3 ,        4 , 'construction'    , 2       , False        , True         , (100,100,150) ], # [  0,   0, 128]      29
                [       'tunnel'               , 3 ,        4 , 'construction'    , 2       , False        , True         , ( 90,120,150) ], # [  0,   0, 128]      29
                [       'sign'                 , 4 ,        6 , 'object'          , 3       , False        , False        , ( 30,170,250) ], # [128, 128,   0]      225    
                [       'sign'                 , 4 ,        7 , 'object'          , 3       , False        , False        , (  0,220,220) ], # [128, 128,   0]      225
                [       'tree'                 , 5 ,        8 , 'nature'          , 4       , False        , False        , ( 35,142,107) ], # [128,   0, 128]      105
                [       'sky'                  , 6 ,       10 , 'sky'             , 5       , False        , False        , (180,130, 70) ], # [  0, 128, 128]      178
                [       'person'               , 7 ,       11 , 'human'           , 6       , True         , False        , ( 60, 20,220) ], # [ 64,   0,   0]      48
                [       'person'               , 7 ,       12 , 'human'           , 6       , True         , False        , (  0,  0,255) ], # [ 64,   0,   0]      48
                [       'car'                  , 8 ,       13 , 'vehicle'         , 7       , True         , False        , (142,  0,  0) ], # [  0,  64,   0]      95
                [       'car'                  , 8 ,       14 , 'vehicle'         , 7       , True         , False        , ( 70,  0,  0) ], # [  0,  64,   0]      95
                [       'car'                  , 8 ,       15 , 'vehicle'         , 7       , True         , False        , (100, 60,  0) ], # [  0,  64,   0]      95
                [       'bicycle'              , 9,        17 , 'vehicle'         , 7       , True         , False        , (230,  0,  0) ], # [ 64,  64,   0]      144
                [       'bicycle'              , 9,        18 , 'vehicle'         , 7       , True         , False        , ( 32, 11,119) ], # [ 64,  64,   0]      144
                ]

        self.img_list = read_img_list(os.path.join(self.data_root,'subset',self.TRAIN_LIST)) \
                        if train_phase else read_img_list(os.path.join(self.data_root,'subset',self.VAL_LIST))
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
        self.classes = ('background', 'road', 'sidewalk', 'construction', 
                        'sign', 'tree', 'sky', 'person', 'car', 'bicycle')

    def _to_grayvalue(self, rgb):
        return int(0.2989*rgb[0] + 0.5870*rgb[1] + 0.1140*rgb[2])

    def _decode_mask(self, gt_img):
        h, w, _ = gt_img.shape
        gt_dst = np.zeros((h, w))

        for label in self.labels:
            # print(label)
            color = label[-1]
            label_name, label_ind = label[0], label[1]
            indices = np.where(np.all(gt_img == color, axis=-1))
            print(label_name, color, label_ind, np.shape(indices))
            for x, y in zip(indices[0], indices[1]):
                gt_dst[x][y] = label_ind
        return gt_dst

    def __getitem__(self, index):
        # print("name: ", self.img_list[index])
        filename = self.img_list[index]

        with open(os.path.join(self.images_root,filename+'.png'), 'rb') as f: # jpg
            # print("image: ", filename)
            image = load_image(f).convert('RGB')
            # image = image.resize((self.img_w, self.img_h))
        
        if self.train_phase:
            with open(os.path.join(self.trimap_root,filename+'.png'), 'rb') as f:
                # print("trimap: ", filename)
                trimap = load_image(f).convert('L')
                trimap_np = np.array(trimap)
                # print(trimap_np.shape)
                img_labels = np.unique(trimap_np)
                # trimap_out = self.__gentrimaps__(trimap_np)

            clf_labels = np.zeros(self.n_classes)
            for c in range(self.n_classes):
                if c+1 in img_labels:
                    clf_labels[c] = 1
        else:
            with open(os.path.join(self.eval_root,filename+'.png'), 'rb') as f:
                elabel = load_image(f).convert('L')
                # print("elabel: ", np.unique(np.array(elabel)))
                # elabel = self._decode_mask(np.array(elabel))
                # elabel = Image.fromarray(elabel)

        image_org = image.copy()
        if self.train_phase:
            image, trimap, image_org = self.co_transform((image, trimap, image_org))
            trimap= self.label_transform(trimap)
            clf_labels = self.label_transform(clf_labels)
        else:
            image, image_org, elabel = self.co_transform((image, image_org, elabel))
            elabel = self.label_transform(elabel)

        image = self.img_transform(image)
        image_org = self.label_transform(image_org)
        if self.train_phase:
            return np.array(image), np.array(trimap), np.array(image_org), clf_labels, filename
        else:
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
