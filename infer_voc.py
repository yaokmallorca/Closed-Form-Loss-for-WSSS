# from datasets.corrosion import Corrosion
from datasets.voc import Voc
# from datasets.voc_other import Voc
from torch.utils.data import DataLoader
# import generators.unet as unet
import generators.deeplabv3 as deeplab
# import generators.encoder as encoder

import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import os.path as osp
import numpy as np
# from utils.metrics import scores
import torchvision.transforms as transforms
from utils.transforms import ResizedImage3, IgnoreLabelClass, ToTensorLabel, NormalizeOwn, ZeroPadding
from utils.metrics import scores
from torchvision.transforms import ToTensor
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from torchsummary import summary

class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (0 == gray_image) # 255
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 0 # 255
        return color_image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([g, r, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def make_palette(num_classes):
    """
    Maps classes to colors in the style of PASCAL VOC.
    Close values are mapped to far colors for segmentation visualization.
    See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit

    Takes:
        num_classes: the number of classes
    Gives:
        palette: the colormap as a k x 3 array of RGB colors
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in range(0, num_classes):
        label = k
        i = 0
        while label:
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    return palette

def color_seg(seg, palette):
    """
    Replace classes with their colors.

    Takes:
        seg: H x W segmentation image of class IDs
    Gives:
        H x W x 3 image of class colors
    """
    return palette[seg.flat].reshape(seg.shape + (3,))

def vis_seg(img, seg, palette, alpha=0.5):
    """
    Visualize segmentation as an overlay on the image.

    Takes:
        img: H x W x 3 image in [0, 255]
        seg: H x W segmentation image of class IDs
        palette: K x 3 colormap for all classes
        alpha: opacity of the segmentation in [0, 1]
    Gives:
        H x W x 3 image with overlaid segmentation
    """
    vis = np.array(img, dtype=np.float32)
    mask = seg > 0
    vis[mask] *= 1. - alpha
    vis[mask] += alpha * palette[seg[mask].flat]
    vis = vis.astype(np.uint8)
    return vis

def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def addTransparency(img, img_blender, factor = 0.3 ):
    img = img.convert('RGBA')
    print(img.size, img_blender.size)
    img = Image.blend(img_blender, img, factor)
    return img

"""
overall miou:  0.7459056023217526
overall class miou:  [0.90847889 0.6136272  0.29012716 0.60662176 0.64203647 0.7250848
 0.80822231]
"""

def evaluate_generator(Features = False):
    home_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir",help="A directory containing img (Images) \
                        and cls (GT Segmentation) folder")
    parser.add_argument("snapshot",help="Snapshot with the saved model")
    parser.add_argument("--val_orig", help="Do Inference on original size image.\
                        Otherwise, crop to 320x320 like in training ",action='store_true')
    parser.add_argument("--norm",help="Normalize the test images",\
                        action='store_true')
    args = parser.parse_args()

    sub_classes = ('plane', 'bike', 'person', 'car', 'dog', 'cat', 'background')

    # print(args.val_orig, args.norm)
    if args.val_orig:
        img_transform = transforms.Compose([ToTensor()])
        if args.norm:
            img_transform = transforms.Compose([ToTensor(),NormalizeOwn(dataset='voc')])
        label_transform = transforms.Compose([ToTensorLabel()]) # IgnoreLabelClass(),
        co_transform = transforms.Compose([ResizedImage3((321,321))])

        testset = Voc(home_dir, args.dataset_dir, n_classes=7, img_transform=img_transform, \
            label_transform = label_transform,co_transform=co_transform,train_phase=False)
        testloader = DataLoader(testset, batch_size=1)
    else:
        img_transform = transforms.Compose([ZeroPadding(),ToTensor()])
        if args.norm:
            img_transform = img_transform = transforms.Compose([ZeroPadding(),ToTensor(),NormalizeOwn(dataset='voc')])
        label_transform = transforms.Compose([ToTensorLabel()]) # IgnoreLabelClass(),

        testset = Voc(home_dir, args.dataset_dir, n_classes=7,img_transform=img_transform, \
            label_transform=label_transform,train_phase=False)
        testloader = DataLoader(testset, batch_size=1)

    # generator = encoder.resnet101()
    generator = deeplab.ResDeeplab(backbone='resnet', num_classes=6, has_clf=True)
    # generatro = fcn.FCN8s_soft()
    # generator = unet.AttU_Net(output_ch=3, Reconstruct=False,Aspp=False, Centroids=Centroids, RGB=RGB)
    print(args.snapshot)
    assert(os.path.isfile(args.snapshot))
    snapshot = torch.load(args.snapshot)

    # saved_net = {k.partition('module.')[2]: v for i, (k,v) in enumerate(snapshot['state_dict'].items())}
    generator_dict = generator.state_dict()
    saved_net = {k.partition('module.')[2]: v for i, (k,v) in enumerate(snapshot['state_dict'].items()) if k.partition('module.')[2] in generator_dict}
    generator_dict.update(saved_net)
    print('Snapshot Loaded')
    generator.load_state_dict(saved_net)
    generator.eval().cuda()
    # generator = nn.DataParallel(generator).cuda()
    print('Generator Loaded')
    n_classes = 7
    crf = False
    gts, preds, clustering, gtc = [], [], [], []
    acc_seg, rec_seg, prec_seg = [], [], []
    acc_clu, rec_clu, prec_clu = [], [], []
    acc_w, rec_w, prec_w = [], [], []

    print('Prediction Goint to Start')
    colorize = VOCColorize()
    palette = make_palette(n_classes)
    # palette[0] = np.array([64, 64, 128])
    IMG_DIR = osp.join(args.dataset_dir, 'VOCSUB/JPEGImages')
    # IMG_DIR = osp.join(args.dataset_dir, 'BioFouling/test_img')
    # TODO: Crop out the padding before prediction
    miou_val, cls_iu_val = [], []


    for img_id, (img, img_org, elabel, name) in enumerate(testloader):
        matting_name = os.path.join('results/closed_form/voc_debug/matting', '{}.png'.format(name[0]))
        filename = os.path.join('results/closed_form/voc_debug', '{}.png'.format(name[0]))
        activation = {}
        print("Generating Predictions for Image {}".format(name[0]))
        gt_mask = elabel.numpy()[0]

        img = Variable(img.cuda())
        # print(img.size())
        # img.cpu().numpy()[0]
        img_path = osp.join(IMG_DIR, name[0]+'.png')
        # img_pil = Image.open(img_path)
        # img_pil = img_pil.resize((513,513), Image.ANTIALIAS)
        img_array = cv2.imread(img_path)
        img_array = cv2.resize(img_array, (321,321), interpolation = cv2.INTER_AREA) 
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # out_pred_map = generator(img)
        # soft_pred = nn.Softmax2d()(out_pred_map)
        # org_soft = soft_pred.data.cpu().numpy()[0]
        # pred_hard = np.argmax(org_soft,axis=0).astype(np.uint8)

        out_pred_map, clf_preds = generator(img)
        alphas = np.zeros((n_classes+1, 321, 321))
        # n_classes = out_pred_map.size()[1]
        for c in range(n_classes-1):
            if clf_preds[0][c] >= 0.5:
                # print(out_pred_map.size())
                print("class {}: ".format(sub_classes[c]), clf_preds[0][c])
                solution = out_pred_map[0][c].detach().cpu().numpy().reshape(-1, 1)
                solution = solution.reshape(321,321)
                alpha = np.minimum(np.maximum(solution, 0), 1)
                seg = alpha.copy()
                seg[alpha >0.5] = 1
                seg[alpha<=0.5] = 0
                alphas[c+1] = seg
                
                cv2.imwrite(matting_name[0:-4]+"_alpha_{}.png".format(sub_classes[c]), alpha*255)
                alpha = 1 - alpha
                output = (alpha * 255.0).astype(np.uint8)
                b_channel, g_channel, r_channel = cv2.split(img_array)
                img_BGRA = cv2.merge((r_channel, g_channel, b_channel, output))
                cv2.imwrite(matting_name[0:-4] + "_{}.png".format(sub_classes[c]), img_BGRA)
        pred_hard = np.argmax(alphas,axis=0).astype(np.uint8)
        print(np.unique(pred_hard))
        # for gt_, pred_ in zip(gt_mask, pred_hard):
        #     gts.append(gt_)
        #     preds.append(pred_)

        miou, cls_iu, _ = scores(gt_mask, pred_hard, n_class = n_classes)
        print("miou: ", miou, ", cls_iu: ", cls_iu)
        miou_val.append(miou)
        cls_iu_val.append([cls_iu[0], cls_iu[1], cls_iu[2], cls_iu[3], cls_iu[4], cls_iu[5], cls_iu[6]])
        masked_im = Image.fromarray(vis_seg(img_array, pred_hard, palette))
        masked_im.save('{}.png'.format(filename[0:-4]))
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        # input('s')
    miou_val = np.array(miou_val)
    cls_iu_val = np.array(cls_iu_val)
    cls_iu_val = np.nan_to_num(cls_iu_val)
    print("overall miou: ", miou_val.mean())
    print("overall class miou: ", cls_iu_val.sum(axis=0)/np.count_nonzero(cls_iu_val, axis=0))



if __name__ == '__main__':
    evaluate_generator()