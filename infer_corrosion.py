from datasets.corrosion import Corrosion
# from datasets.biofouling import Biofouling
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

def vis_seg(img, seg, palette, alpha):
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
    print(vis.shape, mask.shape, alpha.shape)
    # vis[mask] *= 1. - alpha
    print(vis[mask].shape, alpha[mask].shape)
    vis[mask][:,0] *= alpha[mask]
    vis[mask][:,1] *= alpha[mask]
    vis[mask][:,2] *= alpha[mask]
    # vis[mask] += alpha * palette[seg[mask].flat]
    vis[mask] += alpha[mask] * palette[1]
    vis = vis.astype(np.uint8)
    return vis


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

    # print(args.val_orig, args.norm)
    if args.val_orig:
        img_transform = transforms.Compose([ToTensor()])
        if args.norm:
            img_transform = transforms.Compose([ToTensor(),NormalizeOwn(dataset='corrosion')])
        label_transform = transforms.Compose([IgnoreLabelClass(),ToTensorLabel()])
        co_transform = transforms.Compose([ResizedImage3((513,513))])

        testset = Corrosion(home_dir, args.dataset_dir,img_transform=img_transform, \
            label_transform = label_transform,co_transform=co_transform,train_phase=False)
        testloader = DataLoader(testset, batch_size=1)
    else:
        img_transform = transforms.Compose([ZeroPadding(),ToTensor()])
        if args.norm:
            img_transform = img_transform = transforms.Compose([ZeroPadding(),ToTensor(),NormalizeOwn(dataset='corrosion')])
        label_transform = transforms.Compose([IgnoreLabelClass(),ToTensorLabel()])

        testset = Corrosion(home_dir,args.dataset_dir,img_transform=img_transform, \
            label_transform=label_transform,train_phase=False)
        testloader = DataLoader(testset, batch_size=1)

    # generator = encoder.resnet101()
    generator = deeplab.ResDeeplab(num_classes=1)
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
    n_classes = 3
    crf = False
    gts, preds, clustering, gtc = [], [], [], []
    acc_seg, rec_seg, prec_seg = [], [], []
    acc_clu, rec_clu, prec_clu = [], [], []
    acc_w, rec_w, prec_w = [], [], []

    print('Prediction Goint to Start')
    colorize = VOCColorize()
    palette = make_palette(n_classes)
    palette[1] = np.array([0, 0, 128])
    # print(palette)
    IMG_DIR = osp.join(args.dataset_dir, 'corrosion/JPEGImages')
    # TODO: Crop out the padding before prediction
    for img_id, (img, trimap, img_org, name) in enumerate(testloader):
        print(name)
        filename = os.path.join('results/closed_form/corrosion', '{}.jpg'.format(name[0]))
        activation = {}
        print("Generating Predictions for Image {}".format(name[0]))

        # sp_array = parse_json_superpixel(name[0])

        img = Variable(img.cuda())
        print(img.size())
        # img.cpu().numpy()[0]
        img_path = osp.join(IMG_DIR, name[0]+'.jpg')
        # img_pil = Image.open(img_path)
        # img_pil = img_pil.resize((513,513), Image.ANTIALIAS)
        img_array = cv2.imread(img_path)
        img_array = cv2.resize(img_array, (513,513), interpolation = cv2.INTER_AREA) 
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        out_pred_map = generator(img)
        print(out_pred_map.size())
        solution = out_pred_map[0][0].detach().cpu().numpy().reshape(-1, 1)
        solution = solution.reshape(513,513)
        sns.heatmap(solution)
        plt.show()
        input('s')
        # print(out_pred_map.max(), out_pred_map.min())
        # out_pred_np = out_pred_map[0].detach().cpu().numpy()
        alpha = np.minimum(np.maximum(solution, 0), 1)
        # alpha = (out_pred_np - out_pred_np.min()) / (out_pred_np.max() - out_pred_np.min()) 
        # print(alpha.shape, np.unique(alpha))
        cv2.imwrite(filename[0:-4]+"_alpha.png", alpha*255)

        # create vis
        # vis = np.zeros((img_array.shape[0], img_array.shape[1], 3)).astype(np.uint8)
        # vis[:,:,-1] = 255

        alpha = 1 - alpha
        # vis_seg(img, seg, palette, alpha)
        # out = Image.fromarray(vis_seg(img_array, alpha, palette, alpha))
        # cv2.imwrite(filename[0:-4]+".png", out)

        
        output = (alpha * 255.0).astype(np.uint8)
        # output_pil = Image.fromarray(output)
        b_channel, g_channel, r_channel = cv2.split(img_array)
        print(b_channel.shape, g_channel.shape, r_channel.shape, output.shape)
        img_BGRA = cv2.merge((r_channel, g_channel, b_channel, output))
        # img_BGRA = cv2.addWeighted(img_array,0.4,foreground,0.1,0)
        print(img_BGRA.shape)
        cv2.imwrite(filename[0:-4]+".png", img_BGRA)
        
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        # input('s')



if __name__ == '__main__':
    evaluate_generator()