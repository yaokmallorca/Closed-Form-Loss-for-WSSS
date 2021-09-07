from __future__ import unicode_literals
import random
import numpy as np
from collections import OrderedDict
import torch

# from datasets.pascalvoc import PascalVOC
# from datasets.biofouling_full import Biofouling
from datasets.cityscape import CityScape
# import generators.deeplabv2 as deeplab
# import generators.deeplabv3 as deeplab
# import generators.encoder as encoder
import generators.erfnet as erfnet

from torchvision import transforms
from torchvision.transforms import ToTensor,Compose
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from utils.transforms import IgnoreLabelClass, ToTensorLabel, NormalizeOwn,ZeroPadding, OneHotEncode, ToFloatTensorLabel
from utils.transforms import RandomSizedCrop3, ResizedImage3
from utils.lr_scheduling import poly_lr_scheduler, poly_lr_step_scheduler
from utils.validate import val, val_matting
from utils.helpers import pascal_palette_invert
from utils.metrics import scores
from utils.laplacian import ClosedFormLoss
from utils.loss import ScribbleLoss, DiceLoss

# plot heatmap
import matplotlib.pyplot as plt
import seaborn as sns

from functools import reduce
import os
import os.path as osp
import cv2
import argparse
import PIL.Image as Image
import datetime 

# from utils.log import setup_logging, ResultsLog, save_checkpoint, export_args_namespace

Reconstruct = False
ASPP = False
G_RGB = False
G_Centroids = False

home_dir = os.path.dirname(os.path.realpath(__file__))
colnames = ['epoch', 'iter', 'LD', 'LD_fake', 'LD_real', 'LG', 'LG_ce', 'LG_adv', 'LG_semi']
# DATASET_PATH = '/media/data/seg_dataset/BioFouling/JPEGImages'
def parse_args():

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("prefix",
                        help="Prefix to identify current experiment")

    parser.add_argument("dataset_dir", default='/media/data/CityScapes', 
                        help="A directory containing img (Images) and cls (GT Segmentation) folder")

    parser.add_argument("--mode", choices=('base','close'),default='close',
                        help="base (baseline),close")

    parser.add_argument("--lam_adv",default=0.5,
                        help="Weight for Adversarial loss for Segmentation Network training")

    parser.add_argument("--lam_semi",default=0.1,
                        help="Weight for Semi-supervised loss")

    parser.add_argument("--nogpu",action='store_true',
                        help="Train only on cpus. Helpful for debugging")

    parser.add_argument("--max_epoch",default=400,type=int,
                        help="Maximum iterations.")

    parser.add_argument("--start_epoch",default=0,type=int,
                        help="Resume training from this epoch")

    parser.add_argument("--snapshot", default='snapshots',
                        help="Snapshot to resume training")

    parser.add_argument("--snapshot_dir",default=os.path.join(home_dir,'data','snapshots'),
                        help="Location to store the snapshot")

    parser.add_argument("--batch_size",default=8,type=int, # 10
                        help="Batch size for training")

    parser.add_argument("--val_orig",action='store_true',
                        help="Do Inference on original size image. Otherwise, crop to 320x320 like in training ")

    parser.add_argument("--d_label_smooth",default=0.1,type=float,
                        help="Label smoothing for real images in Seg network")

    parser.add_argument("--no_norm",action='store_true',
                        help="No Normalizaion on the Images")

    parser.add_argument("--init_net",choices=('imagenet','mscoco', 'unet'),default='mscoco',
                        help="Pretrained Net for Segmentation Network")

    parser.add_argument("--g_lr",default=1e-3, type=float, # 1e-5
                        help="lr for generator")

    parser.add_argument("--seed",default=3000,type=int,
                        help="Seed for random numbers used in semi-supervised training")

    parser.add_argument("--wait_semi",default=0,type=int,
                        help="Number of Epochs to wait before using semi-supervised loss")

    parser.add_argument("--split",default=1.0,type=float) # 0.5
    # args = parser.parse_args()

    parser.add_argument("--lr_step", default='999999999999999,999999999999999', type=str, 
                        help='Steps for decreasing learning rate')
    args = parser.parse_args()
    return args



def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    # print("ignore_mask: ", ignore_mask.shape)
    D_label = np.ones(ignore_mask.shape)*label
    # print("D_label: ", D_label.shape)
    D_label[ignore_mask] = 255 # 255
    # print("D_label: ", D_label.shape)
    D_label = Variable(torch.FloatTensor(D_label)).cuda(args.gpu)
    # print("D_label: ", D_label.size())
    return D_label

def one_hot_float(soft_label):
    soft1 = soft_label
    soft0 = 1. - soft_label
    soft0, soft1 = soft0[np.newaxis,:,:], soft1[np.newaxis,:,:]
    one_hot = np.concatenate([soft0, soft1], axis=0)
    return one_hot

def label_smooth(target, epsilon=0.1, n_classes=2):
    batch_size, h, w = target.size()
    one_hot = torch.zeros(batch_size, n_classes, h, w).cuda()
    one_hot.scatter_(1, target.unsqueeze(1), 1)
    # soft_target = target.float()
    return ((1. - epsilon)*one_hot + (epsilon/n_classes))
'''
    Use PreTrained Model for Initial Weights
'''
def init_weights(model,init_net):
    if init_net == 'imagenet':
        # Pretrain on ImageNet
        inet_weights = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        del inet_weights['fc.weight']
        del inet_weights['fc.bias']
        state = model.state_dict()
        state.update(inet_weights)
        model.load_state_dict(state)
    elif init_net == 'mscoco':
        # TODO: Upload the weights somewhere to use load.url()
        filename = os.path.join(home_dir,'data','MS_DeepLab_resnet_pretrained_COCO_init.pth')
        assert(os.path.isfile(filename))
        saved_net = torch.load(filename)
        new_state = model.state_dict()
        saved_net = {k.partition('Scale.')[2]: v for i, (k,v) in enumerate(saved_net.items())}
        new_state.update(saved_net)
        model.load_state_dict(new_state)
    elif init_net == 'unet':
        unet.init_weights(model, init_type='kaiming')


def snapshot(model,valoader,epoch,best_miou,snapshot_dir,prefix):
    miou, cls_iu = val_matting(model,valoader,nclass=10,Centroids=G_Centroids)
    snapshot = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }
    if miou > best_miou:
        best_miou = miou
        torch.save(snapshot,os.path.join(snapshot_dir,'{}_closedform_best.pth.tar'.format(prefix)))
    elif epoch % 20 == 0:
        torch.save(snapshot,os.path.join(snapshot_dir,'{}_{}_closedform.pth.tar'.format(prefix, epoch)))
    return best_miou, miou, cls_iu


def mse_loss(input, target, ignored_index=128, reduction='mean'):
    mask = target == ignored_index
    target = target/255.0
    out = (input[~mask]-target[~mask])**2
    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out

def train_close(generator,steps,optimG,trainloader,valoader,args):
    nt = datetime.datetime.now()
    reg = False
    closed_form_loss = ClosedFormLoss(n_classes=9, img_w=320, img_h=160, 
        reg=reg, trimap_confidence=100, has_clf=True)
    # closed_form_loss = ClosedFormLoss(n_classes=9, img_w=321, img_h=321, 
    #     reg=reg, trimap_confidence=100, has_clf=True)
    clf_loss = nn.BCELoss()
    lambda_loss = 1
    best_miou = -1

    for epoch in range(args.start_epoch,args.max_epoch+1):
        generator.train()
        #  (img, mask, cmask, _, _, img_org)
        # for batch_id, (img, mask, cmask, _, emask, img_names) in enumerate(trainloader):
        for batch_id, (img, trimaps, img_org, clf_labels, img_names) in enumerate(trainloader):
            if args.nogpu:
                img = Variable(img)
            else:
                # img = Variable(img.cuda())
                img, clf_labels, trimaps, img_org = Variable(img.cuda()), clf_labels.float().cuda(), trimaps, img_org
            itr = len(trainloader)*(epoch) + batch_id
            # print(img.size(), trimaps.size(), torch.unique(trimaps), img_org.size())
            # print("clf_labels", clf_labels)
            cpmap, clf_preds = generator(img, has_clf=True)
            # cpmap, clf_preds = generator(img)
            clfloss = clf_loss(clf_preds, clf_labels)

            closeloss = closed_form_loss(cpmap.cpu(), img_org, trimaps, clf_preds=clf_preds)
            loss = closeloss + clfloss
            for param_group in optimG.param_groups:
                curr_lr = param_group['lr']
            # optimG = poly_lr_step_scheduler(optimG, curr_lr, itr,steps)
            optimG = poly_lr_scheduler(optimG, curr_lr, itr)
            optimG.zero_grad()
            loss.backward()
            optimG.step()

            if reg:
                log_str = "[{}][{}][{:.1E}]Loss: {:0.8f}, {:0.8f}, {:0.8f},"\
                            .format(epoch,itr,curr_lr,loss.data, closeloss.data, regloss.data)
            else:
                log_str = "[{}][{}][{:.1E}]Loss: {:0.8f}, {:0.8f}".format(epoch,itr,curr_lr,closeloss.data,clfloss.data)
            # log_str = "[{}][{}][{:.1E}]Loss: {:0.8f}, {:0.8f}".format(epoch,itr,curr_lr,closeloss.data,mseloss.data)
            print(log_str)
            writer.add_scalar("Loss/train", closeloss, epoch)
            writer.add_scalar("Loss/train", clfloss, epoch)
            writer.add_scalar("Loss/train", loss, epoch)
            # log_f.write(log_str+'\n')
        best_miou, curr_miou, cls_iu = snapshot(generator,valoader,epoch,best_miou,args.snapshot_dir,args.prefix)
        print("validation miou: ", curr_miou, ", best miou: ", best_miou, ", cls_iu: ", cls_iu)
        writer.add_scalar("Loss/train", curr_miou, epoch)


def main():

    args = parse_args()

    random.seed(0)
    torch.manual_seed(0)
    if not args.nogpu:
        torch.cuda.manual_seed_all(0)

    if args.no_norm:
        imgtr = [ToTensor()]
    else:
        imgtr = [ToTensor(),NormalizeOwn(dataset='cityscape')]

    if len(args.lr_step) != 0:
        steps = list(map(lambda x: int(x), args.lr_step.split(',')))

    # softmax
    labtr = [ToTensorLabel()] # IgnoreLabelClass(), ToFloatTensorLabel
    # cotr = [ResizedImage5((320,320))]
    cotr = [ResizedImage3((160,320))] #    RandomSizedCrop3

    print("dataset_dir: ", args.dataset_dir)
    trainset_l = CityScape(home_dir,args.dataset_dir,n_classes=9,img_transform=Compose(imgtr), 
                           label_transform=Compose(labtr),co_transform=Compose(cotr),
                           split=args.split,labeled=True, label_correction=True)
    trainloader_l = DataLoader(trainset_l,batch_size=args.batch_size,shuffle=True,
                               num_workers=2,drop_last=True)
    if args.split != 1:
        trainset_u = CityScape(home_dir,args.dataset_dir,n_classes=9,img_transform=Compose(imgtr), 
                               label_transform=Compose(labtr),co_transform=Compose(cotr),
                               split=args.split,labeled=False, label_correction=True)
        trainloader_u = DataLoader(trainset_l,batch_size=args.batch_size,shuffle=True,
                                   num_workers=2,drop_last=True)

    #########################
    # Validation Dataloader #
    ########################
    if args.val_orig:
        if args.no_norm:
            imgtr = [ZeroPadding(),ToTensor()]
        else:
            imgtr = [ZeroPadding(),ToTensor(),NormalizeOwn()]
        # softmax
        labtr = [ToFloatTensorLabel()] # IgnoreLabelClass() ToTensorLabel
        cotr = []
    else:
        if args.no_norm:
            imgtr = [ToTensor()]
        else:
            imgtr = [ToTensor(),NormalizeOwn()]
        # softmax
        labtr = [ToTensorLabel()] #  ToFloatTensorLabel
        # cotr = [ResizedImage5((320,320))]
        cotr = [ResizedImage3((160,320))]

    valset = CityScape(home_dir,args.dataset_dir, n_classes=9, img_transform=Compose(imgtr), \
        label_transform = Compose(labtr),co_transform=Compose(cotr),train_phase=False)
    valoader = DataLoader(valset,batch_size=1)

    #############
    # GENERATOR #
    #############
    # generator = deeplabv2.ResDeeplab(Reconstruct=True)

    # softmax generator: in_chs=3, out_chs=2
    # generator = deeplab.ResDeeplab(backbone='xception', num_classes=1)
    # generator = fcn8s.FCN8s_softAtOnce()
    generator = erfnet.ERFNet(num_classes=9)
    # generator = deeplab.ResDeeplab(backbone='resnet', num_classes=9, has_clf=True)

    if osp.isfile(args.snapshot):
        print("load checkpoint => ", args.snapshot)
        checkpoint = torch.load(args.snapshot)
        generator_dict = generator.state_dict()
        args.start_epoch = checkpoint['epoch']
        saved_net = {k.partition('module.')[2]: v for i, (k,v) in enumerate(checkpoint['state_dict'].items()) if k.partition('module.')[2] in generator_dict}
        generator_dict.update(saved_net)
        generator.load_state_dict(saved_net)
    # else:
    #     init_weights(generator,args.init_net)
    print("start_epoch: ", args.start_epoch)


    optimG = optim.Adam(filter(lambda p: p.requires_grad, \
            generator.parameters()),args.g_lr, [0.9, 0.999])

    if not args.nogpu:
        generator = nn.DataParallel(generator).cuda()

    if args.mode == 'base':
        # train_base(generator,optimG,trainloader_l,valoader,args)
        train_dice(generator, steps, optimG, trainloader_l, valoader,args)
    elif args.mode == 'close':
        train_close(generator, steps, optimG, trainloader_l, valoader,args)
    else:
        # train_semir(generator,discriminator,optimG,optimD,trainloader_l,valoader,args)
        print("training mode incorrect")

if __name__ == '__main__':
    main()
