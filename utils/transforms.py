import torch
from PIL import Image
import numpy as np
import math
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter

"""
'corrosion': {
    'mean': np.array([0.45942432, 0.43057796, 0.32076004],float),
    'std': np.array([0.14271824, 0.13926049, 0.11461112],float)
}
subvoc
JPEGImages : normMean = [0.4517059  0.43470722 0.40886918]
JPEGImages : normstdevs = [0.23672307 0.23163801 0.2359839 ]
"""

stats = {
    'voc': {
        'mean': np.array([0.4517059,0.43470722,0.40886918],float),
        'std': np.array([0.23672307,0.23163801,0.2359839],float)
    }, 
    'corrosion': {
        'mean': np.array([0.4855246,  0.45945138, 0.3439404],float),
        'std': np.array([0.134174, 0.13275488, 0.11363208],float)
    },
    'box': {
        'mean': np.array([0.50441545, 0.504785, 0.5031589],float),
        'std': np.array([0.19157167, 0.18259047, 0.1844328],float)
    },
    'bio' : {
        'mean': np.array([0.38009313, 0.39160565, 0.25242192],float),
        'std': np.array([0.12935428, 0.16977909, 0.1502691],float)
    }
}

class OneHotEncode(object):
    """
        Takes a Tensor of size 1xHxW and create one-hot encoding of size nclassxHxW
    """
    def __init__(self,nclass=2):
        self.nclass = nclass

    def __call__(self,label):
        label_a = np.array(transforms.ToPILImage()(label.byte().unsqueeze(0)),np.uint8)

        ohlabel = np.zeros((self.nclass,label_a.shape[0],label_a.shape[1])).astype(np.uint8)

        for c in range(self.nclass):
            ohlabel[c:,:,:] = (label_a == c).astype(np.uint8)

        # # Do Some assertion
        # print("Assertion about to be made")
        # for c in range(self.nclass):
        #     for i in range(321):
        #         for j in range(321):
        #             if ohlabel[c][i][j] == 1:
        #                 assert(label_a[i][j] == c)

        return torch.from_numpy(ohlabel)

class OneHotEncode_smooth(object):
    """
        Takes a Tensor of size 1xHxW and create one-hot encoding of size nclassxHxW
    """
    def __init__(self,nclass=2):
        self.nclass = nclass

    def __call__(self,label):
        label_a = np.array(transforms.ToPILImage()(label.byte().unsqueeze(0)),np.uint8)

        ohlabel = np.zeros((self.nclass,label_a.shape[0],label_a.shape[1])).astype(np.float)

        for c in range(self.nclass):
            ohlabel[c,:,:][label_a == c] = 0.1
            ohlabel[c,:,:][label_a != c] = 0.9

        # # Do Some assertion
        # print("Assertion about to be made")
        # for c in range(self.nclass):
        #     for i in range(321):
        #         for j in range(321):
        #             if ohlabel[c][i][j] == 1:
        #                 assert(label_a[i][j] == c)

        return torch.from_numpy(ohlabel)


class NormalizeOwn(object):
    """
        Normalize the dataset to zero mean and unit standard deviation.
    """
    def __init__(self,dataset='voc'): # corrosion !!!!!!!!!!!!
        self.dataset = dataset

    def __call__(self,img):
        return transforms.Normalize(mean=stats[self.dataset]['mean'],std=stats[self.dataset]['std'])(img)

class IgnoreLabelClass(object):
    """
        Convert a label for a class to be ignored to some other class
    """
    def __init__(self,ignore=255,base=0):
        self.ignore = ignore
        self.base = base

    def __call__(self,label):
        return Image.eval(label,lambda p: self.base if p == self.ignore else p)

class ToTensorLabel(object):
    """
        Take a Label as PIL.Image with 'P' mode and convert to Tensor
    """
    def __init__(self,tensor_type=torch.LongTensor):
        self.tensor_type = tensor_type

    def __call__(self,label):
        label = np.array(label,dtype=np.uint8)
        label = torch.from_numpy(label).type(self.tensor_type)

        return label

class ToFloatTensorLabel(object):
    """
        Take a Label as PIL.Image with 'P' mode and convert to Tensor
    """
    def __init__(self,tensor_type=torch.FloatTensor):
        self.tensor_type = tensor_type

    def __call__(self,label):
        label = np.array(label,dtype=np.float32)
        label = torch.from_numpy(label).type(self.tensor_type)
        return label


class ZeroPadding(object):
    """
        Add zero padding to the image to right and bottom to resize it.
        Needed at test phase to make all images 513x513.

        Input: PIL Image with 'RGB' mode
        Output: Zero padded PIL image with agin with 'RGB' mode

    """
    def __init__(self,size=(513,513)):
        self.size = size


    def __call__(self,img):
        assert(img.size[0]<=self.size[0] and img.size[1] <= self.size[1])

        img_new = np.zeros((self.size[0],self.size[1],3),np.uint8)
        img_orig = np.array(img,np.uint8)
        img_new[:img_orig.shape[0],:img_orig.shape[1],:] = img_orig
        return img_new

class RandomGaussianBlur5(object):
    def __call__(self, data):
        img = data[0]
        label = data[1]
        elabel = data[2]
        clabel = data[3]
        img_org = data[4]
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return img, label, elabel, clabel, img_org

class RandomSizedCrop(object):
    """
        RandomSizedCrop for both the image and the label
    """
    def __init__(self,size,img_interpolation=Image.BILINEAR,label_interpolation=Image.NEAREST):
        self.size = size
        self.img_interpolation = img_interpolation
        self.label_interpolation = label_interpolation

    """
        Apply the random resized crop to both (img,label)
        Expects img,label to be PIL.Image objects
    """
    def __call__(self,data):
        img = data[0]
        label = data[1]
        for attempt in range(10):
            rand_scale = random.uniform(0.08,1.0)
            rand_aspect_ratio = random.uniform(3. / 4, 4. / 3)

            area = img.size[0]*img.size[1]
            target_area = rand_scale*area

            w = int(round(math.sqrt(target_area * rand_aspect_ratio)))
            h = int(round(math.sqrt(target_area / rand_aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                label = label.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                assert(label.size == (w,h))

                return img.resize(self.size, self.img_interpolation),label.resize(self.size,self.label_interpolation)
        #Add a fallback method
        img_scale = transforms.Scale(self.size[0],interpolation=self.img_interpolation)
        label_scale = transforms.Scale(self.size[0],interpolation=self.label_interpolation)
        crop = transforms.CenterCrop(self.size)
        return crop(img_scale(img)), crop(label_scale(label))

class RandomSizedCrop3(object):
    """
        RandomSizedCrop for both the image and the label
    """
    def __init__(self,size,img_interpolation=Image.BILINEAR,label_interpolation=Image.NEAREST):
        self.size = size
        self.img_interpolation = img_interpolation
        self.label_interpolation = label_interpolation

    """
        Apply the random resized crop to both (img,label)
        Expects img,label to be PIL.Image objects
    """
    def __call__(self,data):
        img = data[0]
        label = data[1]
        elabel = data[2]
        for attempt in range(10):
            rand_scale = random.uniform(0.08,1.0)
            rand_aspect_ratio = random.uniform(3. / 4, 4. / 3)

            area = img.size[0]*img.size[1]
            target_area = rand_scale*area

            w = int(round(math.sqrt(target_area * rand_aspect_ratio)))
            h = int(round(math.sqrt(target_area / rand_aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                label = label.crop((x1, y1, x1 + w, y1 + h))
                elabel = elabel.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                assert(label.size == (w,h))
                assert(elabel.size == (w,h))

                return img.resize(self.size, self.img_interpolation),label.resize(self.size,self.label_interpolation),elabel.resize(self.size,self.label_interpolation)
        #Add a fallback method
        img_scale = transforms.Resize(self.size[0],interpolation=self.img_interpolation) # Scale
        label_scale = transforms.Resize(self.size[0],interpolation=self.label_interpolation)
        elabel_scale = transforms.Resize(self.size[0],interpolation=self.label_interpolation)
        crop = transforms.CenterCrop(self.size)
        return crop(img_scale(img)), crop(label_scale(label)), crop(elabel_scale(elabel))

class RandomSizedCrop4(object):
    """
        RandomSizedCrop for both the image and the label
    """
    def __init__(self,size,img_interpolation=Image.BILINEAR,label_interpolation=Image.NEAREST):
        self.size = size
        self.img_interpolation = img_interpolation
        self.label_interpolation = label_interpolation

    """
        Apply the random resized crop to both (img,label)
        Expects img,label to be PIL.Image objects
    """
    def __call__(self,data):
        img = data[0]
        label = data[1]
        elabel = data[2]
        clabel = data[3]
        for attempt in range(10):
            rand_scale = random.uniform(0.08,1.0)
            rand_aspect_ratio = random.uniform(3. / 4, 4. / 3)

            area = img.size[0]*img.size[1]
            target_area = rand_scale*area

            w = int(round(math.sqrt(target_area * rand_aspect_ratio)))
            h = int(round(math.sqrt(target_area / rand_aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                label = label.crop((x1, y1, x1 + w, y1 + h))
                elabel = elabel.crop((x1, y1, x1 + w, y1 + h))
                clabel = clabel.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                assert(label.size == (w,h))
                assert(elabel.size == (w,h))
                assert(clabel.size == (w,h))

                return (img.resize(self.size, self.img_interpolation),
                        label.resize(self.size,self.label_interpolation),
                        elabel.resize(self.size,self.label_interpolation),
                        clabel.resize(self.size,self.label_interpolation))
        #Add a fallback method
        img_scale = transforms.Scale(self.size[0],interpolation=self.img_interpolation)
        label_scale = transforms.Scale(self.size[0],interpolation=self.label_interpolation)
        elabel_scale = transforms.Scale(self.size[0],interpolation=self.label_interpolation)
        clabel_scale = transforms.Scale(self.size[0],interpolation=self.label_interpolation)
        crop = transforms.CenterCrop(self.size)
        return crop(img_scale(img)), crop(label_scale(label)), crop(elabel_scale(elabel)), crop(clabel_scale(clabel))

class RandomSizedCrop5(object):
    """
        RandomSizedCrop for both the image and the label
    """
    def __init__(self,size,img_interpolation=Image.BILINEAR,label_interpolation=Image.NEAREST):
        self.size = size
        self.img_interpolation = img_interpolation
        self.label_interpolation = label_interpolation

    """
        Apply the random resized crop to both (img,label)
        Expects img,label to be PIL.Image objects
    """
    def __call__(self,data):
        img = data[0]
        label = data[1]
        elabel = data[2]
        clabel = data[3]
        img_org = data[4]
        for attempt in range(10):
            rand_scale = random.uniform(0.08,1.0)
            rand_aspect_ratio = random.uniform(3. / 4, 4. / 3)

            area = img.size[0]*img.size[1]
            target_area = rand_scale*area

            w = int(round(math.sqrt(target_area * rand_aspect_ratio)))
            h = int(round(math.sqrt(target_area / rand_aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                label = label.crop((x1, y1, x1 + w, y1 + h))
                elabel = elabel.crop((x1, y1, x1 + w, y1 + h))
                clabel = clabel.crop((x1, y1, x1 + w, y1 + h))
                img_org = img_org.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))
                assert(label.size == (w,h))
                assert(elabel.size == (w,h))
                assert(clabel.size == (w,h))
                assert(img_org.size == (w,h))

                return (img.resize(self.size, self.img_interpolation),
                        label.resize(self.size,self.label_interpolation),
                        elabel.resize(self.size,self.label_interpolation),
                        clabel.resize(self.size,self.label_interpolation),
                        img_org.resize(self.size,self.label_interpolation))
        #Add a fallback method
        img_scale = transforms.Scale(self.size[0],interpolation=self.img_interpolation)
        label_scale = transforms.Scale(self.size[0],interpolation=self.label_interpolation)
        elabel_scale = transforms.Scale(self.size[0],interpolation=self.label_interpolation)
        clabel_scale = transforms.Scale(self.size[0],interpolation=self.label_interpolation)
        img_org_scale = transforms.Scale(self.size[0],interpolation=self.label_interpolation)
        crop = transforms.CenterCrop(self.size)
        return crop(img_scale(img)), crop(label_scale(label)), crop(elabel_scale(elabel)), crop(clabel_scale(clabel)), crop(img_org_scale(img_org))


class ResizedImage(object):
    """
        RandomSizedCrop for both the image and the label  
    """
    def __init__(self,size,img_interpolation=Image.BILINEAR,label_interpolation=Image.NEAREST):
        self.size = size
        self.img_interpolation = img_interpolation
        self.label_interpolation = label_interpolation

    """
        Apply the random resized crop to both (img,label)
        Expects img,label to be PIL.Image objects
    """
    def __call__(self,data):
        img = data[0]
        label = data[1]
        #Add a fallback method
        img_scale = transforms.Scale(self.size[0],interpolation=self.img_interpolation)
        label_scale = transforms.Scale(self.size[0],interpolation=self.label_interpolation)
        # crop = transforms.CenterCrop(self.size)
        return img_scale(img), label_scale(label)

class ResizedImage3(object):
    """
        RandomSizedCrop for both the image and the label  
    """
    def __init__(self,size,img_interpolation=Image.BILINEAR,label_interpolation=Image.NEAREST):
        self.size = size
        self.img_interpolation = img_interpolation
        self.label_interpolation = label_interpolation

    """
        Apply the random resized crop to both (img,label)
        Expects img,label to be PIL.Image objects
    """
    def __call__(self,data):
        img = data[0]
        label = data[1]
        elabel = data[2]
        #Add a fallback method
        img_scale = transforms.Scale(self.size,interpolation=self.img_interpolation)
        label_scale = transforms.Scale(self.size,interpolation=self.label_interpolation)
        elabel_scale = transforms.Scale(self.size,interpolation=self.label_interpolation)
        # crop = transforms.CenterCrop(self.size)
        return img_scale(img), label_scale(label), elabel_scale(elabel)

class ResizedImageTrimap(object):
    """
        RandomSizedCrop for both the image and the label  
    """
    def __init__(self,size,img_interpolation=Image.BILINEAR,label_interpolation=Image.NEAREST):
        self.size = size
        self.img_interpolation = img_interpolation
        self.label_interpolation = label_interpolation

    """
        Apply the random resized crop to both (img,label)
        Expects img,label to be PIL.Image objects
    """
    def __call__(self,data):
        img = data[0]
        label = data[1]
        elabel = data[2]
        n_classes = label[0].shape[0]

        #Add a fallback method
        img_scale = transforms.Scale(self.size,interpolation=self.img_interpolation)
        label_scale = transforms.Scale(self.size,interpolation=self.label_interpolation)
        elabel_scale = transforms.Scale(self.size,interpolation=self.label_interpolation)
        # crop = transforms.CenterCrop(self.size)

        return img_scale(img), label_scale(label), elabel_scale(elabel)

class ResizedImage4(object):
    """
        RandomSizedCrop for both the image and the label  
    """
    def __init__(self,size,img_interpolation=Image.BILINEAR,label_interpolation=Image.NEAREST):
        self.size = size
        self.img_interpolation = img_interpolation
        self.label_interpolation = label_interpolation

    """
        Apply the random resized crop to both (img,label)
        Expects img,label to be PIL.Image objects
    """
    def __call__(self,data):
        img = data[0]
        label = data[1]
        elabel = data[2]
        clabel = data[3]
        #Add a fallback method
        img_scale       = transforms.Scale((self.size[0], self.size[0]),interpolation=self.img_interpolation)
        label_scale     = transforms.Scale((self.size[0], self.size[0]),interpolation=self.label_interpolation)
        elabel_scale    = transforms.Scale((self.size[0], self.size[0]),interpolation=self.label_interpolation)
        clabel_scale    = transforms.Scale((self.size[0], self.size[0]),interpolation=self.label_interpolation)
        # crop = transforms.CenterCrop(self.size)
        return img_scale(img), label_scale(label), elabel_scale(elabel), clabel_scale(clabel)

class ResizedImage5(object):
    """
        RandomSizedCrop for both the image and the label  
    """
    def __init__(self,size,img_interpolation=Image.BILINEAR,label_interpolation=Image.NEAREST):
        self.size = size
        self.img_interpolation = img_interpolation
        self.label_interpolation = label_interpolation

    """
        Apply the random resized crop to both (img,label)
        Expects img,label to be PIL.Image objects
    """
    def __call__(self,data):
        img = data[0]
        label = data[1]
        elabel = data[2]
        clabel = data[3]
        img_org = data[4]
        #Add a fallback method
        img_scale = transforms.Scale((self.size[0], self.size[0]),interpolation=self.img_interpolation)
        label_scale = transforms.Scale((self.size[0], self.size[0]),interpolation=self.label_interpolation)
        elabel_scale = transforms.Scale((self.size[0], self.size[0]),interpolation=self.label_interpolation)
        clabel_scale = transforms.Scale((self.size[0], self.size[0]),interpolation=self.label_interpolation)
        img_org_scale = transforms.Scale((self.size[0], self.size[0]),interpolation=self.label_interpolation)
        # crop = transforms.CenterCrop(self.size)
        return img_scale(img), label_scale(label), elabel_scale(elabel), clabel_scale(clabel), img_org_scale(img_org)
