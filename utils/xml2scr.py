import os, random
import os.path as osp
import numpy as np
import xml.etree.ElementTree as ET
import cv2
import shutil

import PIL.Image as Image
import seaborn as sns
import matplotlib.pyplot as plt
import vis

class pascal_voc():
	def __init__(self, path, out_w, out_h, n_classes):
		self.root = path
		self._classes = ('empty', # always index 0
						 'plane', 'bike', 'bird', 'boat',
						 'bottle', 'bus', 'car', 'cat', 'chair',
						 'cow', 'table', 'dog', 'horse',
						 'motorbike', 'person', 'plant',
						 'sheep', 'sofa', 'train', 'monitor', 'background')
		self.out_w = out_w
		self.out_h = out_h
		self.n_classes = n_classes

		self.sub_name = ('empty', 'plane', 'bike', 'person', 'car', 'dog', 'cat', 'background')
		self.sub_dict = {'plane': 0, 'bike': 0, 'person': 0, 'car': 0, 'dog': 0, 'cat': 0, 'background': 0}
		# self.get_list()
		self.get_train_list()
		self.palette = vis.make_palette(len(self.sub_name))


	# def get_list(self):
	# 	self.xml_list = os.listdir(os.path.join(self.root, 'pascal_2012'))
	# 	random.shuffle(self.xml_list)

	def get_train_list(self):
		self.train_list = os.listdir(os.path.join('/media/data/VOCdevkit/VOCSUB', 'JPEGImages'))

	def xml_path_at(self, index):
		return osp.join(self.root, index)

	def extract_subset(self):
		# each category has 50
		for xml_name in self.xml_list:
			if self.sub_dict['plane'] > 20 and self.sub_dict['bike'] > 20 and \
			   self.sub_dict['person'] > 20 and self.sub_dict['car'] > 20 and \
			   self.sub_dict['dog'] > 20 and self.sub_dict['cat'] > 20:
				return
			else:
				print(xml_name)
				self.load_annotation(xml_name)
				print("plane: ", self.sub_dict['plane'], ', bike: ', self.sub_dict['bike'], 
					  ", person: ", self.sub_dict['person'], ', car: ', self.sub_dict['car'], 
					  ', dog: ', self.sub_dict['dog'], ', cat: ', self.sub_dict['cat'])
				print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

	def extract_test_subset(self):
		# each category has 50
		for xml_name in self.xml_list:
			name = xml_name[0:-4] + '.jpg'
			if name not in self.train_list:
				if self.sub_dict['plane'] > 10 and self.sub_dict['bike'] > 10 and \
				   self.sub_dict['person'] > 20 and self.sub_dict['car'] > 20 and \
				   self.sub_dict['dog'] > 20 and self.sub_dict['cat'] > 20:
					return
				else:
					print(xml_name)
					self.load_annotation(xml_name)
					print("plane: ", self.sub_dict['plane'], ', bike: ', self.sub_dict['bike'], 
						  ", person: ", self.sub_dict['person'], ', car: ', self.sub_dict['car'], 
						  ', dog: ', self.sub_dict['dog'], ', cat: ', self.sub_dict['cat'])
					print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

	def load_annotation(self, index):
		is_save = False

		# imgname = os.path.join(self.root, 'JPEGImages', index[0:-4] + '.png')
		filename = os.path.join('/media/data/VOCdevkit/VOC2012', 'pascal_2012', index[0:-4] + '.xml')
		dstname = os.path.join(self.root, 'SCRLabels', index[0:-4] + '.png')
		# visname = os.path.join(self.root, 'Vis', index[0:-4] + '.png')

		tree = ET.parse(filename)
		objs = tree.findall('size')
		for ix, obj in enumerate(objs):
			w = int(obj.find('width').text)
			h = int(obj.find('height').text)
		img_scr = np.zeros((h, w))
		# img_scr.fill(128)
		objs = tree.findall('polygon')
		num_objs = len(objs)

		# Load object bounding boxes into a data frame.
		for ix, obj in enumerate(objs):
			tag = obj.find('tag').text
			if tag not in self.sub_name:
				tag = 'background'
				ind = self.sub_name.index(tag)
			elif tag != 'background':
				is_save = True
				ind = self.sub_name.index(tag)
			else:
				ind = self.sub_name.index(tag)
			print(tag, ind)
			self.sub_dict[tag] += 1
			pts = obj.findall('point')
			# print("number of pts: ", len(pts))
			for _, pt in enumerate(pts):
				x = int(pt.find('X').text)
				y = int(pt.find('Y').text)
				if x>=w or x<0 or y>=h or y<0:
					continue
				img_scr[y][x] = ind
		# cv2.imwrite(dstname, img_scr)
		kernel = np.ones((3,3),np.uint8)
		img_scr = cv2.dilate(img_scr,kernel, iterations = 3)
		vis_scr = Image.fromarray(img_scr.copy())
		vis_scr = np.array(vis_scr.resize((self.out_w, self.out_h))).astype(np.uint8)

		img_scr[img_scr == 0] = 128
		img_scr[img_scr == 21] = 0
		img_pil = Image.fromarray(img_scr)
		img_scr = np.array(img_pil.resize((self.out_w, self.out_h)))
		if is_save:
			cv2.imwrite(dstname, img_scr)
			print(np.unique(img_scr))

		# show scribble annotation
		# img_array = cv2.imread(imgname)
		# img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
		# print(img_array.shape, vis_scr.shape, np.unique(vis_scr))
		# masked_im = Image.fromarray(vis.vis_seg(img_array, vis_scr, self.palette))
		# masked_im.save(visname)
		return

	def vis_scribble(self, index):
		imgname = os.path.join(self.root, 'JPEGImages', index[0:-4] + '.png')
		scrname = os.path.join(self.root, 'SCRLabels', index[0:-4] + '.png')
		visname = os.path.join(self.root, 'Vis', index[0:-4] + '.png')
		img_array = cv2.imread(imgname)
		img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
		img_scr = cv2.imread(scrname, 0)
		img_scr[img_scr == 0] 	= 21
		img_scr[img_scr == 128]	= 0
		masked_im = Image.fromarray(vis.vis_seg(img_array, img_scr, self.palette))
		masked_im.save(visname)



def test():
	voc = pascal_voc('/media/data/VOCdevkit/VOCSUB', 513, 513, 20)
	for name in voc.train_list:
		# name = "2009_000409.xml"
		print(name)
		voc.load_annotation(name)
		voc.vis_scribble(name)
		print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
		# input('s')






if __name__ == '__main__':
	test()