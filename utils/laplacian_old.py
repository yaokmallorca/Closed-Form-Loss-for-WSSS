from __future__ import division
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.sparse
import scipy.sparse.linalg

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

def _rolling_block(A, block=(3, 3)):
	"""Applies sliding window to given matrix."""
	shape = (A.size()[0] - block[0] + 1, A.size()[1] - block[1] + 1) + block
	strides = (A.stride()[0], A.stride()[1]) + A.stride()
	return torch.as_strided(A, size=shape, stride=strides)

def compute_laplacian(img, mask=None, eps=10**(-7), win_rad=1):
	win_size = (win_rad * 2 + 1) ** 2
	h, w, d = img.size()
	c_h, c_w = h - 2 * win_rad, w - 2 * win_rad
	win_diam = win_rad * 2 + 1
	indsM = torch.arange(h*w).reshape(h,w)
	ravelImg = img.reshape(h*w, d)
	win_inds = _rolling_block(indsM, block=(win_diam, win_diam))
	# win_inds = win_inds.reshape(c_h, c_w, win_size)

	win_inds = win_inds.reshape(c_h, c_w, win_size)
	if mask is not None:
		mask = cv2.dilate(
			mask.astype(np.uint8),
			np.ones((win_diam, win_diam), np.uint8)
		).astype(np.bool)
		win_inds_np = win_inds.numpy()
		win_mask = np.sum(mask.ravel()[win_inds_np], axis=2)
		win_inds = win_inds[win_mask > 0, :]
	else:
		win_inds = win_inds.reshape(-1, win_size)

	winI = ravelImg[win_inds].type(torch.DoubleTensor)
	win_mu = torch.mean(winI, dim=1, keepdims=True)
	win_var = torch.einsum('...ji,...jk ->...ik', winI, winI) / win_size - torch.einsum('...ji,...jk ->...ik', win_mu, win_mu)

	tmp = win_var + (eps/win_size)*torch.eye(3).double()
	inv = torch.inverse(win_var + (eps/win_size)*torch.eye(3).double())
	X = torch.einsum('...ij,...jk->...ik', winI - win_mu, inv)
	vals = torch.eye(win_size).double() - (1.0/win_size)*(1 + torch.einsum('...ij,...kj->...ik', X, winI - win_mu))

	nz_indsCol = win_inds.repeat([1, win_size]).flatten()
	tmp = torch.flatten(win_inds).unsqueeze(1)
	nz_indsRow = tmp.repeat([1, win_size]).flatten()
	nz_indsVal = vals.flatten()
	# print(nz_indsVal, nz_indsVal.size())
	coo_inds = torch.stack((nz_indsRow, nz_indsCol), dim=0)
	L = torch.sparse.FloatTensor(coo_inds, nz_indsVal, torch.Size((h*w, h*w)))
	return L # .to_dense()

def closed_form_matting_with_prior(image, prior, prior_confidence, const_map=None):
	laplacian = compute_laplacian(image, ~const_map if const_map is not None else None)
	# confidence = torch.diag(torch.from_numpy(prior_confidence).flatten()).to_sparse()
	confidence_array = scipy.sparse.diags(prior_confidence.ravel())
	h, w = confidence_array.shape
	indices = np.array([[x, x] for x in range(h)]).T
	values = confidence_array.data
	i = torch.LongTensor(indices)
	v = torch.DoubleTensor(values).squeeze(0)
	shape = confidence_array.shape
	confidence_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

	A = laplacian + confidence_tensor
	# print("prior: ", np.unique(prior), prior.dtype)
	# print("prior_confidence: ", np.unique(prior_confidence), prior_confidence.dtype)
	B = prior.ravel() * prior_confidence.ravel()
	# print("target1: ", np.unique(B), B.shape, B.dtype)
	B = torch.from_numpy(B).double().unsqueeze(1)
	# print("target2: ", B.unique())
	return A, B, laplacian


def _rolling_block_np(A, block=(3, 3)):
	"""Applies sliding window to given matrix."""
	shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
	strides = (A.strides[0], A.strides[1]) + A.strides
	return as_strided(A, shape=shape, strides=strides)


"""
closed-form matting functin
(L+lambda*D) * alpha = lamba*D*S
where:
	img: N = wxh
	L: laplacian mat (NxN)
	D: diagnal mat of scribbles (NxN)
	S: scribbles image (Nx1)

parameters:
	image: input image (wxhx3) 
	trimap: trimap image (wxh) 
	notice: 
		image and trimap must divided by 100.

return:
	A = L+lambda*D
	B = lamba*D*S
"""
def closed_form_matting(image, trimap, trimap_confidence=100.0):
	# print(image.size(), trimap.shape)
	assert image.size()[:2] == trimap.shape, ('trimap must be 2D matrix with height and width equal '
											 'to image.')
	consts_map = (trimap < 0.1) | (trimap > 0.9)
	return closed_form_matting_with_prior(image, trimap, trimap_confidence * consts_map, consts_map)

class ClosedFormLoss(nn.Module):
	"""docstring for ClosedFormLoss"""
	def __init__(self, n_classes, img_w=513, img_h=513, reg=False, trimap_confidence=100):
		super(ClosedFormLoss, self).__init__()
		self._criterion = nn.MSELoss()
		self._reg = reg
		self._trimap_confidence = trimap_confidence
		self._sigmoid = nn.Sigmoid()
		self.n_classes = n_classes
		self.img_w, self.img_h = img_w, img_h
		self.classes = ('plane', 'bike', 'person', 'car', 'dog', 'cat', 'background')
		# self._criterion = nn.SmoothL1Loss()


	def __gentrimaps(self, trimap):
		trimap_list = np.unique(trimap)
		trimaps = np.zeros((self.n_classes, self.img_w, self.img_h))
		trimaps.fill(128)
		for i in range(self.n_classes):
			tri = trimaps[i]
			tag_ind = i+1
			for ind in trimap_list:
				if ind == 128:
					continue
				if ind == tag_ind:
					tri[trimap == ind] = 255
				else:
					tri[trimap == ind] = 0
		return trimaps


	def forward(self, cprob, img_org, trimap):
		n_batches = cprob.size()[0]
		loss = torch.zeros((n_batches, self.n_classes))

		if self._reg:
			reg = torch.zeros((n_batches, self.n_classes))
		for n in range(n_batches):
			trimap_np = trimap[n].numpy()
			trimap_cls = self.__gentrimaps(trimap_np)
			for c in range(self.n_classes):
				trimap_one_cls = trimap_cls[c] / 255.0
				A, target, L = closed_form_matting(img_org[n]/255.0, trimap_one_cls, 
					trimap_confidence=self._trimap_confidence)
				# print(A.size(), target.size(), L.size(), cprob.size())
				output = cprob[n][c].reshape((-1,1)).double().requires_grad_(True)
				# print(output.size())
				left = torch.sparse.mm(A, output)
				loss[n][c] = self._criterion(left, target) / A.size()[0]
				if self._reg:
					t = torch.sparse.mm(L, output).to_dense()
					# print(t.size())
					reg[n][c] = torch.matmul(t, output)

		if self._reg:
			return loss.sum().cuda(), (reg.sum()/(n_batches*self.n_classes)).cuda()
		else:
			return loss.sum().cuda()

	"""
	def forward(self, cprob, img_org, trimap):
		n_batches = cprob.size()[0]
		loss = torch.zeros(n_batches)

		if self._reg:
			reg = torch.zeros(n_batches)
		for n in range(n_batches):
			trimap_np = trimap[n].numpy()/255.0
			# const_map = (trimap_np < 0.1) | (trimap_np > 0.9)
			A, target, L = closed_form_matting(img_org[n]/255.0, trimap_np, 
				trimap_confidence=self._trimap_confidence)
			# output = self._sigmoid(cprob[n][0].reshape(-1,1)).double()
			output = cprob[n].reshape((-1,1)).double().requires_grad_(True)
			# print(A.size(), A.type(), output.size(), output.type())
			left = torch.sparse.mm(A, output)
			loss[n] = self._criterion(left, target) / A.size()[0]
			if self._reg:
				t = torch.sparse.mm(L, output).to_dense()
				print(t.size())
				reg[n] = torch.matmul(t, output)
				# reg[n] = torch.norm(alpha) / (alpha.size()[0] * alpha.size()[1])
		if self._reg:
			return loss.sum().cuda(), (reg.sum()/n_batches).cuda()
		else:
			return loss.sum().cuda()
	"""

def laplacian_test():
	img = np.arange(75).reshape(5,5,3) / 255.
	trimap = np.zeros((5,5))
	trimap.fill(128)
	trimap[1][1] = 255
	trimap[1][2] = 255
	trimap[1][3] = 255
	trimap[2][0] = 0
	trimap[2][2] = 0
	trimap = trimap / 255.
	img_tensor = torch.from_numpy(img)
	A, target = closed_form_matting(img_tensor, trimap)
	print("A: ", A.to_dense(), A.size())
	print("target: ", target, target.size())

	# A_np, target_np = closed_form_matting_np(img, trimap)
	# print("A_np: ", A_np.toarray(), A_np.shape)
	# print("target_np: ", target_np, target_np.shape)

if __name__ == '__main__':
	laplacian_test()



