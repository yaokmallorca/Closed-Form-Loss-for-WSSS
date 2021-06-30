import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import orth

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class ScribbleLoss(nn.Module):
	"""docstring for ScribbleLoss"""
	def __init__(self, n_classes):
		super(ScribbleLoss, self).__init__()
		self.sigmoid = nn.Sigmoid()
		self.n_classes = n_classes - 1

	def forward(self, inputs, trimaps):
		n_batches = inputs.size()[0]
		loss = Variable(torch.zeros((n_batches, 1)), requires_grad=True).cuda()
		for n in range(n_batches):
			trimap = trimaps[n]
			mask = torch.zeros(trimap.shape).cuda()
			mask[trimap == 255] = 1

			# cprob = self.sigmoid(inputs[n][self.n_classes])
			cprob = inputs[n][self.n_classes]
			alpha = cprob * mask
			loss[n] = (mask - alpha).sum() / len(torch.where(mask!=0)[0])
		return loss.sum()/n_batches

class SoftCrossEntropy2d(nn.Module):

	def __init__(self, reduction='mean', ignore_label=255):
		super(SoftCrossEntropy2d, self).__init__()
		self.reduction = reduction
		self.ignore_label = ignore_label
		self.log_softmax = nn.LogSoftmax(dim=1)

	"""
		L = \frac{1}{batch_size} \sum_{j=1}^{batch_size} \sum_{i=1}^{N}y_{ji}log(p_{ji})
		N - total number of pixels
	"""
	def forward(self, predict, target, reduction='mean'):
		"""
			Args:
				predict:(batch, c, h, w)
				target:(batch, c, h, w)
				weight (Tensor, optional): a manual rescaling weight given to each class.
										   If given, has to be a Tensor of size "nclasses"
		"""
		assert not target.requires_grad
		assert predict.dim() == 4
		assert predict.size(0) == target.size(0), "channle: {0} vs {0} ".format(predict.size(0), target.size(0))
		assert predict.size(1) == target.size(1), "channle: {1} vs {1} ".format(predict.size(1), target.size(1))
		assert predict.size(2) == target.size(2), "channle: {2} vs {2} ".format(predict.size(2), target.size(2))
		assert predict.size(3) == target.size(3), "channle: {3} vs {3} ".format(predict.size(3), target.size(3))
		log_prob = self.log_softmax(predict)
		log_prob = log_prob.view(log_prob.size(0), -1)
		batch_loss = - torch.sum(target.view(target.size(0), -1) * log_prob, dim=1) / (predict.size(2) * predict.size(3))
		if reduction == 'none':
			return batch_loss
		elif reduction == 'mean':
			return torch.mean(batch_loss)
		elif reduction == 'sum':
			return torch.sum(batch_loss)
		else:
			raise NotImplementedError('Unkown reduction mode')

class BinaryDiceLoss(nn.Module):
	"""Dice loss of binary class
	Args:
		smooth: A float number to smooth loss, and avoid NaN error, default: 1
		p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
		predict: A tensor of shape [N, *]
		target: A tensor of shape same with predict
		reduction: Reduction method to apply, return mean over batch if 'mean',
			return sum if 'sum', return a tensor of shape [N,] if 'none'
	Returns:
		Loss tensor according to arg reduction
	Raise:
		Exception if unexpected reduction
	"""
	def __init__(self, smooth=1, p=2, reduction='mean'):
		super(BinaryDiceLoss, self).__init__()
		self.smooth = smooth
		self.p = p
		self.reduction = reduction

	def forward(self, predict, target):
		assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
		predict = predict.contiguous().view(predict.shape[0], -1)
		target = target.contiguous().view(target.shape[0], -1)

		num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
		den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

		loss = 1 - num / den

		if self.reduction == 'mean':
			return loss.mean()
		elif self.reduction == 'sum':
			return loss.sum()
		elif self.reduction == 'none':
			return loss
		else:
			raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
	"""Dice loss, need one hot encode input
	Args:
		weight: An array of shape [num_classes,]
		ignore_index: class index to ignore
		predict: A tensor of shape [N, C, *]
		target: A tensor of same shape with predict
		other args pass to BinaryDiceLoss
	Return:
		same as BinaryDiceLoss
	"""
	def __init__(self, weight=None, logit="sigmoid", ignore_index=None, **kwargs):
		super(DiceLoss, self).__init__()
		self.kwargs = kwargs
		self.weight = weight
		self.ignore_index = ignore_index
		self.logit = logit


	def forward(self, predict, target):
		assert predict.shape == target.shape, 'predict & target shape do not match'
		dice = BinaryDiceLoss(**self.kwargs)
		total_loss = 0
		if self.logit == 'sigmoid':
			predict = F.sigmoid(predict)
		else:
			predict = F.softmax(predict, dim=1)

		for i in range(target.shape[1]):
			if i != self.ignore_index:
				dice_loss = dice(predict[:, i], target[:, i])
				if self.weight is not None:
					assert self.weight.shape[0] == target.shape[1], \
						'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
					dice_loss *= self.weights[i]
				total_loss += dice_loss

		return total_loss/target.shape[1]


if __name__ == '__main__':
	ce_torch = nn.NLLLoss()
	ce_own = SoftCrossEntropy2d()
	loss = nn.MultiLabelSoftMarginLoss()

	data_np = np.random.uniform(0, 1, size=(3,2,5,5))
	target_np = np.random.randint(0, 2, size=(3,1,5,5))
	target_oh = np.zeros(data_np.shape)

	target_oh[0][0][target_np[0][0] == 0] = 1
	target_oh[0][1][target_np[0][0] == 1] = 1
	target_oh[1][0][target_np[1][0] == 0] = 1
	target_oh[1][1][target_np[1][0] == 1] = 1
	target_oh[2][0][target_np[2][0] == 0] = 1
	target_oh[2][1][target_np[2][0] == 1] = 1

	target_torch = torch.from_numpy(target_np).type(torch.LongTensor)
	target_torch = target_torch.squeeze(1)
	target_own = torch.from_numpy(target_oh).type(torch.FloatTensor)
	data = torch.from_numpy(data_np).type(torch.FloatTensor)
	data.requires_grad = True

	log_data = nn.LogSoftmax(dim=1)(data)
	output_torch = ce_torch(log_data, target_torch)
	output_torch.backward()
	print('torch loss: ', output_torch)

	output_multi = loss(data, target_own)
	print('multi loss: ', output_multi)

	print(data.size(), target_own.size())
	output_own = ce_own(data, target_own)
	output_own.backward()
	print("own loss: ", output_own)

