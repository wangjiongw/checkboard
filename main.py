import imp
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from module import FCMethod as Method

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import os, time
import numpy as np
import random
import pdb

batch_size = 128
workers = 4
use_cuda = True
lr = 0.001
end_lr = 0.0001
epochs = 4000
end_epochs = int(0.9*epochs)
momentum = 0.9
weight_decay = 0.001
optimizer_type = 'AdamW'
print_freq = 10
only_val = False

data_root = r'./data'
class MyDataset(Dataset):
	def __init__(self, root, split, use_cuda=True):
		self.root = root
		self.split = split
		self.use_cuda = use_cuda
		self.data_path = os.path.join(root, '{}data.txt'.format(split))
		self.label_path = os.path.join(root, '{}label.txt'.format(split))
		self.x, self.y = self._create_dataset(self.data_path, self.label_path)
		

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		'''
		self.x: N*C, self.y: N [tensor data on cuda]
		'''
		return self.x[idx], self.y[idx]

	def _create_dataset(self, data_path, label_path):
		x = np.loadtxt(data_path).astype(np.float32)
		y = np.loadtxt(label_path)
		y = np.argmax(y,1)
		x, y = torch.from_numpy(x), torch.from_numpy(y)
		if self.use_cuda: x, y = x.cuda(), y.cuda()
		return x, y




def calc_acc(pred, y):
	pred = pred.argmax(1)
	acc_cnt = torch.sum(pred==y)
	batchsize = y.shape[0]
	acc = float(acc_cnt) / float(batchsize)
	return acc


def train_iterations(model, dataset, epoch, optimizer, criterion, calc, print_freq, batch_size, all_idx):
	model.train()
	all_acc = 0.0
	all_loss = 0.0
	for iteration in range(print_freq):
		choice_idx = random.sample(all_idx, batch_size)
		# pdb.set_trace()
		x = dataset.x[choice_idx]
		y = dataset.y[choice_idx]

		pred = model(x)
		optimizer.zero_grad()
		loss = criterion(pred, y)
		loss.backward()
		optimizer.step()

		acc = calc(pred, y)
		all_acc += acc * pred.shape[0]
		all_loss += loss.item()

	return float(all_acc)/float(print_freq * batch_size), all_loss / float(print_freq)

def evaluate_iterations(model, dataset, epoch, calc, print_freq, batch_size):
	model.eval()
	all_acc = 0.0
	idx = 0
	all_sample = len(dataset.x)
	while idx < all_sample:
		next_idx = min(all_sample, idx+batch_size)
		x, y = dataset.x[idx:next_idx], dataset.y[idx:next_idx]
		pred = model(x)
		acc = calc(pred, y)
		all_acc += acc * (next_idx - idx)
		idx = next_idx

	avg_acc = all_acc / all_sample
	return avg_acc



def main():
	# pdb.set_trace()
	## init tensorboard
	now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
	tb_path = os.path.join(os.path.join('./', 'tb', now_time))
	if not os.path.exists(tb_path): os.makedirs(tb_path)
	tb = SummaryWriter(tb_path)

	## init datasets
	train_set = MyDataset(data_root, 'train', use_cuda = use_cuda)
	test_set = MyDataset(data_root, 'test', use_cuda = use_cuda)
	val_set = MyDataset(data_root, 'val', use_cuda = use_cuda)


	## init model and optimizer
	model = Method()
	if use_cuda: model = model.cuda()

	if optimizer_type == 'SGD':
		optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=momentum, weight_decay = weight_decay)
	else:
		optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
	lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - min(1, (float(x) / float(end_epochs))) * ((lr-end_lr)/lr)))
	criterion = nn.CrossEntropyLoss(reduction='mean')

	if only_val:
		state_dict = torch.load('best.pth')
		model.load_state_dict(state_dict)
		acc = evaluate_iterations(model, val_set, epoch, calc_acc, print_freq, batch_size)
		print('Validation accuracy is {:.6f}'.format(acc))
		return

	best_acc = -0.1
	best_epoch = -1
	all_idx = list(range(len(train_set.x)))
	for epoch in range(epochs):
		## each epoch has print_freq iterations, which not contains all trainset samples
		start_time = time.time()
		train_acc, avg_loss = train_iterations(model, train_set, epoch, optimizer, criterion, calc_acc, print_freq, batch_size, all_idx)

		tb.add_scalar('metric/Train_acc', train_acc, epoch)
		tb.add_scalar('metric/LR', optimizer.param_groups[0]["lr"], epoch)
		lr_scheduler.step()

		acc = evaluate_iterations(model, val_set, epoch, calc_acc, print_freq, batch_size)
		tb.add_scalar('metric/Val_acc', acc, epoch)
		print('Epoch {:04d} | LR: {:.8f} | Train acc: {:.6f} Loss: {:.6f} | Val acc {:.6f} | Time: {:.1f}s'.format(epoch, optimizer.param_groups[0]["lr"], train_acc, avg_loss, acc, time.time() - start_time))
		
		if acc > best_acc:
			# torch.save(model.state_dict(), 'best.pth')
			best_acc = acc
			best_epoch = epoch
			best_model = model


	tb.close()
	acc = evaluate_iterations(best_model, test_set, epoch, calc_acc, print_freq, batch_size)
	print('Test accuracy on best validation model is {:.6f}'.format(acc))

	acc = evaluate_iterations(model, test_set, epoch, calc_acc, print_freq, batch_size)
	print('Test accuracy on latest validation model is {:.6f}'.format(acc))

if __name__ == "__main__":
	main()
