from pydoc import ModuleScanner
import torch
import torch.nn as nn
import torch.nn.functional as F

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
end_lr = 0.00001
momentum = 0.99
weight_decay = 0.01
epochs = 2000
end_epochs = int(0.9*epochs)
print_freq = 10
only_val = False
save_ckpt = False

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
		self.x: N*C, self.y: N
		'''
		return self.x[idx], self.y[idx]

	def _create_dataset(self, data_path, label_path):
		x = np.loadtxt(data_path).astype(np.float32)
		y = np.loadtxt(label_path)
		y = np.argmax(y, 1)
		x, y = torch.from_numpy(x), torch.from_numpy(y)
		if self.use_cuda: x, y = x.cuda(), y.cuda()
		return x, y

class MyMethod(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc = nn.Sequential(
			nn.Linear(2,100),
			nn.ReLU(),
			nn.Linear(100,100),
			nn.ReLU(),
			nn.Linear(100,100),
			nn.ReLU(),
			nn.Linear(100,100),
			nn.ReLU(),
			nn.Linear(100,100),
			nn.ReLU(),
			nn.Linear(100,2)
			)
		self.apply(self._init_weight)

	def _init_weight(self, m):
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			nn.init.constant_(m.bias, 0)


	def forward(self, x):
		# pdb.set_trace()
		x = self.fc(x)
		return x


class NewFCN(nn.Module):
	def __init__(self, depth, width):
		super().__init__()
		self.fc1 = nn.Linear(2, width)
		self.fc_layers = list()
		for i in range(depth - 2):
			self.fc_layers.append(nn.Linear(width, width))
		self.fc_layers = nn.ModuleList(self.fc_layers)
		self.fc_final = nn.Linear(width, 2)
		self.act = torch.abs
		
		# self.fc2 = nn.Linear(width, width)
		# self.fc3 = nn.Linear(width, width)
		# self.fc4 = nn.Linear(width, width)
		# self.fc5 = nn.Linear(width, width)
		self.apply(self._init_weight)

	def _init_weight(self, m):
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			nn.init.constant_(m.bias, 0)


	def forward(self, x):
		# pdb.set_trace()
		# x = self.fc(x)
		x = self.act(self.fc1(x))

		for i in range(len(self.fc_layers) - 1):
			x = self.act(self.fc_layers[i](x))
		
		x = self.fc_layers[-1](x)
		x = self.fc_final(x)
		return x


def calc_acc(pred, y):
	pred = pred.argmax(1)
	acc_cnt = torch.sum(pred==y)
	batchsize = y.shape[0]
	acc = float(acc_cnt) / float(batchsize)
	return acc


def train_iterations(models, dataset, epoch, optimizers, criterion, calc, print_freq, batch_size, all_idx):
	assert len(models) == len(optimizers)
	for model in models:
		model.train()
	all_acc = 0.0
	all_loss = 0.0
	for iteration in range(print_freq):
		choice_idx = random.sample(all_idx, batch_size)
		# pdb.set_trace()
		x = dataset.x[choice_idx]
		y = dataset.y[choice_idx]

		
		for i in range(len(models)):
			optimizers[i].zero_grad()
			if i == 0:
				pred = models[i](x)
			else:
				pred += models[i](x)
		# pred = model(x)
		pred = pred / len(models)
		loss = criterion(pred, y)
		loss.backward()
		for optimizer in optimizers:
			# optimizer.zero_grad()
			optimizer.step()

		acc = calc(pred, y)
		all_acc += acc * pred.shape[0]
		all_loss += loss.item()

	return float(all_acc)/float(print_freq * batch_size), all_loss / float(print_freq)



def evaluate_iterations(models, dataset, epoch, calc, print_freq, batch_size):
	for i in range(len(models)):
		models[i].eval()
	all_acc = 0.0
	idx = 0
	all_sample = len(dataset.x)
	while idx < all_sample:
		next_idx = min(all_sample, idx+batch_size)
		x, y = dataset.x[idx:next_idx], dataset.y[idx:next_idx]
		for i in range(len(models)):
			if i == 0:
				pred = models[i](x)
			else:
				pred += models[i](x)
		# pred = model(x)
		pred = pred / 3
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
	# model = MyMethod()
	model1 = NewFCN(6, 100)
	model2 = NewFCN(10, 100)
	model3 = NewFCN(6, 300)
	models = [model1, model2, model3]
	if use_cuda: 
		for i in range(len(models)):
			models[i] = models[i].cuda()

	# optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=momentum, weight_decay = weight_decay)
	optimizers = list()
	lr_schedulers = list()
	for i in range(len(models)):
		optimizer = torch.optim.AdamW(models[i].parameters(), lr = lr, weight_decay = weight_decay)
		optimizers.append(optimizer)
		lr_schedulers.append(torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - min(1, (float(x) / float(end_epochs))) * ((lr-end_lr)/lr))))

	# optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
	# lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - min(1, (float(x) / float(end_epochs))) * ((lr-end_lr)/lr)))
	criterion = nn.CrossEntropyLoss(reduction='mean')

	if only_val:
		state_dict = torch.load('best.pth')
		for i in range(len(models)):
			models[i].load_state_dict(state_dict)
		acc = evaluate_iterations(models, test_set, epoch, calc_acc, print_freq, batch_size)
		print('Validation accuracy is {:.6f}'.format(acc))
		return

	best_acc = -0.1
	best_epoch = -1
	all_idx = list(range(len(train_set.x)))
	for epoch in range(epochs):
		## each epoch has print_freq iterations, which not contains all trainset samples
		start_time = time.time()
		train_acc, avg_loss = train_iterations(models, train_set, epoch, optimizers, criterion, calc_acc, print_freq, batch_size, all_idx)

		tb.add_scalar('metric/Train_acc', train_acc, epoch)
		tb.add_scalar('metric/LR', optimizer.param_groups[0]["lr"], epoch)
		for lr_scheduler in lr_schedulers:
			lr_scheduler.step()

		acc = evaluate_iterations(models, val_set, epoch, calc_acc, print_freq, batch_size)
		tb.add_scalar('metric/Val_acc', acc, epoch)
		print('Epoch {:04d} | LR: {:.8f} | Train acc: {:.6f} Loss: {:.6f} | Val acc {:.6f} | Time: {:.1f}s'.format(epoch, optimizer.param_groups[0]["lr"], train_acc, avg_loss, acc, time.time() - start_time))
		
		if acc > best_acc:
                    # torch.save(model.state_dict(), 'best.pth')
                    best_acc = acc
                    best_epoch = epoch
                    best_models = models
                    if save_ckpt:
                        for i in range(len(models)):
                            torch.save(models[i].state_dict(), os.path.join(tb_path, 'val_best_{}.pth'.format(i+1)))


	acc = evaluate_iterations(best_models, test_set, best_epoch, calc_acc, print_freq, batch_size)
	tb.add_scalar('metric/Test_acc', acc, best_epoch)
	print('Best val accuracy is {:.6f} on {:04d} epoch'.format(best_acc, best_epoch))
	print('Test accuracy is {:.6f}'.format(acc))
	


if __name__ == "__main__":
	main()
