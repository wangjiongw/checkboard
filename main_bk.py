import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import os, time
import numpy as np
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
print_freq = 1000
only_val = False

data_root = r'./data'
class MyDataset(Dataset):
	def __init__(self, root, split):
		self.root = root
		self.split = split
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
		y = np.argmax(y,1)
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


def calc_acc(pred, y):
	pred = pred.argmax(1)
	acc_cnt = torch.sum(pred==y)
	batchsize = y.shape[0]
	acc = float(acc_cnt) / float(batchsize)
	return acc

def train_one_epoch(model, dataloader, epoch, optimizer, criterion, calc):
	# pdb.set_trace()
	# start_time = time.time()
	model.train()
	all_acc = 0.0
	all_data = 0.0
	all_loss = 0.0
	for iteration, (x, y) in enumerate(dataloader):
		if use_cuda:
			x = x.cuda()
			y = y.cuda()

		pred = model(x)

		optimizer.zero_grad()
		loss = criterion(pred, y)
		loss.backward()
		optimizer.step()

		acc = calc(pred, y)
		all_acc += acc * pred.shape[0]
		all_data += pred.shape[0]
		all_loss += loss.item()
		# if iteration % print_freq == 0:
		if False:
			print('Epoch {:02d} | Train | Iter: {:04d}/len(dataloader), loss: {:.4f}'.format(epoch, iteration, loss.item()))
	# print('Epoch {:02d} | Train | accuracy is {:.4f}'.format(epoch, float(all_acc)/float(all_data)))
	return float(all_acc)/float(all_data), all_loss / float(iteration)


def evaluate(model, dataloader, epoch, calc):
	model.eval()
	# start_time = time.time()
	all_acc = 0.0
	all_data = 0.0
	for iteration, (x, y) in enumerate(dataloader):
		if use_cuda:
			x = x.cuda()
			y = y.cuda()

		pred = model(x)
		acc = calc(pred, y)
		batchsize = x.shape[0]
		all_acc += acc * batchsize
		all_data += batchsize
		# if iteration % print_freq == 0:
		if False:
			print('Test | Iter: {:04d}/{:04d}, acc: {:.4f}'.format(iteration, len(dataloader), acc))

	avg_acc = all_acc / all_data
	# print('Epoch {:02d} | Test  | accuracy {:.4f}'.format(epoch, avg_acc))
	return avg_acc



def main():
	# pdb.set_trace()
	train_set = MyDataset(data_root, 'train')
	test_set = MyDataset(data_root, 'test')
	val_set = MyDataset(data_root, 'val')

	trainloader = DataLoader(train_set, batch_size = batch_size, num_workers = workers, shuffle=True, drop_last=False)
	testloader = DataLoader(test_set, batch_size = batch_size, num_workers = workers, shuffle=False, drop_last=False)
	valloader = DataLoader(val_set, batch_size = batch_size, num_workers = workers, shuffle=False, drop_last=False)

	model = MyMethod()
	if use_cuda: model = model.cuda()

	# optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=momentum, weight_decay = weight_decay)
	optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = weight_decay)
	lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - min(1, (float(x) / float(end_epochs))) * ((lr-end_lr)/lr)))
	criterion = nn.CrossEntropyLoss(reduction='mean')

	if only_val:
		state_dict = torch.load('best.pth')
		model.load_state_dict(state_dict)
		acc = evaluate(mode, valloader, epoch, calc_acc)
		print('Validation accuracy is {:.4f}'.format(acc))
		return

	best_acc = -0.1
	best_epoch = -1
	for epoch in range(epochs):
		start_time = time.time()
		train_acc, avg_loss = train_one_epoch(model, trainloader, epoch, optimizer, criterion, calc_acc)
		lr_scheduler.step()
		acc = evaluate(model, testloader, epoch, calc_acc)
		print('Epoch {:04d} | LR: {:.8f} | Train acc: {:.6f} Loss: {:.6f} | Test acc {:.6f} | Time: {:.1f}s'.format(epoch, optimizer.param_groups[0]["lr"], train_acc, avg_loss, acc, time.time() - start_time))
		if acc > best_acc:
			torch.save(model.state_dict(), 'best.pth')
			best_acc = acc
			best_epoch = epoch

		# print('Epoch: {:02d} current_acc: {:.4f}, best_acc: {:.4f} on epoch {:02d}'.format(epoch, acc, best_acc, best_epoch))


	acc = evaluate(model, valloader, epoch, calc_acc)
	print('Validation accuracy is {:.4f}'.format(acc))

if __name__ == "__main__":
	main()
