import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root, split, use_cuda=True, loss='CE'):
        self.root = root
        self.split = split
        self.use_cuda = use_cuda
        self.data_path = os.path.join(root, '{}data.txt'.format(split))
        self.label_path = os.path.join(root, '{}label.txt'.format(split))
        self.x, self.y = self._create_dataset(self.data_path, self.label_path, loss)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        '''
        self.x: N*C, self.y: N [tensor data on cuda]
        '''
        return self.x[idx], self.y[idx]

    def _create_dataset(self, data_path, label_path, loss_type):
        x = np.loadtxt(data_path).astype(np.float32)
        y = np.loadtxt(label_path)
        if loss_type == 'CE':
            y = np.argmax(y, 1)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if self.use_cuda: x, y = x.cuda(), y.cuda()
        return x, y
