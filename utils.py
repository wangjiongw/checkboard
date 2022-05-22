import os
import sys
import random

import numpy as np
import torch

def calc_acc(pred, y):
	pred = pred.argmax(1)
	acc_cnt = torch.sum(pred == y)
	batchsize = y.shape[0]
	acc = float(acc_cnt) / float(batchsize)
	return acc


def save_ckpt(model, acc, epoch, path, keyword='', subset='val'):
	torch.save(model.state_dict(), os.path.join(path, 'best_{}_{}.pth'.format(subset, keyword)))

