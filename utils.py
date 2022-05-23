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


def save_ckpt(model, acc, epoch, path, model_sizes, subset='val'):
    for i in range(len(model_sizes)):
        torch.save(model.model_list[i].state_dict(), os.path.join(path, 'best_{}_{}_{}.pth'.format(subset, i + 1, model_sizes[i])))

