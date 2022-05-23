# import imp
import json
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from module import FCMethod as Method
from module import MyModelList

from torch.utils.tensorboard import SummaryWriter

import os, time
import numpy as np
import random
import pdb
import argparse

from dataset import MyDataset
from utils import calc_acc, save_ckpt

# batch_size = 128
# workers = 4
# use_cuda = True
# lr = 0.001
# end_lr = 0.0001
# epochs = 4000
# end_epochs = int(0.9*epochs)
# momentum = 0.9
# weight_decay = 0.001
# optimizer_type = 'AdamW'
# print_freq = 10
# only_val = False

# data_root = r'./data'


def parse_args():
    parser = argparse.ArgumentParser(description='training setting for checkerboard')
    parser.add_argument('--exp_name', type=str, default='fcnet', required=True)
    # data setting
    parser.add_argument('--data_root', type=str, default=os.path.join('.', 'data'), required=False, help='path of dataset files')
    parser.add_argument('--num_workers', type=int, default=4, required=False)
    # model setting
    parser.add_argument('--model_size', nargs='+', default=['6-100'], required=False)
    parser.add_argument('--act_type', type=str, default='relu', choices=('relu', 'abs'), required=False)
    parser.add_argument('--layer', type=str, default='fc_layer', choices=('fc_layer', 'han_layer'), required=False)
    # optimization setting
    parser.add_argument('--loss', type=str, default='CE', required=False)
    parser.add_argument('--optimizer', type=str, default='AdamW', required=False)
    parser.add_argument('--batch_size', type=int, default=128, required=False)
    parser.add_argument('--lr', type=float, default=0.001, required=False)
    parser.add_argument('--end_lr', type=float, default=0.0001, required=False)
    parser.add_argument('--epochs', type=int, default=4000, required=False)
    parser.add_argument('--end_epochs', type=int, default=3600, required=False)
    parser.add_argument('--momentum', type=float, default=0.9, required=False)
    parser.add_argument('--weight_decay', type=float, default=1e-3, required=False)
    parser.add_argument('--print_freq', type=int, default=10, required=False)
    parser.add_argument('--only_val', action='store_true', default=False)

    args = parser.parse_args()
    args.end_epochs = int(0.9 * args.epochs)
    args.end_lr = 0.04 * args.lr
    print(json.dumps(vars(args), indent=4))
    return args


def train_iterations(model, dataset, epoch, optimizers, criterion, calc, print_freq, batch_size, all_idx):
    assert len(model.model_list) == len(optimizers), "number of models MUST be the same as optimizers"
    model.set_train()
    all_acc = 0.0
    all_loss = 0.0
    for iteration in range(print_freq):
        choice_idx = random.sample(all_idx, batch_size)
        # pdb.set_trace()
        x = dataset.x[choice_idx]
        y = dataset.y[choice_idx]

        pred = model(x)
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss = criterion(pred, y)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        acc = calc(pred, y)
        all_acc += acc * pred.shape[0]
        all_loss += loss.item()

    return float(all_acc)/float(print_freq * batch_size), all_loss / float(print_freq)


def evaluate_iterations(model, dataset, epoch, calc, print_freq, batch_size):
    model.set_eval()
    all_acc = 0.0
    idx = 0
    all_sample = len(dataset.x)
    while idx < all_sample:
        next_idx = min(all_sample, idx + batch_size)
        x, y = dataset.x[idx: next_idx], dataset.y[idx: next_idx]
        pred = model(x)
        acc = calc(pred, y)
        all_acc += acc * (next_idx - idx)
        idx = next_idx

    avg_acc = all_acc / all_sample
    return avg_acc


def main():
    # pdb.set_trace()
    args = parse_args()

    ## init tensorboard
    now_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    today = time.strftime('%Y%m%d', time.localtime(time.time()))
    tb_path = os.path.join(os.path.join('./', 'tb_{}'.format(today), '{}_{}_{}'.format(args.exp_name, '-'.join(args.model_size), now_time)))
    if not os.path.exists(tb_path):
        os.makedirs(tb_path)
    tb = SummaryWriter(tb_path)
    if torch.cuda.is_available():
        use_cuda = True
    else:
        use_cuda = False

    ## init datasets
    train_set = MyDataset(args.data_root, 'train', use_cuda=use_cuda)
    test_set = MyDataset(args.data_root, 'test', use_cuda=use_cuda)
    val_set = MyDataset(args.data_root, 'val', use_cuda=use_cuda)


    ## init model and optimizer
    # model = Method()
    model = MyModelList(args.model_size, args.layer, args.act_type)
    # print(type(model))
    # print(type(model.model_list[0]))
    if use_cuda:
        model.to_cuda()

    optimizers = list()
    lr_schedulers = list()
    for i in range(len(model.model_list)):
        # print(type(model.model_list[i]))
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.model_list[i].parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(model.model_list[i].parameters(),
                                          lr=args.lr,
                                          weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - min(1, (float(x) / float(args.end_epochs))) * ((args.lr-args.end_lr)/args.lr)))
        optimizers.append(optimizer)
        lr_schedulers.append(lr_scheduler)

    # init loss
    if args.loss == 'CE':
        criterion = nn.CrossEntropyLoss(reduction='mean')
    else:
        criterion = nn.MSELoss(reduction='mean')

    if args.only_val:
        for i, size in enumerate(args.model_size):
            state_dict = torch.load('best-{}.pth'.format(size))
            model[i].load_state_dict(state_dict)
        acc = evaluate_iterations(model, val_set, epoch, calc_acc, args.print_freq, args.batch_size)
        print('Best Model Validation accuracy is {:.6f}'.format(acc))
        acc = evaluate_iterations(model, test_set, epoch, calc_acc, args.print_freq, args.batch_size)
        print('Best Model Test accuracy is {:.6f}'.format(acc))

        return

    best_acc_val = -0.1
    best_acc_test = -0.1
    best_epoch = -1
    all_idx = list(range(len(train_set.x)))

    for epoch in range(args.epochs):
        ## each epoch has print_freq iterations, which not contains all trainset samples
        start_time = time.time()
        train_acc, avg_loss = train_iterations(model, train_set, epoch, optimizers, criterion, calc_acc, args.print_freq, args.batch_size, all_idx)

        tb.add_scalar('metric/Train_acc', train_acc, epoch)
        tb.add_scalar('metric/LR', optimizer.param_groups[0]["lr"], epoch)
        for scheduler in lr_schedulers:
            scheduler.step()

        acc_val = evaluate_iterations(model, val_set, epoch, calc_acc, args.print_freq, args.batch_size)
        tb.add_scalar('metric/Val_acc', acc_val, epoch)
        acc_test = evaluate_iterations(model, test_set, epoch, calc_acc, args.print_freq, args.batch_size)
        tb.add_scalar('metric/Test_acc', acc_test, epoch)
        if (epoch + 1) % min(200, args.epochs) == 0:
            print('Epoch {:04d} | LR: {:.8f} | Train acc: {:.6f} Loss: {:.6f} | Val acc {:.6f} | Test acc {:.6f} | Time: {:.1f}s'.format(epoch + 1, optimizer.param_groups[0]["lr"], train_acc, avg_loss, acc_val, acc_test, time.time() - start_time))

        if acc_val > best_acc_val:
            # torch.save(model.state_dict(), 'best.pth')
            save_ckpt(model, acc_val, epoch, tb_path, args.model_size, subset='val')
            best_val_acc = acc_val
            best_val_epoch = epoch
            best_val_model = model

        if acc_test > best_acc_test:
            # torch.save(model.state_dict(), 'best.pth')
            save_ckpt(model, acc_test, epoch, tb_path, args.model_size, subset='test')
            best_test_acc = acc_test
            best_test_epoch = epoch
            best_test_model = model

    tb.close()
    # best model
    acc_1 = evaluate_iterations(best_val_model, test_set, epoch, calc_acc, args.print_freq, args.batch_size)
    print('Test accuracy on best validation model is {:.6f} [{:.6f}]'.format(acc_1, best_val_acc))
    acc_2 = evaluate_iterations(best_test_model, test_set, epoch, calc_acc, args.print_freq, args.batch_size)
    print('Test accuracy on best test model is {:.6f}'.format(acc_2))

    # latest model
    acc_test = evaluate_iterations(model, test_set, epoch, calc_acc, args.print_freq, args.batch_size)
    acc_val = evaluate_iterations(model, val_set, epoch, calc_acc, args.print_freq, args.batch_size)
    print('Test accuracy on latest validation model is {:.6f} [Val: {:.6f}]'.format(acc_test, acc_val))

    with open('exp_log_{}.txt'.format(today), 'a') as fw:
        line = '\n\n|'
        for k, v in vars(args).items():
            line += '| {}: {} |'.format(k, v)
        line += '|\n'
        fw.write(line)
        fw.write('Test Acc | best val model: {:.6f} | best test model {:.6f}| latest model: {:.6f} [val: {:.6f}] |'.format(acc_1, acc_2, acc_test, acc_val))
        fw.write('\n')

if __name__ == "__main__":
    main()
