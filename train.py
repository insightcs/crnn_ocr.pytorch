#coding:utf-8
from __future__ import print_function
from __future__ import division

import os
import argparse
import random
import math
import time
import torch
import shutil
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import utils.utils as utils
from torch.autograd import Variable

from warpctc_pytorch import CTCLoss
from datasets.dataset import Dataset
from datasets.dataset import DatasetCollater
from datasets.dataset import RandomSequentialSampler
from models.crnn import CRNN
from models.crnn import DenseNet
from tqdm import tqdm
from utils.torchsummaryX import summary


def parse_args():
    parser = argparse.ArgumentParser(description='Text Recognition')
    parser.add_argument('--arch', type=str, default='crnn', 
                        choices=['crnn', 'densenet'], help='')
    
    parser.add_argument('--rnn', action='store_true', 
                        help='enables cuda')
    parser.add_argument('--num_hidden', type=int, default=256, 
                        help='size of the lstm hidden state')

    parser.add_argument('--data_root', required=True, 
                        help='path to dataset')
    parser.add_argument('--alphabet', type=str, default='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<', 
                        help='')

    parser.add_argument('--image_h', type=int, default=32, 
                        help='the height of the input image to network')
    parser.add_argument('--image_w', type=int, default=500, 
                        help='the width of the input image to network')
    parser.add_argument('--keep_ratio', action='store_true', 
                        help='whether to keep ratio for image resize')

    parser.add_argument('--max_epoch', type=int, default=25, 
                        help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='input batch size')
    parser.add_argument('--lr', type=float, default=0.01, 
                        help='learning rate for Critic, not used by adadealta')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='learning rate decay')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--drop_rate', type=float, default=0.0,
                        help='dropout probability (default: 0.2)')
    parser.add_argument('--optimizer', type=str, default='rmsprop', choices=["sgd", "adam", "rmsprop", "adadelta"], 
                        help='')

    parser.add_argument('--checkpoint', type=str, default='./checkpoints', 
                        help='Where to store samples and models')
    parser.add_argument('--pretrained', type=str, default=None, 
                        help="path to pretrained model (to continue training)")
    parser.add_argument('--resume', type=str, default=None, metavar='FILENAME',
                        help='name of the latest checkpoint (default: None)')
    
    parser.add_argument('--display_interval', type=int, default=100, 
                        help='Interval to be displayed')
    parser.add_argument('--n_test_disp', type=int, default=10, 
                        help='Number of samples to display when test')
    parser.add_argument('--val_interval', type=int, default=100, 
                        help='Interval to be displayed')
    parser.add_argument('--save_interval', type=int, default=100, 
                        help='Interval to be displayed')

    parser.add_argument('--cuda', action='store_true', 
                        help='enables cuda')
    parser.add_argument('--workers', type=int, default=0, 
                        help='number of data loading workers', )
    parser.add_argument('--manual_seed', type=int, default=1234, 
                        help='reproduce experiemnt')
    args = parser.parse_args()
    return args

def main():
    # Set parameters of the trainer
    global args, device
    args = parse_args()

    print('='*60)
    print(args)
    print('='*60)
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        #cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # load alphabet from file
    if os.path.isfile(args.alphabet):
        alphabet = ''
        with open(args.alphabet, mode='rb') as f:
            for line in f.readlines():
                alphabet += line.decode('utf-8')[0]
        args.alphabet = alphabet

    converter = utils.CTCLabelConverter(args.alphabet, ignore_case=False)

    # data loader
    image_size = (args.image_h, args.image_w)
    collater = DatasetCollater(image_size, keep_ratio=args.keep_ratio)
    train_dataset = Dataset(mode='train', data_root=args.data_root, transform=None)
    #sampler = RandomSequentialSampler(train_dataset, args.batch_size)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   collate_fn=collater, shuffle=True,
                                   num_workers=args.workers)

    val_dataset = Dataset(mode='val', data_root=args.data_root, transform=None)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size,
                                     collate_fn=collater, shuffle=True,
                                     num_workers=args.workers)


    # network
    num_classes = len(args.alphabet) + 1
    num_channels = 1
    if args.arch == 'crnn':
        model = CRNN(args.image_h, num_channels, num_classes, args.num_hidden)
    elif args.arch == 'densenet':
        model = DenseNet(num_channels=num_channels, num_classes=num_classes, 
                         growth_rate=12, block_config=(3,6,9),#(3,6,12,16),
                        compression=0.5, num_init_features=64, bn_size=4, 
                        rnn=args.rnn, num_hidden=args.num_hidden,
                        drop_rate=0, small_inputs=True, efficient=False)
    else:
        raise ValueError('unknown architecture {}'.format(args.arch))
    model = model.to(device)
    summary(model, torch.zeros((2, 1, 32, 650)).to(device))
    #print('='*60)
    #print(model)
    #print('='*60)

    # loss
    criterion = CTCLoss()
    criterion = criterion.to(device)

    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), weight_decay=args.weight_decay)
    else:
        raise ValueError('unknown optimizer {}'.format(args.optimizer))
    print('='*60)
    print(optimizer)
    print('='*60)

    # Define learning rate decay schedule
    global scheduler
    #exp_decay = math.exp(-0.1)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
    #step_size = 10000
    #gamma_decay = 0.8
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma_decay)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma_decay)

    # initialize model
    if args.pretrained and os.path.isfile(args.pretrained):
            print(">> Using pre-trained model '{}'".format(os.path.basename(args.pretrained)))
            state_dict = torch.load(args.pretrained)
            model.load_state_dict(state_dict)
            print("loading pretrained model done.")

    global is_best, best_accuracy
    is_best = False
    best_accuracy = 0.0
    start_epoch = 0
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            # load checkpoint weights and update model and optimizer
            print(">> Loading checkpoint:\n>> '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_accuracy = checkpoint['best_accuracy']
            print(">>>> loaded checkpoint:\n>>>> '{}' (epoch {})".format(args.resume, start_epoch))
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            # important not to forget scheduler updating
            #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate, last_epoch=start_epoch - 1)
        else:
            print(">> No checkpoint found at '{}'".format(args.resume))

    # Create export dir if it doesnt exist
    checkpoint = "{}".format(args.arch)
    checkpoint += "_{}".format(args.optimizer)
    checkpoint += "_lr_{}".format(args.lr)
    checkpoint += "_decay_rate_{}".format(args.decay_rate)
    checkpoint += "_bsize_{}".format(args.batch_size)
    checkpoint += "_height_{}".format(args.image_h)
    checkpoint += "_keep_ratio" if args.keep_ratio else "_width_{}".format(image_size[1])

    args.checkpoint = os.path.join(args.checkpoint, checkpoint)
    if not os.path.exists(args.checkpoint):
        os.makedirs(args.checkpoint)

    print('start training...')
    for epoch in range(start_epoch, args.max_epoch):
        # Aujust learning rate for each epoch
        scheduler.step()

        # Train for one epoch on train set
        _ = train(train_loader, val_loader, model, criterion, optimizer, epoch, converter)

def train(train_loader, val_loader, model, criterion, optimizer, epoch, converter):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    # Switch to train mode
    for p in model.parameters():
        p.requires_grad = True
    model.train()

    end = time.time()
    for i, sample in enumerate(tqdm(train_loader, desc='Train Epoch {}'.format(epoch+1))):
        
        # Aujust learning rate
        #scheduler.step()

        # Measure data loading time
        data_time.update(time.time() - end)

        # Zero out gradients so we can accumulate new ones over batches
        optimizer.zero_grad()

        # step 2. Get our inputs targets ready for the network.
        images, targets = sample

        batch_size = images.size(0)
        encoded_targets, target_lengths = converter.encode(targets)

        # step 3. Run out forward pass.
        images = images.to(device)
        log_probs = model(images)
        input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.int)

        # step 4. Compute the loss, gradients, and update the parameters
        loss = criterion(log_probs, encoded_targets, input_lengths, target_lengths) / batch_size
        losses.update(loss.item())
        model.zero_grad()
        loss.backward()

        # Do one step for multiple batches accumulated gradients are used
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.display_interval == 0 or i == 0 or (i + 1) == len(train_loader):
            print('\nTrain: [{}/{}]\t'
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t'
                  'Load Data {data_time.val:.3f}s ({data_time.avg:.3f}s\t'
                  'Loss {loss.val:.6f} ({loss.avg:.6f})'.format(
                      i + 1, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))

        # Evaluate on validation set
        val_acc = 0.0
        val_loss = 100000.0
        if (i + 1) % args.val_interval == 0 or (i + 1) == len(train_loader):
            with torch.no_grad():
                val_acc, val_loss = validate(val_loader, model, criterion, epoch, converter)
            for p in model.parameters():
                p.requires_grad = True
                model.train()

        # Remember best accuracy and save checkpoint
        global is_best, best_accuracy
        is_best = val_acc > 0.0 and val_acc >= best_accuracy
        best_accuracy = max(val_acc, best_accuracy)

        if (i + 1) % args.save_interval == 0 or (i + 1) == len(train_loader):
            save_checkpoint({
                'arch': args.arch,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_accuracy': best_accuracy,
                'loss': val_loss,
                'optimizer': optimizer.state_dict(),
            }, i+1, is_best, args.checkpoint)

    return losses.avg

def validate(val_loader, model, criterion, epoch, converter):
    accuracy = utils.AverageMeter()
    losses = utils.AverageMeter()

    # Switch to evaluate mode
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    num_correct = 0
    num_verified = 0

    for i, sample in enumerate(val_loader):
        images, targets = sample

        batch_size = images.size(0)
        encoded_targets, target_lengths = converter.encode(targets)

        images = images.to(device)
        log_probs = model(images)
        input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.int)

        # cal val loss
        loss = criterion(log_probs, encoded_targets, input_lengths, target_lengths) / batch_size
        losses.update(loss.item())

        # cal val acc
        num_verified += len(targets)
        _, preds = log_probs.max(2, keepdim=True)
        preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds, _ = converter.decode(preds.data, input_lengths.data, raw=False)
        for pred, target in zip(sim_preds, targets):
            if pred == target:
                num_correct += 1
        accuracy.update(num_correct / num_verified)

    # print text recognition
    raw_preds, _ = converter.decode(preds.data, input_lengths.data, raw=True)
    raw_preds = raw_preds[:args.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, targets):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
    print('Val accuracy: {accuracy.avg:.2%}\tloss: {loss.avg:.6f}'.format(accuracy=accuracy, loss=losses))
    return accuracy.avg, losses.avg

def save_checkpoint(state, step, is_best, checkpoint):
    filename = os.path.join(checkpoint, '{}_epoch_{}_step_{}_acc_{:.2%}_loss_{:.6f}.pth.tar'
        .format(state['arch'], state['epoch'], step, state['best_accuracy'], state['loss']))

    try:
        os.remove(filename)
    except OSError:
        pass

    torch.save(state, filename)
    if is_best:
        print('>>>> save best model at epoch: {}'.format(state['epoch']))
        filename_best = os.path.join(checkpoint, '{}_epoch_{}_acc_{:.2%}_best.pth.tar'
            .format(state['arch'], state['epoch'], state['best_accuracy']))
        try:
            os.remove(filename_best)
        except OSError:
            pass
        shutil.copyfile(filename, filename_best)

if __name__ == '__main__':
    main()
