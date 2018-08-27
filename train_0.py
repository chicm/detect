from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss import FocalLoss
from retinanet import RetinaNet
from datagen import ListDataset

from torch.autograd import Variable
import time

from imgdataset import get_train_loader, get_small_train_loader
import settings

batch_size = 16

def run_train(args):
    assert torch.cuda.is_available(), 'Error: CUDA not found!'
    best_loss = float('inf')  # best test loss
    start_epoch = 0  # start from epoch 0 or last epoch

    # Data
    print('==> Preparing data..')

    trainloader = get_train_loader(img_dir=settings.IMG_DIR, batch_size=batch_size)
    #trainloader = get_small_train_loader()
    print(trainloader.num)
    #testloader = get_train_loader(img_dir=settings.IMG_DIR)

    # Model
    net = RetinaNet()
    #net.load_state_dict(torch.load('./model/net.pth'))
    net.load_state_dict(torch.load('./ckps/best_2.pth'))
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.cuda()

    criterion = FocalLoss()
    #optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    iter_save = 100
    bgtime = time.time()
    # Training
    for epoch in range(start_epoch, start_epoch+100):
        print('\nEpoch: %d' % epoch)
        net.train()
        #net.module.freeze_bn()
        train_loss = 0
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
            inputs = Variable(inputs.cuda())
            loc_targets = Variable(loc_targets.cuda())
            cls_targets = Variable(cls_targets.cuda())

            optimizer.zero_grad()
            loc_preds, cls_preds = net(inputs)
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data[0]
            sample_num = (batch_idx+1)*batch_size
            print('Epoch: {}, num: {}/{} train_loss: {:.3f} | avg_loss: {:.3f} min: {:.1f}'.format(
                epoch, sample_num, trainloader.num, loss.data[0], train_loss/(batch_idx+1), (time.time() - bgtime)/60), end='\r')

            if batch_idx % iter_save == 0:
                torch.save(net.module.state_dict(), './ckps/best_{}.pth'.format(batch_idx//iter_save % 5))

# Test
def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
        inputs = Variable(inputs.cuda(), volatile=True)
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        test_loss += loss.data[0]
        print('test_loss: %.3f | avg_loss: %.3f' % (loss.data[0], test_loss/(batch_idx+1)))

    # Save checkpoint
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_loss = test_loss

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    
    run_train(args)
    #test(epoch)
