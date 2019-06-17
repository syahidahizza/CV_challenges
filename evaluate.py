import argparse
import os
import random
import shutil
import time
import warnings
import pickle

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from skimage import transform
from inceptionv4 import *
from autoaugment import *
from dataset import *
from util import AverageMeter, ProgressMeter, accuracy

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
number_preprocess = [5,6,15,20]
    
def parse_args() :
    parser = argparse.ArgumentParser(description='PyTorch Final Testing')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-p', '--print-freq', default=200, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='checkpoint_cars_1-e4_decay_real.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: checkpoint_cars_1-e4_decay_real.pth.tar)')
    parser.add_argument('-data', metavar='DIR', default='cars/',
                        help='path to dataset')
    parser.add_argument('-pred_both', default=False, type=bool,
                        help='path to dataset')

    args = parser.parse_args()
    return args

def denom(images):
    images[:,0,:,:] = images[:,0,:,:]*0.229 + 0.485
    images[:,1,:,:] = images[:,1,:,:]*0.224 + 0.456
    images[:,2,:,:] = images[:,2,:,:]*0.225 + 0.406
    return images

def add(image, heat_map, alpha=0.6, display=False, save=None, cmap='viridis', axis='on', verbose=False):
    height = image.shape[0]
    width = image.shape[1]

    # resize heat map
    heat_map_resized = transform.resize(heat_map, (height, width))

    # normalize heat map
    max_value = np.max(heat_map_resized)
    min_value = np.min(heat_map_resized)
    normalized_heat_map = (heat_map_resized - min_value) / (max_value - min_value)

    # display
    plt.imshow(image)
    plt.imshow(255 * normalized_heat_map, alpha=alpha, cmap=cmap)
    plt.axis(axis)

    if display:
        plt.show()

    if save is not None:
        if verbose:
            print('save image: ' + save)
        plt.savefig(save, bbox_inches='tight', pad_inches=0)
        
def main() :
    args = parse_args()
    network = inceptionv4(num_classes=1000, pretrained='imagenet')

    validate_datasets = []
    validate_loader = []
    for number in number_preprocess:
        validate_datasets += [CarsDataset(os.path.join(args.data,'devkit/cars_test_annos_withlabels.mat'),
                                          os.path.join(args.data,'cars_test'),
                                          os.path.join(args.data,'devkit/cars_meta.mat'),
                                          transform=transforms.Compose([
                                              transforms.Resize(size=(512, 512)),
                                              ImageNetPolicyTest(number=number),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                        ])
                                      )]
        validate_loader += [torch.utils.data.DataLoader(
        validate_datasets[-1],
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)]

    network.cuda()
    net_1 = torch.nn.parallel.DataParallel(network,device_ids=[0]).cuda(0)

    checkpoint = torch.load(args.resume)
    net_1.load_state_dict(checkpoint['state_dict'])

    all_pred_data = np.zeros((len(validate_loader),8041,1000))

    if args.pred_both:
        all_pred_data_1 = np.zeros((len(validate_loader),8041,1000))
    
    all_y_data = np.zeros((8041))
    net_1.eval()
    
    for k in range(len(validate_loader)):
        print('Validate augment : '+str(number_preprocess[k]))
        for i, (X,y) in enumerate(validate_loader[k]):
            if args.pred_both:
                output,output_1 = net_1(X,both_pred=args.pred_both)
                all_pred_data[k,i*args.batch_size:(i+1)*args.batch_size] = output.detach().cpu().numpy()
                all_pred_data_1[k,i*args.batch_size:(i+1)*args.batch_size] = output_1.detach().cpu().numpy()
            else:
                output = net_1(X,both_pred=args.pred_both)
                all_pred_data[k,i*args.batch_size:(i+1)*args.batch_size] = output.detach().cpu().numpy()

            if k==0:
                all_y_data[i*args.batch_size:(i+1)*args.batch_size] = y.detach().cpu().numpy()
                
            if i % args.print_freq == 0:
                print('Done '+str(i)+'/'+str(8041//args.batch_size + 1))
                
    if args.pred_both:
        print('final acc augmentation only is : '+str(np.mean(
            np.argmax(np.sum(all_pred_data,axis=0)+0.02*np.sum(all_pred_data_1,axis=0),axis=1) == all_y_data) * 100))
    else:
        print('final acc augmentation only is : '+str(np.mean(
            np.argmax(np.sum(all_pred_data,axis=0),axis=1) == all_y_data) * 100))

if __name__ == '__main__':
    main()
