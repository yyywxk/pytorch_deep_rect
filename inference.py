#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/1 18:50
# @Author  : yyywxk
# @File    : inference.py

import argparse
import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import time
import shutil

import torch
import torchvision

from models.deeprect.model import RectanglingNetwork

import cv2 as cv

os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # set the GPUs

import warnings

warnings.filterwarnings('ignore')


class FixResize(object):
    def __init__(self, w, h, interpolation=cv2.INTER_LINEAR):
        self.w = w
        self.h = h
        self.interpolation = interpolation

    def __call__(self, img):
        img = cv2.resize(img, (self.w, self.h), interpolation=self.interpolation)

        return img


class ToTensor(object):
    """Convert label ndarrays in sample to Tensors."""

    def __call__(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img.astype(np.float32)).float()

        return img


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    """

    def __init__(self, mean=1.0, std=127.5):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img).astype(np.float32)
        img /= self.std
        img -= self.mean

        return img


def parse_args():
    '''
    To define parameters and settings
    '''
    parser = argparse.ArgumentParser(description='PyTorch Codes for Deep Rectangling inference.')
    # dataset
    parser.add_argument('--workers', type=int, default=0, metavar='N', help='dataloader threads')
    # model path
    parser.add_argument('--model_path', type=str,
                        # default='./run/DIR-D/experiment_1/model_best.pth',
                        default='./run/AIRD/experiment_0/model_best.pth',
                        help='load your model')
    # define the image resolution
    parser.add_argument('--height', type=int, default=384, help='height of input images (default: 384)')
    parser.add_argument('--width', type=int, default=512, help='width of input images (default: 512)')
    # test settings
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA testing')
    parser.add_argument('--resize', action='store_true', default=False, help='test using resized gt')

    parser.add_argument("--mask", action='store_true', default=False, help='generate mask.')

    # save path
    parser.add_argument('--save_path', type=str,
                        default='./inference_results/',
                        help='save your prediction data')
    parser.add_argument('--save_mask_path', type=str,
                        default='./masks/',
                        help='save your prediction data')
    parser.add_argument('--input_path', type=str,
                        default='./inference/',
                        help='save your prediction data')

    # --------------------------------- Parameters of deeprect ----------------------------
    # define the mesh resolution
    parser.add_argument('--GRID_W', default=16, type=int, help='width of the mesh(default: 8)')
    parser.add_argument('--GRID_H', default=12, type=int, help='height of the mesh(default: 6)')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            torch.cuda.empty_cache()
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if os.path.exists(args.save_path) and os.path.isdir(args.save_path):
        print('Go predicting!\n')
        shutil.rmtree(args.save_path)
        if os.path.exists(args.save_mask_path) and args.mask:
            shutil.rmtree(args.save_mask_path)

    try:
        os.mkdir(args.save_path)
        if args.mask:
            os.mkdir(args.save_mask_path)
    except OSError:
        print("Creation of the testing directory %s failed" % args.save_path)
    else:
        print("Successfully created the testing directory %s " % args.save_path)

    if (not os.path.isfile(args.model_path)) or (not os.path.exists(args.model_path)):
        raise RuntimeError("no checkpoint found at '{}'".format(args.model_path))

    if not os.path.exists(args.input_path):
        raise RuntimeError("no input found at '{}'".format(args.input_path))

    args.image_path = args.input_path + 'input/'
    args.mask_path = args.input_path + 'mask/'

    if not os.path.exists(args.image_path):
        raise RuntimeError("no image found at '{}'".format(args.image_path))
    if not os.path.exists(args.mask_path):
        raise RuntimeError("no mask found at '{}'".format(args.mask_path))

    return args


class Tester(object):
    def __init__(self, args):
        self.args = args
        # Load model
        model = RectanglingNetwork(args.n_colors, args.GRID_H, args.GRID_W, args.width, args.height, args.cuda)

        self.model = model

        pin_memory = False
        if args.cuda:
            pin_memory = True  # GPU will be faster
            self.model = self.model.cuda()
        checkpoint = torch.load(args.model_path)
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        print('Best PSNR=', checkpoint['best_pred'])
        print('Ep=', checkpoint['epoch'])

    def test(self):
        self.model.eval()
        for i, name_img in enumerate(os.listdir(self.args.image_path)):
            image = cv2.imread(os.path.join(self.args.image_path, name_img))
            mask = cv2.imread(os.path.join(self.args.mask_path, name_img))

            ti = torchvision.transforms.Compose([
                FixResize(self.args.width, self.args.height, interpolation=cv2.INTER_LINEAR),
                Normalize(mean=1.0, std=127.5),
                ToTensor()
            ])

            tm = torchvision.transforms.Compose([
                FixResize(self.args.width, self.args.height, interpolation=cv2.INTER_NEAREST),
                Normalize(mean=1.0, std=127.5),
                ToTensor()
            ])

            test_input, test_mask = ti(image), tm(mask)
            if self.args.cuda:
                test_input, test_mask = test_input.unsqueeze(0).cuda(), test_mask.unsqueeze(0).cuda()
            with torch.no_grad():
                test_mesh_primary, test_warp_image_primary, test_warp_mask_primary, \
                test_mesh_final, test_warp_image_final, test_warp_mask_final = self.model(test_input, test_mask)
                # test_warp_image_final, test_warp_mask_final = self.model(test_input, test_mask, train_flag=False)
                # test_warp_image_final = self.model(test_input, test_mask, train_flag=False)

            test_warp_image = (test_warp_image_final[0] + 1.0) * 127.5
            test_warp_mask = (test_warp_mask_final[0] + 1.0) * 127.5

            if self.args.cuda:
                result = test_warp_image.permute(1, 2, 0).cpu().numpy().astype('uint8')  # cv2.imread
                # result = test_warp_mask.permute(1, 2, 0).cpu().numpy().astype('uint8')
                if self.args.mask:
                    mask_result = test_warp_mask.permute(1, 2, 0).cpu().numpy().astype('uint8')
            else:
                result = test_warp_image.permute(1, 2, 0).numpy().astype('uint8')  # cv2.imread
                # result = test_warp_gt.permute(1, 2, 0).numpy().astype('uint8')
                if self.args.mask:
                    mask_result = test_warp_mask.permute(1, 2, 0).numpy().astype('uint8')

            result = cv2.resize(result, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

            path = self.args.save_path + name_img
            cv2.imwrite(path, result)


class MyException(Exception):
    def __init__(self, msg):
        '''
        :param msg: Error
        '''
        self.msg = msg


def main(args):
    print('Start inferencing!')
    print(args)
    tester = Tester(args)
    tester.test()


if __name__ == "__main__":
    since = time.time()
    args = parse_args()
    main(args)
    time_elapsed = time.time() - since
    print('Finish testing!Totally cost: {:.0f}m {:.5f}s'.format(time_elapsed // 60, time_elapsed % 60))
