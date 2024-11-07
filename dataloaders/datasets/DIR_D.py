#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/23 16:48
# @Author  : yyywxk
# @File    : DIR-D.py

import os
# from PIL import Image
import cv2
import random
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpus


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


class Images_Dataset_train(Dataset):
    """Class for getting individual transformations and data
    Args:
        input_dir = path of input images
        mask_dir = path of mask images
        gt_dir = path of groundtruth images
        transformI = Input Image transformation (default: None)
        transformM = Input Mask transformation (default: None)
        transformG = Input Groundtruth transformation (default: None)
    Output:
        ix = Transformed input images
        mx = Transformed mask images
        gx = Transformed groundtruth images
        """

    def __init__(self, input_dir, mask_dir, gt_dir, input_size_H=384, input_size_W=512,
                 transformI=None, transformM=None, transformG=None):
        self.input_images = []
        self.mask_images = []
        self.gt_images = []

        self.input_size_H = input_size_H
        self.input_size_W = input_size_W
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.gt_dir = gt_dir
        self.transformI = transformI
        self.transformM = transformM
        self.transformG = transformG

        if self.transformI:
            self.ti = self.transformI
        else:
            self.ti = torchvision.transforms.Compose([
                # FixResize(input_size_W, input_size_H, interpolation=cv2.INTER_LINEAR),
                Normalize(mean=1.0, std=127.5),
                ToTensor()
            ])

        if self.transformM:
            self.tm = self.transformM
        else:
            self.tm = torchvision.transforms.Compose([
                # FixResize(input_size_W, input_size_H, interpolation=cv2.INTER_NEAREST),
                Normalize(mean=1.0, std=127.5),
                ToTensor()
            ])

        if self.transformG:
            self.tg = self.transformG
        else:
            self.tg = torchvision.transforms.Compose([
                # FixResize(input_size_W, input_size_H, interpolation=cv2.INTER_LINEAR),
                Normalize(mean=1.0, std=127.5),
                ToTensor()
            ])

        for name_img in os.listdir(input_dir):
            # i1 = Image.open(os.path.join(input_dir, name_img)).convert('RGB')
            # self.input_images.append(i1)
            # m1 = Image.open(os.path.join(mask_dir, name_img)).convert('RGB')
            # self.mask_images.append(m1)
            # g1 = Image.open(os.path.join(gt_dir, name_img)).convert('RGB')
            # self.gt_images.append(g1)

            i1 = os.path.join(input_dir, name_img)
            self.input_images.append(i1)
            m1 = os.path.join(mask_dir, name_img)
            self.mask_images.append(m1)
            g1 = os.path.join(gt_dir, name_img)
            self.gt_images.append(g1)

    def __len__(self):

        return len(self.input_images)

    def __getitem__(self, index):
        # i1 = self.input_images[index]
        # m1 = self.mask_images[index]
        # g1 = self.gt_images[index]

        # i1 = Image.open(self.input_images[index]).convert('RGB')
        # m1 = Image.open(self.mask_images[index]).convert('RGB')
        # g1 = Image.open(self.gt_images[index]).convert('RGB')

        i1 = cv2.imread(self.input_images[index])
        m1 = cv2.imread(self.mask_images[index])
        _, m1 = cv2.threshold(m1, 127, 255, cv2.THRESH_BINARY)  # mask is not clean
        # i1[m1[..., 2] < 1] = 0  # input background 255 to 0
        g1 = cv2.imread(self.gt_images[index])

        seed = np.random.randint(0, 2 ** 32)  # make a seed with numpy generator

        # apply this seed to img transforms
        set_seed(seed)
        img = self.ti(i1)

        # apply this seed to mask transforms
        set_seed(seed)
        mask = self.tm(m1)

        # apply this seed to mask transforms
        set_seed(seed)
        gt = self.tg(g1)

        sample = {'image': img, 'mask': mask, 'gt': gt}

        return sample


class Images_Dataset_test(Dataset):
    """Class for getting individual transformations and data
    Args:
        input_dir = path of input images
        mask_dir = path of mask images
        gt_dir = path of groundtruth images
        transformI = Input Image transformation (default: None)
        transformM = Input Mask transformation (default: None)
        transformG = Input Groundtruth transformation (default: None)
    Output:
        ix = Transformed input images
        mx = Transformed mask images
        gx = Transformed groundtruth images
        """

    def __init__(self, input_dir, mask_dir, gt_dir, input_size_H=384, input_size_W=512,
                 transformI=None, transformM=None, transformG=None):
        self.input_images = []
        self.mask_images = []
        self.gt_images = []

        self.input_size_H = input_size_H
        self.input_size_W = input_size_W
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.gt_dir = gt_dir
        self.transformI = transformI
        self.transformM = transformM
        self.transformG = transformG

        if self.transformI:
            self.ti = self.transformI
        else:
            self.ti = torchvision.transforms.Compose([
                # FixResize(input_size_W, input_size_H, interpolation=cv2.INTER_LINEAR),
                Normalize(mean=1.0, std=127.5),
                ToTensor()
            ])

        if self.transformM:
            self.tm = self.transformM
        else:
            self.tm = torchvision.transforms.Compose([
                # FixResize(input_size_W, input_size_H, interpolation=cv2.INTER_NEAREST),
                Normalize(mean=1.0, std=127.5),
                ToTensor()
            ])

        if self.transformG:
            self.tg = self.transformG
        else:
            self.tg = torchvision.transforms.Compose([
                # FixResize(input_size_W, input_size_H, interpolation=cv2.INTER_LINEAR),
                Normalize(mean=1.0, std=127.5),
                ToTensor()
            ])

        path_list = os.listdir(input_dir)
        path_list.sort()
        for name_img in path_list:
            # i1 = Image.open(os.path.join(input_dir, name_img)).convert('RGB')
            # self.input_images.append(i1)
            # m1 = Image.open(os.path.join(mask_dir, name_img)).convert('RGB')
            # self.mask_images.append(m1)
            # g1 = Image.open(os.path.join(gt_dir, name_img)).convert('RGB')
            # self.gt_images.append(g1)

            i1 = os.path.join(input_dir, name_img)
            self.input_images.append(i1)
            m1 = os.path.join(mask_dir, name_img)
            self.mask_images.append(m1)
            g1 = os.path.join(gt_dir, name_img)
            self.gt_images.append(g1)

    def __len__(self):

        return len(self.input_images)

    def __getitem__(self, index):
        # i1 = self.input_images[index]
        # m1 = self.mask_images[index]
        # g1 = self.gt_images[index]

        # i1 = Image.open(self.input_images[index]).convert('RGB')
        # m1 = Image.open(self.mask_images[index]).convert('RGB')
        # g1 = Image.open(self.gt_images[index]).convert('RGB')

        i1 = cv2.imread(self.input_images[index])
        m1 = cv2.imread(self.mask_images[index])
        _, m1 = cv2.threshold(m1, 127, 255, cv2.THRESH_BINARY)  # mask is not clean
        g1 = cv2.imread(self.gt_images[index])

        seed = np.random.randint(0, 2 ** 32)  # make a seed with numpy generator

        # apply this seed to img transforms
        set_seed(seed)
        img = self.ti(i1)

        # apply this seed to mask transforms
        set_seed(seed)
        mask = self.tm(m1)

        # apply this seed to mask transforms
        set_seed(seed)
        gt = self.tg(g1)

        sample = {'image': img, 'mask': mask, 'gt': gt, 'name': self.input_images[index].split('/')[-1]}

        return sample
