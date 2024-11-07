#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/23 19:10
# @Author  : yyywxk
# @File    : model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import spatial_transform_local
from . import spatial_transform_local_feature
import time


def shift2mesh(mesh_shift, width, height, grid_w, grid_h, cuda=False):
    '''

    :param mesh_shift: tensor
    :param width:
    :param height:
    :param grid_w:
    :param grid_h:
    :return:
    '''
    batch_size = mesh_shift.shape[0]
    h = height / grid_h
    w = width / grid_w
    ori_pt = []
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            ww = j * w
            hh = i * h
            # p = tf.constant([ww, hh], shape=[2], dtype=tf.float32)
            p = torch.Tensor([ww, hh])
            ori_pt.append(torch.unsqueeze(p, 0))
            # ori_pt.append(tf.expand_dims(p, 0))
    # ori_pt = tf.concat(ori_pt, axis=0)
    # ori_pt = tf.reshape(ori_pt, [grid_h + 1, grid_w + 1, 2])
    # ori_pt = tf.tile(tf.expand_dims(ori_pt, 0), [batch_size, 1, 1, 1])
    ori_pt = torch.cat(ori_pt, 0)
    ori_pt = ori_pt.view(grid_h + 1, grid_w + 1, 2).permute(2, 0, 1)
    # ori_pt = ori_pt.reshape(2, grid_h + 1, grid_w + 1)
    ori_pt = torch.unsqueeze(ori_pt, 0)
    ori_pt = ori_pt.repeat(batch_size, 1, 1, 1)
    if cuda:
        ori_pt = ori_pt.cuda()
    tar_pt = ori_pt + mesh_shift

    return tar_pt


# feature extraction module
class feature_extractor(nn.Module):
    # torch image: C X H X W
    def __init__(self, n_colors):
        super(feature_extractor, self).__init__()
        self.conv = nn.Sequential(
            # 384*512
            nn.Conv2d(in_channels=n_colors+3, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 192*256
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 96*128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 48*64
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# mesh motion regression module
class regression_Net(nn.Module):
    def __init__(self, grid_h, grid_w):
        super(regression_Net, self).__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.conv = nn.Sequential(
            # 128*24*32
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 256*12*16
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 512*6*8
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 512*3*4
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=[3, 4], padding=0, bias=True),
            nn.ReLU(inplace=True),
            # 2048*1*1
            nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            # 1024*1*1
            nn.Conv2d(in_channels=1024, out_channels=(grid_w + 1) * (grid_h + 1) * 2, kernel_size=1, padding=0,
                      bias=True),
            # (U+1)*(V+1)*2
        )

    def forward(self, x):
        x = self.conv(x)
        # return x.reshape(-1, self.grid_h + 1, self.grid_w + 1, 2)

        return x.view(-1, 2, self.grid_h + 1, self.grid_w + 1)


class RectanglingNetwork(nn.Module):
    # torch image: C X H X W
    def __init__(self, n_colors, grid_h, grid_w, width=512., height=384., cuda_flag=False):
        super(RectanglingNetwork, self).__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.width = width
        self.height = height
        # bulit model
        self.feature_extract = feature_extractor(n_colors)
        self.regressor_coarse = regression_Net(grid_h=grid_h, grid_w=grid_w)
        self.regressor_fine = regression_Net(grid_h=grid_h, grid_w=grid_w)
        self.cuda_flag = cuda_flag

    def forward(self, train_input, train_mask, train_flag=True):
        features = self.feature_extract(torch.concat([train_input, train_mask], 1))

        feature = F.interpolate(features, size=(24, 32), mode='bilinear')

        mesh_shift_primary = self.regressor_coarse(feature)
        mesh_primary = shift2mesh(mesh_shift_primary / 16, 32., 24., grid_w=self.grid_w, grid_h=self.grid_h,
                                  cuda=self.cuda_flag)

        feature_warp = spatial_transform_local_feature.transformer(feature, mesh_primary, grid_w=self.grid_w,
                                                                   grid_h=self.grid_h, cuda=self.cuda_flag)

        mesh_shift_final = self.regressor_fine(feature_warp)

        mesh_primary = shift2mesh(mesh_shift_primary, width=self.width, height=self.height, grid_w=self.grid_w,
                                  grid_h=self.grid_h, cuda=self.cuda_flag)
        mesh_final = shift2mesh(mesh_shift_final + mesh_shift_primary, width=self.width, height=self.height,
                                grid_w=self.grid_w, grid_h=self.grid_h, cuda=self.cuda_flag)

        warp_image_primary, warp_mask_primary = spatial_transform_local.transformer(train_input, train_mask,
                                                                                    mesh_primary, grid_w=self.grid_w,
                                                                                    grid_h=self.grid_h,
                                                                                    cuda=self.cuda_flag)

        warp_image_final, warp_mask_final = spatial_transform_local.transformer(train_input, train_mask, mesh_final,
                                                                                grid_w=self.grid_w, grid_h=self.grid_h,
                                                                                cuda=self.cuda_flag)

        if train_flag:
            return mesh_primary, warp_image_primary, warp_mask_primary, mesh_final, warp_image_final, warp_mask_final
        else:
            return warp_image_final, warp_mask_final


if __name__ == '__main__':
    train_input = 2 * torch.rand(3, 3, 384, 512) - 1
    train_mask = 2 * torch.rand(3, 3, 384, 512) - 1
    model = RectanglingNetwork(width=512., height=384., grid_w=8, grid_h=6)
    mesh_primary, warp_image_primary, warp_mask_primary, mesh_final, warp_image_final, warp_mask_final = model(
        train_input, train_mask)
    print(mesh_primary.shape, warp_image_primary.shape, warp_mask_primary.shape, mesh_final.shape,
          warp_image_final.shape, warp_mask_final.shape)
