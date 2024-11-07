#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/25 16:00
# @Author  : yyywxk
# @File    : spatial_transform_local.py

import torch
import torch.nn.functional as F

from . import tensorDLT_local
# import tensorDLT_local


def transformer(U, mask, theta, grid_h, grid_w, cuda=False, name='SpatialTransformer', **kwargs):
    grid_w = grid_w
    grid_h = grid_h

    def _repeat(x, n_repeats):
        rep_transpose = torch.unsqueeze(torch.ones(n_repeats, ), 1)
        rep = rep_transpose.permute(1, 0).long()
        x = torch.matmul(x.view(-1, 1), rep)
        return x.view(-1)

    def _interpolate(im, x, y, out_size, cuda=False):
        # constants
        # torch image: C X H X W
        num_batch = im.shape[0]
        height = im.shape[2]
        width = im.shape[3]
        channels = im.shape[1]
        im_t = im.permute(0, 2, 3, 1).contiguous()

        x = x.float()
        y = y.float()
        # height_f = height.float()
        # width_f = width.float()
        out_height = out_size[0]
        out_width = out_size[1]
        zero = 0
        max_y = height - 1
        max_x = width - 1

        # do sampling
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        dim2 = width
        dim1 = width * height
        base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width)
        if cuda:
            base = base.cuda()
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        # im_flat = im.view(-1, channels).float()
        im_flat = im_t.view(-1, channels).float()
        # Ia = tf.gather(im_flat, idx_a)
        Ia = im_flat[idx_a]
        # Ib = tf.gather(im_flat, idx_b)
        Ib = im_flat[idx_b]
        # Ic = tf.gather(im_flat, idx_c)
        Ic = im_flat[idx_c]
        # Id = tf.gather(im_flat, idx_d)
        Id = im_flat[idx_d]

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return output

    # input:  batch_size*2*(grid_h+1)*(grid_w+1)
    # output: batch_size*9*grid_h*grid_w
    def get_Hs(theta, width, height, cuda=False):
        num_batch = theta.shape[0]
        h = height / grid_h
        w = width / grid_w
        Hs = []
        for i in range(grid_h):
            for j in range(grid_w):
                hh = i * h
                ww = j * w
                ori_repeat = torch.Tensor([ww, hh, ww + w, hh, ww, hh + h, ww + w, hh + h]).view(1, 8)
                ori = ori_repeat.repeat(num_batch, 1)
                if cuda:
                    ori = ori.cuda()

                # id = i * (grid_w + 1) + grid_w
                # tar_cat = (theta[:, i, j, :], theta[:, i, j + 1, :], theta[:, i + 1, j, :], theta[:, i + 1, j + 1, :])
                tar_cat = (
                    theta[:, :, i:i + 1, j:j + 1], theta[:, :, i:i + 1, j + 1:j + 2],
                    theta[:, :, i + 1:i + 2, j:j + 1], theta[:, :, i + 1:i + 2, j + 1:j + 2])
                tar = torch.concat(tar_cat, 2).permute(0, 2, 3, 1).contiguous()
                # tar = tf.concat([tf.slice(theta, [0, i, j, 0], [-1, 1, 1, -1]),
                #                  tf.slice(theta, [0, i, j + 1, 0], [-1, 1, 1, -1]),
                #                  tf.slice(theta, [0, i + 1, j, 0], [-1, 1, 1, -1]),
                #                  tf.slice(theta, [0, i + 1, j + 1, 0], [-1, 1, 1, -1])], axis=1)
                tar = tar.view(num_batch, 8)
                # tar = tf.Print(tar, [tf.slice(ori, [0, 0], [1, -1])],message="[ori--i:"+str(i)+",j:"+str(j)+"]:", summarize=100,first_n=5)
                # tar = tf.Print(tar, [tf.slice(tar, [0, 0], [1, -1])],message="[tar--i:"+str(i)+",j:"+str(j)+"]:", summarize=100,first_n=5)
                # Hs.append(tf.reshape(tensorDLT_local.solve_DLT(ori, tar), [num_batch, 1, 9]))
                Hs.append(tensorDLT_local.solve_DLT(ori, tar, cuda=cuda).view(num_batch, 9, 1))
        # Hs = tf.reshape(tf.concat(Hs, axis=1), [num_batch, grid_h, grid_w, 9], name='Hs')
        Hs = torch.concat(Hs, 2).view(num_batch, 9, grid_h, grid_w)
        # Hs = torch.concat(Hs, 1).reshape(num_batch, 9, grid_h, grid_w)

        return Hs

    def _meshgrid(height, width):
        x_t_mul_expan = torch.unsqueeze(torch.linspace(0., width - 1.001, width), 1)
        x_t_mul = x_t_mul_expan.permute(1, 0)
        x_t = torch.matmul(torch.ones((height, 1)), x_t_mul)
        # x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
        #                 tf.transpose(tf.expand_dims(tf.linspace(0., tf.cast(width, 'float32') - 1.001, width), 1),
        #                              [1, 0]))
        # y_t = tf.matmul(tf.expand_dims(tf.linspace(0., tf.cast(height, 'float32') - 1.001, height), 1),
        #                 tf.ones(shape=tf.stack([1, width])))
        y_t_mul = torch.unsqueeze(torch.linspace(0., height - 1.001, height), 1)
        y_t = torch.matmul(y_t_mul, torch.ones((1, width)))

        x_t_flat = x_t.view(1, -1)
        y_t_flat = y_t.view(1, -1)

        ones = torch.ones_like(x_t_flat)
        grid = torch.concat((x_t_flat, y_t_flat, ones), 0)

        return grid

    def _transform3(theta, input_dim, mask, cuda=False):
        # torch image: C X H X W
        num_batch = input_dim.shape[0]
        height = input_dim.shape[2]
        width = input_dim.shape[3]
        num_channels = input_dim.shape[1]

        # the width/height should be an an integral multiple of grid_w/grid_h
        width_float = float(width)
        height_float = float(height)

        Hs = get_Hs(theta, width_float, height_float, cuda=cuda)
        # gh = torch.Tensor(height / grid_h).int()
        # gw = torch.Tensor(width / grid_w).int()
        ##########################################
        # print("Hs")
        # print(Hs.shape)
        # H_array = UpSampling2D(size=(384 / grid_h, 512 / grid_w))(Hs)
        H_array = F.interpolate(Hs, scale_factor=(height / grid_h, width / grid_w), mode='nearest')
        H_array = H_array.permute(0, 2, 3, 1).contiguous().view(-1, 3, 3)
        ##########################################

        out_height = height
        out_width = width
        grid = _meshgrid(out_height, out_width)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.view(-1)
        grid = grid.repeat(num_batch)  # stack num_batch grids
        grid = grid.view(num_batch, 3, -1)
        # grid = tf.tile(grid, tf.stack([num_batch]))  # stack num_batch grids
        # grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))
        # print("grid")
        # print(grid.shape)
        ### [bs, 3, N]

        # grid = tf.expand_dims(tf.transpose(grid, [0, 2, 1]), 3)
        grid = torch.unsqueeze(grid.permute(0, 2, 1), 3)
        ### [bs, 3, N] -> [bs, N, 3] -> [bs, N, 3, 1]
        grid = grid.contiguous().view(-1, 3, 1)
        # grid = tf.reshape(grid, [-1, 3, 1])
        ### [bs*N, 3, 1]

        grid_row = grid.view(-1, 3)
        if cuda:
            grid_row = grid_row.cuda()
        # print("grid_row")
        # print(grid_row.shape)
        x_s = torch.sum(torch.mul(H_array[:, 0, :], grid_row), 1)
        y_s = torch.sum(torch.mul(H_array[:, 1, :], grid_row), 1)
        t_s = torch.sum(torch.mul(H_array[:, 2, :], grid_row), 1)

        # The problem may be here as a general homo does not preserve the parallelism
        # while an affine transformation preserves it.
        t_s_flat = t_s.view(-1)
        t_1 = torch.ones_like(t_s_flat)
        t_0 = torch.zeros_like(t_s_flat)
        sign_t = torch.where(t_s_flat >= 0, t_1, t_0) * 2 - 1
        t_s_flat = t_s_flat + sign_t * 1e-8

        x_s_flat = x_s.view(-1) / t_s_flat
        y_s_flat = y_s.view(-1) / t_s_flat

        out_size = (height, width)
        input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_size, cuda=cuda)
        mask_transformed = _interpolate(mask, x_s_flat, y_s_flat, out_size, cuda=cuda)

        # warp_image = input_transformed.view(num_batch, num_channels, height, width)
        # warp_mask = mask_transformed.view(num_batch, num_channels, height, width)
        warp_image = input_transformed.view(num_batch, height, width, -1)
        warp_mask = mask_transformed.view(num_batch, height, width, -1)

        return warp_image.permute(0, 3, 1, 2), warp_mask.permute(0, 3, 1, 2)

    # output = _transform(theta, U, out_size)
    U = U - 1.
    warp_image, warp_mask = _transform3(theta, U, mask, cuda=cuda)
    warp_image = warp_image + 1.
    warp_image = torch.clamp(warp_image, -1, 1)
    return warp_image, warp_mask


if __name__ == '__main__':
    import cv2
    import numpy as np

    i1 = cv2.imread('00001.jpg')
    i1 = np.array(i1).astype(np.float32)
    i1 /= 127.5
    i1 -= 1.0

    m1 = cv2.imread('00002.jpg')
    # _, m1 = cv2.threshold(m1, 127, 255, cv2.THRESH_BINARY)  # mask is not clean
    m1 = np.array(m1).astype(np.float32)
    m1 /= 127.5
    m1 -= 1.0

    img = np.array(i1).astype(np.float32).transpose((2, 0, 1))
    img = torch.from_numpy(img.astype(np.float32)).float()
    img = torch.unsqueeze(img, 0)

    img1 = np.array(m1).astype(np.float32).transpose((2, 0, 1))
    img1 = torch.from_numpy(img1.astype(np.float32)).float()
    img1 = torch.unsqueeze(img1, 0)

    batch_size = 1
    height = 384
    width = 512
    grid_h = 6
    grid_w = 8
    h = height / grid_h
    w = width / grid_w
    ori_pt = []
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            ww = j * w
            hh = i * h
            p = torch.Tensor([ww, hh])
            ori_pt.append(torch.unsqueeze(p, 0))

    ori_pt = torch.cat(ori_pt, 0)
    ori_pt = ori_pt.view(grid_h + 1, grid_w + 1, 2).permute(2, 0, 1)
    ori_pt = torch.unsqueeze(ori_pt, 0)
    ori_pt = ori_pt.repeat(batch_size, 1, 1, 1)

    feature_warp, feature_warp_mask = transformer(img, img1, ori_pt, grid_w=grid_w, grid_h=grid_h, cuda=False)

    test_warp_image = (feature_warp_mask[0] + 1.0) * 127.5

    result = test_warp_image.permute(1, 2, 0).numpy().astype('uint8')  # cv2.imread
    cv2.imwrite('result.jpg', result)
