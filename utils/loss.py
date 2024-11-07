#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/26 16:17
# @Author  : yyywxk
# @File    : loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .vgg_arch import VGGFeatureExtractor


def l1_loss(pred, target, reduction='none'):
    return F.l1_loss(pred, target, reduction=reduction)


def mse_loss(pred, target, reduction='none'):
    return F.mse_loss(pred, target, reduction=reduction)


# pixel-level loss (l_num=1 for L1 loss, l_num=2 for L2 loss, ......)
class intensity_loss(nn.Module):
    def __init__(self, l_num=1, reduction='mean'):
        super(intensity_loss, self).__init__()
        self.l_num = l_num
        self.reduction = reduction

    def forward(self, gen_frames, gt_frames):
        '''

        :param gen_frames: (Tensor) of shape (N, C, H, W). Predicted tensor.
        :param gt_frames: (Tensor) of shape (N, C, H, W). Ground truth tensor.
        :return:
        '''
        if self.l_num == 1:
            return l1_loss(gen_frames, gt_frames, reduction=self.reduction)
        elif self.l_num == 2:
            return mse_loss(gen_frames, gt_frames, reduction=self.reduction)
        else:
            raise ValueError('=> Please input correct l_num (l_num=1 for L1 loss, l_num=2 for L2 loss, ......)')


# intra-grid constraint
class intra_grid_loss(nn.Module):
    def __init__(self, grid_w=8, grid_h=6, w=256, h=384):
        super(intra_grid_loss, self).__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.relu = nn.ReLU(inplace=True)
        # self.min_w = (w / grid_w) / 8
        # self.min_h = (h / grid_h) / 8
        v0, v1 = -1, 1
        rh = (v1 - v0) / (h)
        rw = (v1 - v0) / (w)
        self.min_w = (w / grid_w) / 8 * rw
        self.min_h = (h / grid_h) / 8 * rh

    def forward(self, pts):
        '''

        :param pts: (Tensor): of shape (N, H, W, 2) - (N, 2, H, W)
        :return:
        '''
        delta_x = pts[:, 0, :, 0:self.grid_w] - pts[:, 0, :, 1:self.grid_w + 1]
        delta_y = pts[:, 1, 0:self.grid_h, :] - pts[:, 1, 1:self.grid_h + 1, :]

        loss_x = self.relu(delta_x + self.min_w)
        loss_y = self.relu(delta_y + self.min_h)

        loss = torch.mean(loss_x) + torch.mean(loss_y)
        return loss


# inter-grid constraint
class inter_grid_loss(nn.Module):
    def __init__(self, grid_w=8, grid_h=6):
        super(inter_grid_loss, self).__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w

    def forward(self, train_mesh):
        '''

        :param train_mesh: (Tensor): of shape (N, H, W, 2) - (N, 2, H, W)
        :return:
        '''
        w_edges = train_mesh[:, :, :, 0:self.grid_w] - train_mesh[:, :, :, 1:self.grid_w + 1]

        cos_w = torch.sum(w_edges[:, :, :, 0:self.grid_w - 1] * w_edges[:, :, :, 1:self.grid_w], 1) / (torch.sqrt(
            torch.sum(w_edges[:, :, :, 0:self.grid_w - 1] * w_edges[:, :, :, 0:self.grid_w - 1], 1)) * torch.sqrt(
            torch.sum(w_edges[:, :, :, 1:self.grid_w] * w_edges[:, :, :, 1:self.grid_w], 1)))
        # print("cos_w.shape")
        # print(cos_w.shape)
        delta_w_angle = 1 - cos_w

        h_edges = train_mesh[:, :, 0:self.grid_h, :] - train_mesh[:, :, 1:self.grid_h + 1, :]
        cos_h = torch.sum(h_edges[:, :, 0:self.grid_h - 1, :] * h_edges[:, :, 1:self.grid_h, :], 1) / (torch.sqrt(
            torch.sum(h_edges[:, :, 0:self.grid_h - 1, :] * h_edges[:, :, 0:self.grid_h - 1, :], 1)) * torch.sqrt(
            torch.sum(h_edges[:, :, 1:self.grid_h, :] * h_edges[:, :, 1:self.grid_h, :], 1)))
        delta_h_angle = 1 - cos_h

        loss = torch.mean(delta_w_angle) + torch.mean(delta_h_angle)

        return loss


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self, layer_weights, vgg_type='vgg19', use_input_norm=True, range_norm=False, criterion='l2',
                 cuda=False):
        super(PerceptualLoss, self).__init__()
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(layer_name_list=list(layer_weights.keys()), vgg_type=vgg_type,
                                       use_input_norm=use_input_norm,
                                       range_norm=range_norm)
        if cuda:
            self.vgg = self.vgg.cuda()
        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = intensity_loss(l_num=1)
        elif self.criterion_type == 'l2':
            self.criterion = intensity_loss(l_num=2)
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        percep_loss = 0
        for k in x_features.keys():
            if self.criterion_type == 'fro':
                percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
            else:
                percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]

        return percep_loss


class RectanglingLosses(object):
    def __init__(self, args, cuda_flag=False, grid_w=8, grid_h=6):
        self.args = args
        self.cuda_flag = cuda_flag
        self.grid_h = grid_h
        self.grid_w = grid_w

    def build_loss(self, mode='8terms'):
        if mode == 'l1':
            return intensity_loss(l_num=1)
        elif mode == 'l2':
            return intensity_loss(l_num=2)

        # intensity_loss
        self.intensity_loss = intensity_loss(l_num=1)
        # self.intensity_loss = intensity_loss(l_num=2)
        if mode == '2terms':
            return self._2termsLoss
        elif mode == '2terms+':
            return self._2terms_Loss

        # Perceptual Loss
        self.PerceptualLoss = PerceptualLoss({'conv4_2': 1.}, range_norm=True, cuda=self.cuda_flag)
        if mode == '6terms':
            return self._6termsLoss
        elif mode == '4terms':
            return self._4termsLoss

        # intra_grid_loss
        self.intra_loss = intra_grid_loss(grid_w=self.grid_w, grid_h=self.grid_h,
                                          w=self.args.width, h=self.args.height)
        # inter_grid_loss
        self.inter_loss = inter_grid_loss(grid_w=self.grid_w, grid_h=self.grid_h)

        if mode == '8terms':
            return self._8termsLoss

    def _8termsLoss(self, train_mesh_primary, train_warp_image_primary, train_warp_mask_primary, train_mesh_final,
                    train_warp_image_final, train_warp_mask_final, train_gt,
                    lam_appearance, lam_perception, lam_mask, lam_mesh):
        if self.cuda_flag:
            train_mesh_primary = train_mesh_primary.cuda()
            train_warp_image_primary = train_warp_image_primary.cuda()
            train_warp_mask_primary = train_warp_mask_primary.cuda()
            train_mesh_final = train_mesh_final.cuda()
            train_warp_image_final = train_warp_image_final.cuda()
            train_warp_mask_final = train_warp_mask_final.cuda()
            train_gt = train_gt.cuda()

        # content term
        # define appearance loss (loss 1 of of the content term)
        # lam_appearance = 1
        primary_appearance_loss = self.intensity_loss(train_warp_image_primary, train_gt)
        final_appearance_loss = self.intensity_loss(train_warp_image_final, train_gt)
        appearance_loss = primary_appearance_loss + final_appearance_loss

        # define perception loss (loss 2 of of the content term)
        # lam_perception = 5e-6
        # lam_perception = 5e-4
        primary_perception_loss = self.PerceptualLoss(train_warp_image_primary, train_gt)
        final_perception_loss = self.PerceptualLoss(train_warp_image_final, train_gt)
        perception_loss = primary_perception_loss + final_perception_loss

        # define boundary term
        # lam_mask = 1
        primary_mask_loss = self.intensity_loss(train_warp_mask_primary, torch.ones_like(
            train_warp_mask_primary).cuda() if self.cuda_flag else torch.ones_like(train_warp_mask_primary))
        final_mask_loss = self.intensity_loss(train_warp_mask_final, torch.ones_like(
            train_warp_mask_final).cuda() if self.cuda_flag else torch.ones_like(train_warp_mask_final))
        mask_loss = primary_mask_loss + final_mask_loss

        # define mesh term
        # lam_mesh = 1
        primary_mesh_loss = self.intra_loss(train_mesh_primary) + self.inter_loss(train_mesh_primary)
        final_mesh_loss = self.intra_loss(train_mesh_final) + self.inter_loss(train_mesh_final)
        mesh_loss = primary_mesh_loss + final_mesh_loss

        # print('global', time.time())
        g_loss = appearance_loss * lam_appearance + perception_loss * lam_perception \
                 + lam_mask * mask_loss + mesh_loss * lam_mesh

        return g_loss, appearance_loss, perception_loss, mask_loss, mesh_loss

    def _6termsLoss(self, train_warp_image_primary, train_warp_mask_primary, train_warp_image_final,
                    train_warp_mask_final, train_gt, lam_appearance, lam_perception, lam_mask):
        if self.cuda_flag:
            train_warp_image_primary = train_warp_image_primary.cuda()
            train_warp_mask_primary = train_warp_mask_primary.cuda()
            train_warp_image_final = train_warp_image_final.cuda()
            train_warp_mask_final = train_warp_mask_final.cuda()
            train_gt = train_gt.cuda()

        # content term
        # define appearance loss (loss 1 of of the content term)
        # lam_appearance = 1
        primary_appearance_loss = self.intensity_loss(train_warp_image_primary, train_gt)
        final_appearance_loss = self.intensity_loss(train_warp_image_final, train_gt)
        appearance_loss = primary_appearance_loss + final_appearance_loss

        # define perception loss (loss 2 of of the content term)
        # lam_perception = 5e-6
        # lam_perception = 5e-4
        primary_perception_loss = self.PerceptualLoss(train_warp_image_primary, train_gt)
        final_perception_loss = self.PerceptualLoss(train_warp_image_final, train_gt)
        perception_loss = primary_perception_loss + final_perception_loss

        # define boundary term
        # lam_mask = 1
        primary_mask_loss = self.intensity_loss(train_warp_mask_primary, torch.ones_like(
            train_warp_mask_primary).cuda() if self.cuda_flag else torch.ones_like(train_warp_mask_primary))
        final_mask_loss = self.intensity_loss(train_warp_mask_final, torch.ones_like(
            train_warp_mask_final).cuda() if self.cuda_flag else torch.ones_like(train_warp_mask_final))
        mask_loss = primary_mask_loss + final_mask_loss

        # print('global', time.time())
        g_loss = appearance_loss * lam_appearance + perception_loss * lam_perception \
                 + lam_mask * mask_loss

        return g_loss, appearance_loss, perception_loss, mask_loss

    def _4termsLoss(self, train_warp_image_primary, train_warp_mask_primary, train_warp_image_final,
                    train_warp_mask_final, train_gt, lam_appearance, lam_perception):
        if self.cuda_flag:
            train_warp_image_primary = train_warp_image_primary.cuda()
            train_warp_mask_primary = train_warp_mask_primary.cuda()
            train_warp_image_final = train_warp_image_final.cuda()
            train_warp_mask_final = train_warp_mask_final.cuda()
            train_gt = train_gt.cuda()

        # content term
        # define appearance loss (loss 1 of of the content term)
        # lam_appearance = 1
        primary_appearance_loss = self.intensity_loss(train_warp_image_primary, train_gt)
        final_appearance_loss = self.intensity_loss(train_warp_image_final, train_gt)
        appearance_loss = primary_appearance_loss + final_appearance_loss

        # define perception loss (loss 2 of of the content term)
        # lam_perception = 5e-6
        # lam_perception = 5e-4
        primary_perception_loss = self.PerceptualLoss(train_warp_image_primary, train_gt)
        final_perception_loss = self.PerceptualLoss(train_warp_image_final, train_gt)
        perception_loss = primary_perception_loss + final_perception_loss

        # print('global', time.time())
        g_loss = appearance_loss * lam_appearance + perception_loss * lam_perception

        return g_loss, appearance_loss, perception_loss

    def _2terms_Loss(self, train_warp_image_primary, train_warp_mask_primary, train_warp_image_final,
                     train_warp_mask_final, train_gt, lam_appearance, lam_mask):
        if self.cuda_flag:
            train_warp_image_primary = train_warp_image_primary.cuda()
            train_warp_mask_primary = train_warp_mask_primary.cuda()
            train_warp_image_final = train_warp_image_final.cuda()
            train_warp_mask_final = train_warp_mask_final.cuda()
            train_gt = train_gt.cuda()

        # content term
        # define appearance loss (loss 1 of of the content term)
        # lam_appearance = 1
        primary_appearance_loss = self.intensity_loss(train_warp_image_primary, train_gt)
        final_appearance_loss = self.intensity_loss(train_warp_image_final, train_gt)
        appearance_loss = primary_appearance_loss + final_appearance_loss

        # define boundary term
        # lam_mask = 1
        primary_mask_loss = self.intensity_loss(train_warp_mask_primary, torch.ones_like(
            train_warp_mask_primary).cuda() if self.cuda_flag else torch.ones_like(train_warp_mask_primary))
        final_mask_loss = self.intensity_loss(train_warp_mask_final, torch.ones_like(
            train_warp_mask_final).cuda() if self.cuda_flag else torch.ones_like(train_warp_mask_final))
        mask_loss = primary_mask_loss + final_mask_loss

        # print('global', time.time())
        g_loss = appearance_loss * lam_appearance + lam_mask * mask_loss

        return g_loss, appearance_loss, mask_loss

    def _2termsLoss(self, train_warp_image_primary, train_warp_mask_primary, train_warp_image_final,
                    train_warp_mask_final, train_gt, lam_appearance):
        if self.cuda_flag:
            train_warp_image_primary = train_warp_image_primary.cuda()
            train_warp_mask_primary = train_warp_mask_primary.cuda()
            train_warp_image_final = train_warp_image_final.cuda()
            train_warp_mask_final = train_warp_mask_final.cuda()
            train_gt = train_gt.cuda()

        # content term
        # define appearance loss (loss 1 of of the content term)
        # lam_appearance = 1
        primary_appearance_loss = self.intensity_loss(train_warp_image_primary, train_gt)
        final_appearance_loss = self.intensity_loss(train_warp_image_final, train_gt)
        appearance_loss = primary_appearance_loss + final_appearance_loss

        # print('global', time.time())
        g_loss = appearance_loss * lam_appearance

        return g_loss, appearance_loss


if __name__ == '__main__':
    # a = torch.rand(2, 3, 7, 7)
    # b = torch.rand(2, 3, 7, 7)

    # my_loss = intensity_loss(l_num=1)

    # a = torch.rand(3, 7, 9, 2)
    # b = torch.rand(3, 7, 9, 2)
    # my_loss = inter_grid_loss()

    # train_input = 2 * torch.rand(3, 3, 384, 512) - 1
    # train_mask = 2 * torch.rand(3, 3, 384, 512) - 1
    # my_loss = PerceptualLoss({'conv4_2': 1.}, range_norm=True)
    # print(my_loss(train_input, train_mask))

    train_mesh_primary = torch.rand(3, 7, 9, 2)
    train_warp_image_primary = 2 * torch.rand(3, 3, 384, 512) - 1
    train_warp_mask_primary = 2 * torch.rand(3, 3, 384, 512) - 1
    train_mesh_final = torch.rand(3, 7, 9, 2)
    train_warp_image_final = 2 * torch.rand(3, 3, 384, 512) - 1
    train_warp_mask_final = 2 * torch.rand(3, 3, 384, 512) - 1
    train_gt = 2 * torch.rand(3, 3, 384, 512) - 1
    my_loss = RectanglingLosses().build_loss()
    print(my_loss(train_mesh_primary, train_warp_image_primary, train_warp_mask_primary, train_mesh_final,
                  train_warp_image_final, train_warp_mask_final, train_gt))
