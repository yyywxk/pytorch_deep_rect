#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/24 20:44
# @Author  : yyywxk
# @File    : draw_mesh.py

import argparse
import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import time
import shutil

import torch

from dataloaders import make_data_loader
from models.deeprect.model import RectanglingNetwork

import cv2 as cv

# os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # set the GPUs

import warnings

warnings.filterwarnings('ignore')

# --------------------------------- Homogeneous Transformation ----------------------------
# Auxiliary matrices used to solve DLT
Aux_M1 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.float64)

Aux_M2 = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float64)

Aux_M3 = np.array([
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1]], dtype=np.float64)

Aux_M4 = np.array([
    [-1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)

Aux_M5 = np.array([
    [0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float64)

Aux_M6 = np.array([
    [-1],
    [0],
    [-1],
    [0],
    [-1],
    [0],
    [-1],
    [0]], dtype=np.float64)

Aux_M71 = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.float64)

Aux_M72 = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, -1, 0]], dtype=np.float64)

Aux_M8 = np.array([
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, -1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, -1]], dtype=np.float64)

Aux_Mb = np.array([
    [0, -1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, -1],
    [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.float64)


def solve_HT(orig_pt4, pred_pt4, cuda=False):
    '''

    :param orig_pt4: BATCH_SIZE x 8
    :param pred_pt4: BATCH_SIZE x 8
    :return:
    '''
    batch_size = orig_pt4.shape[0]

    orig_pt4 = torch.unsqueeze(orig_pt4, 2)
    pred_pt4 = torch.unsqueeze(pred_pt4, 2)

    M1_tensor = torch.unsqueeze(torch.Tensor(Aux_M1), 0)
    M1_tile = M1_tensor.repeat(batch_size, 1, 1)

    M2_tensor = torch.unsqueeze(torch.Tensor(Aux_M2), 0)
    M2_tile = M2_tensor.repeat(batch_size, 1, 1)

    M3_tensor = torch.unsqueeze(torch.Tensor(Aux_M3), 0)
    M3_tile = M3_tensor.repeat(batch_size, 1, 1)

    M4_tensor = torch.unsqueeze(torch.Tensor(Aux_M4), 0)
    M4_tile = M4_tensor.repeat(batch_size, 1, 1)

    M5_tensor = torch.unsqueeze(torch.Tensor(Aux_M5), 0)
    M5_tile = M5_tensor.repeat(batch_size, 1, 1)

    M6_tensor = torch.unsqueeze(torch.Tensor(Aux_M6), 0)
    M6_tile = M6_tensor.repeat(batch_size, 1, 1)

    M71_tensor = torch.unsqueeze(torch.Tensor(Aux_M71), 0)
    M71_tile = M71_tensor.repeat(batch_size, 1, 1)

    M72_tensor = torch.unsqueeze(torch.Tensor(Aux_M72), 0)
    M72_tile = M72_tensor.repeat(batch_size, 1, 1)

    M8_tensor = torch.unsqueeze(torch.Tensor(Aux_M8), 0)
    M8_tile = M8_tensor.repeat(batch_size, 1, 1)

    Mb_tensor = torch.unsqueeze(torch.Tensor(Aux_Mb), 0)
    Mb_tile = Mb_tensor.repeat(batch_size, 1, 1)

    A1 = torch.matmul(M1_tile.cuda() if cuda else M1_tile, orig_pt4)  # Column 1
    A2 = torch.matmul(M2_tile.cuda() if cuda else M2_tile, orig_pt4)  # Column 2
    A3 = M3_tile.cuda() if cuda else M3_tile  # Column 3
    A4 = torch.matmul(M4_tile.cuda() if cuda else M4_tile, orig_pt4)  # Column 4
    A5 = torch.matmul(M5_tile.cuda() if cuda else M5_tile, orig_pt4)  # Column 5
    A6 = M6_tile.cuda() if cuda else M6_tile  # Column 6
    A7 = torch.matmul(M71_tile.cuda() if cuda else M71_tile, pred_pt4) * torch.matmul(
        M72_tile.cuda() if cuda else M72_tile, orig_pt4)  # Column 7
    A8 = torch.matmul(M71_tile.cuda() if cuda else M71_tile, pred_pt4) * torch.matmul(
        M8_tile.cuda() if cuda else M8_tile, orig_pt4)  # Column 8

    # A_mat: batch_size * 8 * 8
    A_mat_transpose = torch.stack((A1.reshape(-1, 8), A2.reshape(-1, 8), A3.reshape(-1, 8), A4.reshape(-1, 8),
                                   A5.reshape(-1, 8), A6.reshape(-1, 8), A7.reshape(-1, 8), A8.reshape(-1, 8)), dim=1)
    A_mat = A_mat_transpose.permute(0, 2, 1)  # BATCH_SIZE x 8 (A_i) x 8
    # print('--Shape of A_mat:', A_mat.shape)

    # Form b matrix
    b_mat = torch.matmul(Mb_tile.cuda() if cuda else Mb_tile, pred_pt4)
    # print('--shape of b:', b_mat.shape)

    # Solve the Ax = b
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "solve"):
        # PyTorch version >= 1.8.0
        H_8el = torch.linalg.solve(A_mat, b_mat)  # BATCH_SIZE x 8.
    else:
        H_8el = torch.solve(b_mat, A_mat).solution
    # print('--shape of H_8el', H_8el.shape)

    # Add ones to the last cols to reconstruct H for computing reprojection error
    h_ones = torch.ones(batch_size, 1, 1)
    if cuda:
        h_ones = h_ones.cuda()
    H_9el = torch.concat((H_8el, h_ones), 1)
    H_flat = H_9el.reshape(-1, 3, 3)

    # print('--shape of H_8el', H_flat.shape)

    # H_mat = tf.reshape(H_flat ,[-1 ,3 ,3])   # BATCH_SIZE x 3 x 3
    return H_flat


def cal_homography(mesh_primary, height, width, grid_h, grid_w, cuda=True):
    num_batch = mesh_primary.shape[0]
    h = int(height / grid_h)
    w = int(width / grid_w)
    rect_measure = torch.zeros(num_batch, grid_w * grid_h).cuda() if cuda else torch.zeros(num_batch, grid_w * grid_h)

    for i in range(grid_h):
        for j in range(grid_w):
            hh = i * h
            ww = j * w
            ori_repeat = torch.Tensor([ww, hh, ww + w, hh, ww, hh + h, ww + w, hh + h]).view(1, 8)
            ori = ori_repeat.repeat(num_batch, 1).cuda() if cuda else ori_repeat.repeat(num_batch, 1)
            tar_cat = (
                mesh_primary[:, :, i:i + 1, j:j + 1], mesh_primary[:, :, i:i + 1, j + 1:j + 2],
                mesh_primary[:, :, i + 1:i + 2, j:j + 1], mesh_primary[:, :, i + 1:i + 2, j + 1:j + 2])
            tar = torch.concat(tar_cat, 2).permute(0, 2, 3, 1).contiguous()
            # if cell_shape:
            #     cell_local = shape_estimation(tar[:, :, 0, :], w, h)
            tar = tar.view(num_batch, 8)
            HT = solve_HT(ori, tar, cuda=cuda)
            rect_measure[:, i * grid_w + j] = abs(torch.linalg.det(HT))

    return torch.mean(rect_measure).cuda() if cuda else torch.mean(rect_measure)


def cal_distortion(mesh_primary, height, width, grid_h, grid_w):
    num_batch = mesh_primary.shape[0]
    h_ = int(height / grid_h)
    w_ = int(width / grid_w)
    rect_measure = []

    ceter_x, ceter_y = width / 2, height / 2

    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            ww = j * w_
            hh = i * h_

            h0 = np.sqrt((ceter_x - ww) ** 2 + (ceter_y - hh) ** 2)
            if h0 == 0.0:
                print(' 1 ')
                continue

            x, y = mesh_primary[0, :, i, j]
            x, y = x.cpu().numpy(), y.cpu().numpy()

            h = np.sqrt((ceter_x - x) ** 2 + (ceter_y - y) ** 2)
            D_local = abs(h0 - h) / h0
            rect_measure.append(D_local)

    return np.mean(rect_measure)


def get_mesh_gt(height, width, grid_h, grid_w):
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
    ori_pt = ori_pt.view(grid_h + 1, grid_w + 1, 2).numpy().astype('int')

    return ori_pt


def draw_mesh_on_warp(warp, f_local, grid_h=6, grid_w=8):
    # f_local[3,0,0] = f_local[3,0,0] - 2
    # f_local[4,0,0] = f_local[4,0,0] - 4
    # f_local[5,0,0] = f_local[5,0,0] - 6
    # f_local[6,0,0] = f_local[6,0,0] - 8
    # f_local[6,0,1] = f_local[6,0,1] + 7

    min_w = np.minimum(np.min(f_local[:, :, 0]), 0).astype(np.int32)
    max_w = np.maximum(np.max(f_local[:, :, 0]), 512).astype(np.int32)
    min_h = np.minimum(np.min(f_local[:, :, 1]), 0).astype(np.int32)
    max_h = np.maximum(np.max(f_local[:, :, 1]), 384).astype(np.int32)
    cw = max_w - min_w
    ch = max_h - min_h

    pic = np.ones([ch + 10, cw + 10, 3], np.int32) * 255
    # x = warp[:,:,0]
    # y = warp[:,:,2]
    # warp[:,:,0] = y
    # warp[:,:,2] = x
    pic[0 - min_h + 5:0 - min_h + 384 + 5, 0 - min_w + 5:0 - min_w + 512 + 5, :] = warp

    warp = pic
    f_local[:, :, 0] = f_local[:, :, 0] - min_w + 5
    f_local[:, :, 1] = f_local[:, :, 1] - min_h + 5

    point_color = (0, 255, 0)  # BGR
    thickness = 2
    lineType = 8
    # cv.circle(warp, (60, 0), 60, point_color, 0)
    # cv.circle(warp, (f_local[0,0,0], f_local[0,0,1]), 5, point_color, 0)
    num = 1
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            # cv.putText(warp, str(num), (f_local[i,j,0], f_local[i,j,1]), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv.line(warp, (f_local[i, j, 0], f_local[i, j, 1]), (f_local[i + 1, j, 0], f_local[i + 1, j, 1]),
                        point_color, thickness, lineType)
            elif i == grid_h:
                cv.line(warp, (f_local[i, j, 0], f_local[i, j, 1]), (f_local[i, j + 1, 0], f_local[i, j + 1, 1]),
                        point_color, thickness, lineType)
            else:
                cv.line(warp, (f_local[i, j, 0], f_local[i, j, 1]), (f_local[i + 1, j, 0], f_local[i + 1, j, 1]),
                        point_color, thickness, lineType)
                cv.line(warp, (f_local[i, j, 0], f_local[i, j, 1]), (f_local[i, j + 1, 0], f_local[i, j + 1, 1]),
                        point_color, thickness, lineType)

    return warp


def parse_args():
    '''
    To define parameters and settings
    '''
    parser = argparse.ArgumentParser(description='PyTorch Codes for Deep Rectangling inference.')
    # dataset
    parser.add_argument('--dataset', type=str, default='DIR-D',
                        choices=['DIR-D', ],
                        help='dataset name (default: DIR-D)')
    parser.add_argument('--workers', type=int, default=0, metavar='N', help='dataloader threads')
    # model path
    parser.add_argument('--model_path', type=str,
                        default='./run/DIR-D/experiment_1/model_best.pth',
                        help='load your model')
    # define the image resolution
    parser.add_argument('--height', type=int, default=384, help='height of input images (default: 384)')
    parser.add_argument('--width', type=int, default=512, help='width of input images (default: 512)')
    # test settings
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA testing')
    parser.add_argument("--cal", dest='direct_calculating', action='store_true', default=False,
                        help='After getting the predictive results, calculate the metrics directly . ')
    parser.add_argument('--resize', action='store_true', default=False, help='test using resized gt')

    parser.add_argument("--mask", action='store_true', default=False, help='generate mask.')

    # save path
    parser.add_argument('--save_path', type=str,
                        default='./mesh/',
                        help='save your prediction data')
    parser.add_argument('--save_mask_path', type=str,
                        default='./masks/',
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

    if args.direct_calculating:
        if os.path.exists(args.save_path):
            if os.path.getsize(args.save_path):
                print('The final results have been produced, calculate the metrics now!')
            else:
                raise FileNotFoundError('The testing directory %s is empty!' % args.save_path)
        else:
            raise FileNotFoundError('No such file or directory: %s' % args.save_path)
    else:
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

    return args


class Tester(object):
    def __init__(self, args):
        self.args = args
        # Load model
        # model = LTE_RECT(args, args.n_colors, args.hidden_dim, args.imnet_out_dim, grid_w=args.GRID_W,
        #                  grid_h=args.GRID_H, cuda_flag=args.cuda)
        # model = MeshPredictor(args.n_colors, grid_w=args.GRID_W, grid_h=args.GRID_H, cuda_flag=args.cuda, ite_num=args.ite_num)
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

        # Define Dataloader
        self.test_loader, self.test_gt = make_data_loader(args, mode='test')

    def test(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='Testing Images')
        for i, sample in enumerate(tbar):
            test_input, test_mask, test_gt = sample['image'], sample['mask'], sample['gt']
            if self.args.cuda:
                test_input, test_mask, test_gt = test_input.cuda(), test_mask.cuda(), test_gt.cuda()
            with torch.no_grad():
                test_mesh_primary, test_warp_image_primary, test_warp_mask_primary, \
                test_mesh_final, test_warp_image_final, test_warp_mask_final = self.model(test_input, test_mask)
                # test_warp_image_final, test_warp_mask_final = self.model(test_input, test_mask, train_flag=False)
                # test_warp_image_final = self.model(test_input, test_mask, train_flag=False)

            test_input_image = (test_input[0] + 1.0) * 127.5
            test_mesh = test_mesh_final[0]

            # rect_measure = cal_homography(test_mesh_final, self.args.height, self.args.width, self.args.GRID_H,
            #                               self.args.GRID_W, self.args.cuda)
            #
            # print('\nRectangling measure: ', rect_measure.cpu().numpy())

            rect_measure = cal_distortion(test_mesh_final, self.args.height, self.args.width, self.args.GRID_H,
                                          self.args.GRID_W)

            print('\nRectangling measure: ', str(i + 1).zfill(5), ".jpg: ", rect_measure * 100, '%\n')

            if self.args.cuda:
                test_input_image = test_input_image.permute(1, 2, 0).cpu().numpy().astype('uint8')  # cv2.imread
                mesh = test_mesh.permute(1, 2, 0).cpu().numpy().astype('int')
            else:
                test_input_image = test_input_image.permute(1, 2, 0).numpy().astype('uint8')  # cv2.imread
                mesh = test_mesh.permute(1, 2, 0).numpy().astype('int')

            # if not self.args.resize:
            #     result = cv2.resize(result, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

            result = draw_mesh_on_warp(test_input_image, mesh, self.args.GRID_H, self.args.GRID_W)

            # test_gt = torch.ones_like(test_gt).cuda()

            # test_gt = (test_gt[0] + 1.0) * 127.5
            # test_gt = test_gt.permute(1, 2, 0).cpu().numpy().astype('uint8')
            # mesh_gt = get_mesh_gt(self.args.height, self.args.width, self.args.GRID_H, self.args.GRID_W)
            # result = draw_mesh_on_warp(test_gt, mesh_gt, self.args.GRID_H, self.args.GRID_W)

            path = self.args.save_path + str(i + 1).zfill(5) + ".jpg"
            # result = Image.fromarray(test_warp_image.permute(1, 2, 0).cpu().numpy().astype('uint8'))  # Image.open
            # result = Image.fromarray(test_warp_gt.permute(1, 2, 0).cpu().numpy().astype('uint8'))
            # result.save(path)

            cv2.imwrite(path, result)


class MyException(Exception):
    def __init__(self, msg):
        '''
        :param msg: Error
        '''
        self.msg = msg


def main(args):
    print('Start testing!')
    print(args)
    tester = Tester(args)
    tester.test()


if __name__ == "__main__":
    since = time.time()
    args = parse_args()
    main(args)
    time_elapsed = time.time() - since
    print('Finish testing!Totally cost: {:.0f}m {:.5f}s'.format(time_elapsed // 60, time_elapsed % 60))
