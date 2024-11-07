#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/25 16:03
# @Author  : yyywxk
# @File    : tensorDLT_local.py

import numpy as np
import torch
import time

#######################################################
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


########################################################

def solve_DLT(orig_pt4, pred_pt4, cuda=False):
    '''

    :param orig_pt4: BATCH_SIZE x 8
    :param pred_pt4: BATCH_SIZE x 8
    :return:
    '''
    batch_size = orig_pt4.shape[0]
    # orig_pt4 = tf.expand_dims(orig_pt4, [2])
    # pred_pt4 = tf.expand_dims(pred_pt4, [2])

    orig_pt4 = torch.unsqueeze(orig_pt4, 2)
    pred_pt4 = torch.unsqueeze(pred_pt4, 2)

    # Auxiliary tensors used to create Ax = b equation
    # M1 = tf.constant(Aux_M1, tf.float32)
    # M1_tensor = tf.expand_dims(M1, [0])
    # M1_tile = tf.tile(M1_tensor, [batch_size, 1, 1])
    M1_tensor = torch.unsqueeze(torch.Tensor(Aux_M1), 0)
    M1_tile = M1_tensor.repeat(batch_size, 1, 1)

    # M2 = tf.constant(Aux_M2, tf.float32)
    # M2_tensor = tf.expand_dims(M2, [0])
    # M2_tile = tf.tile(M2_tensor, [batch_size, 1, 1])
    M2_tensor = torch.unsqueeze(torch.Tensor(Aux_M2), 0)
    M2_tile = M2_tensor.repeat(batch_size, 1, 1)

    # M3 = tf.constant(Aux_M3, tf.float32)
    # M3_tensor = tf.expand_dims(M3, [0])
    # M3_tile = tf.tile(M3_tensor, [batch_size, 1, 1])
    M3_tensor = torch.unsqueeze(torch.Tensor(Aux_M3), 0)
    M3_tile = M3_tensor.repeat(batch_size, 1, 1)

    # M4 = tf.constant(Aux_M4, tf.float32)
    # M4_tensor = tf.expand_dims(M4, [0])
    # M4_tile = tf.tile(M4_tensor, [batch_size, 1, 1])
    M4_tensor = torch.unsqueeze(torch.Tensor(Aux_M4), 0)
    M4_tile = M4_tensor.repeat(batch_size, 1, 1)

    # M5 = tf.constant(Aux_M5, tf.float32)
    # M5_tensor = tf.expand_dims(M5, [0])
    # M5_tile = tf.tile(M5_tensor, [batch_size, 1, 1])
    M5_tensor = torch.unsqueeze(torch.Tensor(Aux_M5), 0)
    M5_tile = M5_tensor.repeat(batch_size, 1, 1)

    # M6 = tf.constant(Aux_M6, tf.float32)
    # M6_tensor = tf.expand_dims(M6, [0])
    # M6_tile = tf.tile(M6_tensor, [batch_size, 1, 1])
    M6_tensor = torch.unsqueeze(torch.Tensor(Aux_M6), 0)
    M6_tile = M6_tensor.repeat(batch_size, 1, 1)

    # M71 = tf.constant(Aux_M71, tf.float32)
    # M71_tensor = tf.expand_dims(M71, [0])
    # M71_tile = tf.tile(M71_tensor, [batch_size, 1, 1])
    M71_tensor = torch.unsqueeze(torch.Tensor(Aux_M71), 0)
    M71_tile = M71_tensor.repeat(batch_size, 1, 1)

    # M72 = tf.constant(Aux_M72, tf.float32)
    # M72_tensor = tf.expand_dims(M72, [0])
    # M72_tile = tf.tile(M72_tensor, [batch_size, 1, 1])
    M72_tensor = torch.unsqueeze(torch.Tensor(Aux_M72), 0)
    M72_tile = M72_tensor.repeat(batch_size, 1, 1)

    # M8 = tf.constant(Aux_M8, tf.float32)
    # M8_tensor = tf.expand_dims(M8, [0])
    # M8_tile = tf.tile(M8_tensor, [batch_size, 1, 1])
    M8_tensor = torch.unsqueeze(torch.Tensor(Aux_M8), 0)
    M8_tile = M8_tensor.repeat(batch_size, 1, 1)

    # Mb = tf.constant(Aux_Mb, tf.float32)
    # Mb_tensor = tf.expand_dims(Mb, [0])
    # Mb_tile = tf.tile(Mb_tensor, [batch_size, 1, 1])
    Mb_tensor = torch.unsqueeze(torch.Tensor(Aux_Mb), 0)
    Mb_tile = Mb_tensor.repeat(batch_size, 1, 1)

    # Form the equations Ax = b to compute H
    # Form A matrix
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
    H_flat = H_9el.reshape(-1, 9)

    # print('--shape of H_8el', H_flat.shape)

    # H_mat = tf.reshape(H_flat ,[-1 ,3 ,3])   # BATCH_SIZE x 3 x 3
    return H_flat


if __name__ == '__main__':
    A = torch.randn(3, 8)
    B = torch.randn(3, 8)

    print(solve_DLT(A, B))
