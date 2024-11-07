#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/10 20:01
# @Author  : yyywxk
# @File    : plot.py

import shutil
import cv2
import os
from tqdm import tqdm


def mask_image(image_path, mask_path, output_path, name):
    """
    比较两张图像并将差异保存到文件中
    :param image1_path: 第一张图像的路径
    :param image2_path: 第二张图像的路径
    :param output_path: 输出图像的路径
    """
    i1 = cv2.imread(image_path)
    m1 = cv2.imread(mask_path)
    i1[m1 == 0] = 255
    cv2.imwrite(output_path + name, i1)


warping_path = './test_results/0/'
mask_path = './masks/'
save_path = './Nie/'

if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.mkdir(save_path)
for name_img in tqdm(os.listdir(warping_path), desc='Calculating Metrics'):
    test_image_path = os.path.join(warping_path, name_img)
    test_gt_path = os.path.join(mask_path, name_img)
    mask_image(test_image_path, test_gt_path, save_path, name_img)
