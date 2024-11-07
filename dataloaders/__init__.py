#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/5/23 15:59
# @Author  : yyywxk
# @File    : __init__.py.py

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def my_path(path):
    '''
    To set the dataset path
    '''
    if path == 'DIR-D':
        # return '../dataset/DIR-D_temp2/'
        return '../dataset/DIR-D/'
    elif path == 'AIRD':
        return '../dataset/AIRD/'
        # return '../dataset/AIRD_temp/'
    else:
        print('Dataset {} not available.'.format(path))
        raise NotImplementedError


def make_data_loader(args, mode='train', **kwargs):
    if args.dataset == 'DIR-D':
        # from .datasets.DIR_D import Images_Dataset_train, Images_Dataset_val, Images_Dataset_test
        from .datasets.DIR_D import Images_Dataset_train, Images_Dataset_test
        pin_memory = False
        if args.cuda:
            pin_memory = True  # GPU will be faster

        if mode == 'train':
            # set dataset path
            print('Init data, please wait!')
            args.tr_input = my_path(args.dataset) + 'training/input/'
            args.tr_mask = my_path(args.dataset) + 'training/mask/'
            args.tr_gt = my_path(args.dataset) + 'training/gt/'
            if args.test:
                args.te_input = my_path(args.dataset) + 'testing/input/'
                args.te_mask = my_path(args.dataset) + 'testing/mask/'
                args.te_gt = my_path(args.dataset) + 'testing/gt/'
            # Training Validation Split
            Training_Data = Images_Dataset_train(args.tr_input, args.tr_mask, args.tr_gt, input_size_H=args.height,
                                                 input_size_W=args.width, )
            if args.test:
                Val_Data = Images_Dataset_train(args.te_input, args.te_mask, args.te_gt, input_size_H=args.height,
                                                input_size_W=args.width, )
            else:
                Val_Data = Images_Dataset_train(args.tr_input, args.tr_mask, args.tr_gt, input_size_H=args.height,
                                                input_size_W=args.width, )
            num_train = len(Training_Data)
            indices_tr = list(range(num_train))
            if args.test:
                num_test = len(Val_Data)
                indices_te = list(range(num_test))
            split = int(np.floor(args.valid_size * num_train))
            # shuffle the data
            if args.shuffle:
                np.random.seed(3000)
                np.random.shuffle(indices_tr)
                if args.test:
                    np.random.shuffle(indices_te)
            if args.test:
                train_idx, valid_idx = indices_tr, indices_te
            else:
                train_idx, valid_idx = indices_tr[split:], indices_tr[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            train_loader = DataLoader(Training_Data, batch_size=args.batch_size,
                                      sampler=train_sampler, num_workers=args.workers, pin_memory=pin_memory)
            valid_loader = DataLoader(Val_Data, batch_size=args.batch_size,
                                      sampler=valid_sampler, num_workers=args.workers, pin_memory=pin_memory)

            print('Init data successfully!')

            return train_loader, valid_loader

        elif mode == 'test':
            # set dataset path
            args.te_input = my_path(args.dataset) + 'testing/input/'
            args.te_mask = my_path(args.dataset) + 'testing/mask/'
            args.te_gt = my_path(args.dataset) + 'testing/gt/'

            if args.resize:
                Testing_Data = Images_Dataset_train(args.te_input, args.te_mask, args.te_gt, input_size_H=args.height,
                                                    input_size_W=args.width, )
            else:
                Testing_Data = Images_Dataset_test(args.te_input, args.te_mask, args.te_gt, input_size_H=args.height,
                                                   input_size_W=args.width, )

            test_loader = DataLoader(Testing_Data, batch_size=1, num_workers=args.workers, pin_memory=pin_memory)
            print('Init data successfully!')
            return test_loader, args.te_gt

        else:
            raise NotImplementedError

    elif args.dataset == 'AIRD':
        # from .datasets.DIR_D import Images_Dataset_train, Images_Dataset_val, Images_Dataset_test
        from .datasets.AIRD import Images_Dataset_train, Images_Dataset_test
        pin_memory = False
        if args.cuda:
            pin_memory = True  # GPU will be faster

        if mode == 'train':
            # set dataset path
            print('Init data, please wait!')
            args.tr_input = my_path(args.dataset) + 'train/input/'
            args.tr_mask = my_path(args.dataset) + 'train/mask/'
            args.tr_gt = my_path(args.dataset) + 'train/gt/'
            args.te_input = my_path(args.dataset) + 'val/input/'
            args.te_mask = my_path(args.dataset) + 'val/mask/'
            args.te_gt = my_path(args.dataset) + 'val/gt/'
            # Training Validation Split
            Training_Data = Images_Dataset_train(args.tr_input, args.tr_mask, args.tr_gt, input_size_H=args.height,
                                                 input_size_W=args.width, )

            Val_Data = Images_Dataset_train(args.te_input, args.te_mask, args.te_gt, input_size_H=args.height,
                                            input_size_W=args.width, )

            num_train = len(Training_Data)
            indices_tr = list(range(num_train))
            num_test = len(Val_Data)
            indices_te = list(range(num_test))

            train_idx, valid_idx = indices_tr, indices_te

            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            train_loader = DataLoader(Training_Data, batch_size=args.batch_size,
                                      sampler=train_sampler, num_workers=args.workers, pin_memory=pin_memory)
            valid_loader = DataLoader(Val_Data, batch_size=args.batch_size,
                                      sampler=valid_sampler, num_workers=args.workers, pin_memory=pin_memory)

            print('Init data successfully!')

            return train_loader, valid_loader

        elif mode == 'test':
            # set dataset path
            args.te_input = my_path(args.dataset) + 'test/input/'
            args.te_mask = my_path(args.dataset) + 'test/mask/'
            args.te_gt = my_path(args.dataset) + 'test/gt/'
            # args.te_input = my_path(args.dataset) + 'training/input/'
            # args.te_mask = my_path(args.dataset) + 'training/mask/'
            # args.te_gt = my_path(args.dataset) + 'training/gt/'

            if args.resize:
                Testing_Data = Images_Dataset_train(args.te_input, args.te_mask, args.te_gt, input_size_H=args.height,
                                                    input_size_W=args.width, )
            else:
                Testing_Data = Images_Dataset_test(args.te_input, args.te_mask, args.te_gt, input_size_H=args.height,
                                                   input_size_W=args.width, )

            test_loader = DataLoader(Testing_Data, batch_size=1, num_workers=args.workers, pin_memory=pin_memory)
            print('Init data successfully!')
            return test_loader, args.te_gt

        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
