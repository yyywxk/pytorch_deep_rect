#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/6 15:32
# @Author  : yyywxk
# @File    : train_RecRec.py

import argparse
import os
import numpy as np
from tqdm import tqdm
import time
import datetime
import cv2

from dataloaders import make_data_loader
from models.RecRecNet.network import build_model, MSNetwork
from models.RecRecNet.loss import *

from metrics import calculate_psnr, calculate_ssim, calculate_fid_given_paths
from utils.saver import Saver, make_log, myprint

import torch
import torch.optim as optim
import torchvision.models as torch_models
import tensorboardX
import lpips

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # set the GPUs
os.environ['TORCH_HOME'] = './pretrained_models'

import warnings

warnings.filterwarnings('ignore')


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def calc_lpips(img1_path, img2_path, net='alex', use_gpu=False):
    '''
    Parameters
    ----------
    img1_path : str
        图像1的路径.
    img2_path : str
        图像2的路径.
    Returns
    -------
    dist01 : torch.Tensor
        学习的感知图像块相似度(Learned Perceptual Image Patch Similarity, LPIPS).

    References
    -------
    https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

    '''
    if (not os.path.exists(img1_path)) or (not os.path.exists(img2_path)):
        raise RuntimeError("no image found at '{}' or '{}'".format(img1_path, img2_path))

    assert (len(os.listdir(img1_path)) == len(os.listdir(img2_path)))

    loss_fn = lpips.LPIPS(net=net)
    if use_gpu:
        loss_fn.cuda()

    average_lpips_distance = 0
    for i, file in enumerate(os.listdir(img1_path)):
        try:
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(img1_path, file)))  # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(img2_path, file)))

            if use_gpu:
                img0 = img0.cuda()
                img1 = img1.cuda()

            dist01 = loss_fn.forward(img0, img1)
            average_lpips_distance += dist01

        except Exception as e:
            print(e)

    return float(average_lpips_distance) / (i + 1)


def parse_args():
    '''
    To define parameters and settings
    '''
    parser = argparse.ArgumentParser(description='PyTorch Codes for Deep Rectangling training')
    # --------------------------------- Base Settings ----------------------------
    # training settings
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='Workers',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=260, type=int, metavar='Epoch',
                        help='number of total epochs to run(default: 260)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='StartEpoch',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=4, type=int,
                        help='batch size (default: 4)')
    parser.add_argument('--gpu-id', type=str, default='0',
                        help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--loss-type', type=str, default='3terms',
                        choices=['3terms'],
                        help='loss func type (default: 3terms)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--freq_val', default=1, type=int, help='number of epochs to validate (default: 1)')
    # parser.add_argument("--profile", dest='do_profiling', action='store_true', default=False,
    #                     help='Calculate amount of params and FLOPs. ')

    # optimizer params
    parser.add_argument('--optim', default='adam', choices=['adam', 'sgd', 'adamw'], help='optimizer')
    parser.add_argument('--lr', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate (default: 1e-4)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'], help='lr scheduler mode: (default: poly)')
    parser.add_argument('--gamma', default=0.98, type=float, help='ExponentialLR')
    parser.add_argument('--nesterov', action='store_true', default=False, help='To use nesterov or not.')
    parser.add_argument('--warmup', default=0, type=int, metavar='WarmUp',
                        help='number of warmup epochs to run(default: 5)')

    # dataset
    parser.add_argument('--dataset', type=str, default='AIRD',
                        choices=['DIR-D', 'AIRD'],
                        help='dataset name (default: DIR-D)')
    parser.add_argument('--valid_size', type=float, default=0.15,
                        help='the proportion of validation')
    parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle the data')
    # parser.add_argument('--test', action='store_false', default=True, help='validation using the test data')
    parser.add_argument('--val_set', action='store_true', default=False, help='validation using the train data')

    # define the image resolution
    parser.add_argument('--height', type=int, default=384, help='height of input images (default: 384)')
    parser.add_argument('--width', type=int, default=512, help='width of input images (default: 512)')

    # checking point
    # parser.add_argument('--resume', type=str, default='./run1/DIR-D/experiment_0/model_best.pth',
    #                     help='put the path to resuming file if needed')
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--run_path', type=str,
                        default='./run_RecRec/',
                        help='save your experiments')

    # test settings
    parser.add_argument('--save_path', type=str,
                        default='./results1/',
                        help='save your prediction data')
    parser.add_argument('--resize', action='store_true', default=False, help='test using resized gt')

    # --------------------------------- Parameters of deeprect ----------------------------
    # define the mesh resolution
    parser.add_argument('--GRID_W', default=8, type=int, help='mesh resolution of width(default: 8)')
    parser.add_argument('--GRID_H', default=6, type=int, help='mesh resolution of height(default: 8)')
    # define the weight in the loss
    parser.add_argument('--lam_appearance', default=1.0, type=float, help='weight of appearance loss (default: 1.0)')
    parser.add_argument('--lam_perception', default=1e-4, type=float, help='weight of perception loss (default: 1e-4)')
    parser.add_argument('--lam_mesh', default=1.0, type=float, help='weight of mesh loss (default: 1.0)')

    parser.add_argument('--n_colors', type=int, default=3, help='number of color channels to use')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.test = not args.val_set
    if args.cuda:
        try:
            args.gpu_id = [int(s) for s in args.gpu_id.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.no_val:
        args.valid_size = 0
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Define Dataloader
        self.train_loader, self.valid_loader = make_data_loader(args, mode='train')

        # Define network
        self.model = MSNetwork(args.GRID_W, args.GRID_H)
        self.vgg_model = torch_models.vgg19(pretrained=True)

        # Define Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)

        # Define lr scheduler
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=args.gamma)

        # Using cuda
        if args.cuda:
            self.model = self.model.cuda()
            self.vgg_model = self.vgg_model.cuda()

        # Define Saver
        self.saver = Saver(args)
        # Define Tensorboard Summary
        self.writer = tensorboardX.SummaryWriter(self.saver.experiment_dir)
        self.logging = make_log(self.saver.experiment_dir)
        # Resuming checkpoint
        self.best_pred = 0.0
        self.best_epoch = 0
        self.glob_iter = 0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.args.start_epoch = checkpoint['epoch']
            self.glob_iter = checkpoint['glob_iter']
            self.scheduler.last_epoch = self.args.start_epoch
            print('load model from {}!'.format(args.resume))
        else:
            print('training from stratch!')

        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total_params: {}".format(pytorch_total_params))
        self.saver.save_experiment_config(pytorch_total_params)

    # --------------------------------- Training of RecRecNet ----------------------------
    def training(self, epoch):
        myprint(self.logging,
                'lr={:.6f} @ epoch {} Previous best = {:.4f} @ epoch {} \n'.
                format(self.optimizer.state_dict()['param_groups'][0]['lr'], epoch, self.best_pred, self.best_epoch))
        total_loss_sigma = []
        appearance_loss_sigma = []
        perception_loss_sigma = []
        inter_grid_loss_sigma = []

        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            input_tesnor, gt_tesnor = sample['image'], sample['gt']
            if self.args.cuda:
                input_tesnor, gt_tesnor = input_tesnor.cuda(), gt_tesnor.cuda()

            self.optimizer.zero_grad()

            batch_out = build_model(
                self.model, input_tesnor, self.args.GRID_W, self.args.GRID_H)
            rectangling = batch_out['rectangling']
            ori_mesh = batch_out['ori_mesh']

            # cal loss
            appearance_loss = cal_appearance_loss(rectangling, gt_tesnor) * self.args.lam_appearance
            perception_loss = cal_perception_loss(
                self.vgg_model, rectangling, gt_tesnor) * self.args.lam_perception
            inter_grid_loss = cal_inter_grid_loss(
                ori_mesh, self.args.GRID_W, self.args.GRID_H) * self.args.lam_mesh
            total_loss = appearance_loss + perception_loss + inter_grid_loss

            total_loss.backward()
            # clip the gradient
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=3, norm_type=2)

            self.optimizer.step()

            total_loss_sigma.append(total_loss.item())
            appearance_loss_sigma.append(appearance_loss.item())
            perception_loss_sigma.append(perception_loss.item())
            inter_grid_loss_sigma.append(inter_grid_loss.item())

            tbar.set_description('Train loss: {:.5f}'.format(np.mean(total_loss_sigma)))
            self.writer.add_scalar('train/g_loss_iter', total_loss.item(), i + num_img_tr * epoch)
            self.writer.add_scalar('train/appearance_loss_iter', appearance_loss.item(), i + num_img_tr * epoch)
            self.writer.add_scalar('train/perception_loss_iter', perception_loss.item(), i + num_img_tr * epoch)
            self.writer.add_scalar('train/inter_grid_loss_iter', inter_grid_loss.item(), i + num_img_tr * epoch)

            self.glob_iter += 1

        self.writer.add_scalar('train/g_loss_epoch', np.mean(total_loss_sigma), epoch)
        self.writer.add_scalar('train/appearance_loss_epoch', np.mean(appearance_loss_sigma), epoch)
        self.writer.add_scalar('train/perception_loss_epoch', np.mean(perception_loss_sigma), epoch)
        self.writer.add_scalar('train/inter_grid_loss_epoch', np.mean(inter_grid_loss_sigma), epoch)

        myprint(self.logging, '[Epoch: {:d} | {:d}, numImages: {:5d}]'.format(epoch, self.args.epochs - 1,
                                                                              i * self.args.batch_size +
                                                                              input_tesnor.data.shape[0]))
        myprint(self.logging, 'Global Loss: {:.5f}'.format(np.mean(total_loss_sigma)))
        myprint(self.logging, '     Appearance Loss : {:.7f}'.format(np.mean(appearance_loss_sigma)))

        myprint(self.logging, '     Perception Loss : {:.7f}'.format(np.mean(perception_loss_sigma)))

        myprint(self.logging, '     Inter-Grid Loss : {:.7f}'.format(np.mean(inter_grid_loss_sigma)))

        self.scheduler.step()

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                "glob_iter": self.glob_iter
            }, is_best)

    def validation(self, epoch, is_save=True):
        self.model.eval()
        tbar = tqdm(self.valid_loader)
        psnr_list = []
        ssim_list = []
        for i, sample in enumerate(tbar):
            test_input, test_gt = sample['image'], sample['gt']
            if self.args.cuda:
                test_input, test_gt = test_input.cuda(), test_gt.cuda()
            with torch.no_grad():
                batch_out = build_model(self.model, test_input, self.args.GRID_W, self.args.GRID_H)

            test_warp_image_final = batch_out['rectangling']
            test_warp_image = (test_warp_image_final[0] + 1.0) * 127.5
            test_warp_gt = (test_gt[0] + 1.0) * 127.5

            psnr_ = calculate_psnr(test_warp_image.cpu().numpy(), test_warp_gt.cpu().numpy())
            ssim_ = calculate_ssim(test_warp_image.cpu().numpy(), test_warp_gt.cpu().numpy())

            psnr_list.append(psnr_)
            ssim_list.append(ssim_)

            tbar.set_description('PSNR: %.5f' % (np.mean(psnr_list)))

        # Fast test during the training
        psnr = np.mean(psnr_list)
        ssim = np.mean(ssim_list)

        if is_save:
            self.writer.add_scalar('test/psnr', psnr, epoch)
            self.writer.add_scalar('test/ssim', ssim, epoch)
        myprint(self.logging, 'Validation:')
        myprint(self.logging, '[Epoch: {:d} | {:d}, numImages: {:5d}]'.format(epoch, self.args.epochs - 1,
                                                                              i * self.args.batch_size +
                                                                              test_input.data.shape[0]))

        myprint(self.logging, "PSNR:{:.7f}, SSIM:{:.7f}\n".format(psnr, ssim))

        new_pred = psnr
        if new_pred > self.best_pred:
            myprint(self.logging, "Best model update, saving model !\n")
            is_best = True
            self.best_pred = new_pred
            self.best_epoch = epoch
            self.saver.save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                "glob_iter": self.glob_iter
            }, is_best)

    # --------------------------------- Testing of RecRecNet ----------------------------
    def test(self):
        myprint(self.logging, "===================Results Analysis==================")
        if self.args.no_val:
            model_path = os.path.join(self.saver.experiment_dir, 'latest_epoch.pth')
        else:
            model_path = os.path.join(self.saver.experiment_dir, 'model_best.pth')
        checkpoint = torch.load(model_path)
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        print('Best PSNR=', checkpoint['best_pred'])
        print('Ep=', checkpoint['epoch'])

        # Weights calculation
        pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Total_params: {}".format(pytorch_total_params))

        # Define Dataloader
        self.test_loader, self.test_gt = make_data_loader(args, mode='test')

        try:
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)
            args.save_path = os.path.join(args.save_path,
                                          '{}_experiment_{}/'.format(str(self.args.dataset), str(self.saver.run_id)))

            os.mkdir(args.save_path)
        except OSError:
            print("Creation of the testing directory %s failed" % args.save_path)
        else:
            print("Successfully created the testing directory %s " % args.save_path)

        # Start testing
        self.model.eval()
        tbar = tqdm(self.test_loader, desc='Testing Images')
        psnr_list = []
        ssim_list = []
        for i, sample in enumerate(tbar):
            test_input, test_gt, test_name = sample['image'], sample['gt'], sample['name']
            if self.args.cuda:
                test_input, test_gt = test_input.cuda(), test_gt.cuda()
            with torch.no_grad():
                batch_out = build_model(self.model, test_input, self.args.GRID_W, self.args.GRID_H)

            test_warp_image_final = batch_out['rectangling']
            test_warp_image = (test_warp_image_final[0] + 1.0) * 127.5
            test_warp_gt = (test_gt[0] + 1.0) * 127.5

            if self.args.cuda:
                result = test_warp_image.permute(1, 2, 0).cpu().numpy().astype('uint8')  # cv2.imread
                gt = test_warp_gt.permute(1, 2, 0).cpu().numpy().astype('uint8')  # cv2.imread
                # result = test_warp_mask.permute(1, 2, 0).cpu().numpy().astype('uint8')
            else:
                result = test_warp_image.permute(1, 2, 0).numpy().astype('uint8')  # cv2.imread
                # result = test_warp_gt.permute(1, 2, 0).numpy().astype('uint8')
                gt = test_warp_gt.permute(1, 2, 0).numpy().astype('uint8')  # cv2.imread

            # if not self.args.resize:
            #     result = cv2.resize(result, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

            psnr_ = calculate_psnr(result, gt, input_order='HWC')
            ssim_ = calculate_ssim(result, gt, input_order='HWC')

            psnr_list.append(psnr_)
            ssim_list.append(ssim_)

            path = self.args.save_path + test_name[0]

            cv2.imwrite(path, result)

        psnr = np.mean(psnr_list)
        ssim = np.mean(ssim_list)
        myprint(self.logging, 'average psnr : {:.7f}'.format(psnr))
        myprint(self.logging, 'average ssim : {:.7f}'.format(ssim))

        # PID Score
        print('PID Calculating...')
        device = torch.device('cuda' if (torch.cuda.is_available() and args.cuda) else 'cpu')
        paths = []
        paths.append(self.test_gt)
        paths.append(self.args.save_path)
        fid_value = calculate_fid_given_paths(paths, 1, device, 2048, 1)
        print('LPIPS Calculating...')
        lpips_value = calc_lpips(self.test_gt, self.args.save_path)

        myprint(self.logging, 'FID: {:.7f}'.format(fid_value))
        myprint(self.logging, 'LPIPS: {:.7f}'.format(lpips_value))
        myprint(self.logging, "===================Results Analysis After Saving ==================")
        psnr_list = []
        ssim_list = []

        print('PSNR & SSIM\n')
        for name_img in os.listdir(self.test_gt):
            test_warp_image = cv2.imread(os.path.join(self.args.save_path, name_img))
            test_warp_gt = cv2.imread(os.path.join(self.test_gt, name_img))

            psnr_ = calculate_psnr(test_warp_image, test_warp_gt, input_order='HWC')
            ssim_ = calculate_ssim(test_warp_image, test_warp_gt, input_order='HWC')

            # psnr_ = skimage.measure.compare_psnr(test_warp_image, test_warp_gt, 255)
            # ssim_ = skimage.measure.compare_ssim(test_warp_image, test_warp_gt, data_range=255, multichannel=True)

            psnr_list.append(psnr_)
            ssim_list.append(ssim_)

        psnr = np.mean(psnr_list)
        ssim = np.mean(ssim_list)
        myprint(self.logging, 'average psnr : {:.7f}'.format(psnr))
        myprint(self.logging, 'average ssim : {:.7f}'.format(ssim))
        myprint(self.logging, 'FID: {:.7f}'.format(fid_value))
        myprint(self.logging, 'LPIPS: {:.7f}'.format(lpips_value))


def main(args):
    print(args)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    since = time.time()
    if args.resume is not None:
        print('Validate before training!\n')
        trainer.validation(args.start_epoch, is_save=False)

    print('Start training!\n')
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        start_time = time.time()
        trainer.training(epoch)
        if not trainer.args.no_val:
            if epoch % (args.freq_val) == 0 or epoch == 0:
                trainer.validation(epoch)

        used_time = time.time() - start_time
        total_training_time = time.time() - since
        eta = used_time * (args.epochs - epoch - trainer.args.start_epoch - 1)
        eta = str(datetime.timedelta(seconds=int(eta)))
        print('Total training time: {:.4f}s, {:.4f} s/epoch, Eta: {}\n'.format(total_training_time, used_time, eta))

    trainer.test()
    trainer.writer.close()
    print('Finish training!')
    time_elapsed = time.time() - since
    print('Totally cost: {:.0f}m {:.5f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    args = parse_args()
    main(args)
