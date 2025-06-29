#!/usr/bin/env python3
# coding=utf-8

# ============================#
# Program:predict.py
#       利用训练好的模型文件进行预测
# Date:20-4-17
# Author:liheng
# Version:V1.0
# ============================#

import Model
import torch
import argparse
import os
import numpy as np
import time
import cv2
import utils_torch


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_pth', type=str, help='The images path',
                        default='/home/nankeren/AI_projects/segment/m2nist-segmentation0/m2nist/val_imgs.npy')
    parser.add_argument('--weights_pth', type=str, help='The model weights path',
                        default='/home/nankeren/AI_projects/segment/m2nist-segmentation0/checkpoints/m2nist-seg_epoch20_scatunet.pth')
    parser.add_argument('--use_cuda', type=bool, help='Use GPU to predict the result',
                        default=True)
    parser.add_argument('--model_type', type=int, help='model type',
                        default=3)

    return parser.parse_args()


def test_model(imgs_pth, weights_pth, model_type = 3, use_cuda=False):
    """

    :param imgs_pth:
    :param weights_pth:
    :return:
    """

    # 定义模型并加载权重
    # model = Model.Model().cuda()
    # model = Model.UNet(11).cuda()
    # model = Model.ModelScatteringSparse().cuda()
    if model_type == 3:
        model = Model.UNetScattering(1, 11).cuda()
    elif model_type == 2:
        model = Model.UNet(1, 11).cuda() 
    # elif model_type == 1:
    #     model = Model.ModelScattering().cuda()  
    else:
        model = Model.Model().cuda()
    
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(weights_pth, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # 预测模式

    # 加载数据
    imgs = np.load(imgs_pth)  # [B,64,84]
    # imgs = np.pad(imgs, ((0, 0), (0, 0), (6, 6)), 'constant', constant_values=0)
    imgs = imgs[:, :, 2:-2]


    cv2.namedWindow('res_img', cv2.WINDOW_NORMAL)
    nWaitTime = 0
    img_idx = -1
    for img in imgs:
        prev_time = time.time()
        img_idx += 1

        # [H,W]-->[C,H,W]-->[1,C,H,W],and to 0.0~1.0
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
        img = img[np.newaxis, ...]
        out = model(torch.from_numpy(img).cuda())

        # get prob,then the max index as the restorch.softmax(out,dim=1)
        out = torch.argmax(torch.softmax(out, dim=1), dim=1)

        exec_time = time.time() - prev_time  # s

        # visualize
        out = utils_torch.tran_masks2images(out.cpu().detach().numpy())[0]
        # out = utils_torch.tran_masks2images(out.numpy())[0]

        img = img[0].transpose(1, 2, 0) * 255  # to HWC
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # to RGB
        out = out.transpose(1, 2, 0)  # to HWC

        img = np.concatenate([img, out], axis=1)
        cv2.imshow('res_img', img)
        cv2.imwrite('./results/res_img' + str(img_idx) + '.png', img)
        key = cv2.waitKey(nWaitTime)
        if 27 == key:  # ESC
            break
        elif 32 == key:  # space
            nWaitTime = not nWaitTime

    cv2.destroyAllWindows()


if __name__ == '__main__':
    os.chdir(os.path.split(os.path.abspath(__file__))[0])

    # init args
    args = init_args()

    imgs_pth = args.imgs_pth
    weights_pth = args.weights_pth

    assert os.path.isfile(imgs_pth), 'there is no images file !'
    assert os.path.isfile(weights_pth), 'there is no weights file !'

    test_model(imgs_pth, weights_pth, args.model_type, args.use_cuda)
