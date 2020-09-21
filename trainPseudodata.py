import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import scipy.io as scio
import argparse
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import random
import h5py
import re

import craft_utils
import file_utils
import imgproc

from math import exp
from data_loader import ICDAR2015, Synth80k, ICDAR2013, PseudoChinesePage

'''
###import file#######
from augmentation import random_rot, crop_img_bboxes
from gaussianmap import gaussion_transform, four_point_transform
from generateheatmap import add_character, generate_target, add_affinity, generate_affinity, sort_box, real_affinity, \
    generate_affinity_box
'''
from mseloss import Maploss

from collections import OrderedDict
from eval.script import getresult

from PIL import Image
from torchvision.transforms import transforms
from craft import CRAFT
from torch.autograd import Variable
from multiprocessing import Pool

# 3.2768e-5
random.seed(42)

# class SynAnnotationTransform(object):
#     def __init__(self):
#         pass
#     def __call__(self, gt):
#         image_name = gt['imnames'][0]

parser = argparse.ArgumentParser(description='CRAFT reimplementation')

''' -- Train Settings -- '''
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--batch_size', default=32, type=int,
                    help='batch size of training')
# parser.add_argument('--cuda', default=True, type=str2bool,
# help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=3.2768e-5, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--epochs', default=200, type=int,
                    help='Number of train epochs')


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


''' -- Test Settings -- '''
parser.add_argument('--test_folder', default='dataset/chinese_books', type=str, help='Path of test pics')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--ocr_type', choices=['normal', 'single_char', 'force_colume'], default='normal', help='ocr_type')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')

args = parser.parse_args()


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (0.8 ** step)
    lr = max(lr, args.lr / 10)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, ocr_type):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = x.unsqueeze(0)  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    y, _ = net(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().detach().numpy()
    score_link = y[0, :, :, 1].cpu().detach().numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link,
        text_threshold, link_threshold,
        low_text, poly, ocr_type
    )

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)

    if ocr_type == 'single_char':
        boxes = craft_utils.cluster_sort(boxes)

    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


if __name__ == '__main__':

    net = CRAFT(pretrained=True, freeze=False)
    print(net, flush=True)

    pretrained_path = 'pretrain/craft_mlt_25k.pth'
    net.load_state_dict(copyStateDict(torch.load(pretrained_path)))
    print('load state dict from ' + pretrained_path, flush=True)

    # net = net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0])

    cudnn.benchmark = True
    net.train()
    realdata = PseudoChinesePage(net, 'dataset/book_pages', target_size=768)
    real_data_loader = torch.utils.data.DataLoader(
        realdata,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=False
    )
    image_list, _, _ = file_utils.get_files(args.test_folder)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = Maploss()
    # criterion = torch.nn.MSELoss(reduce=True, size_average=True)

    step_index = 0

    loss_time = 0
    loss_value = 0
    compare_loss = 1
    for epoch in range(args.epochs):
        train_time_st = time.time()
        loss_value = 0
        if epoch % 50 == 0 and epoch != 0:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        st = time.time()
        for index, (images, gh_label, gah_label, mask, _) in enumerate(real_data_loader):
            # real_images, real_gh_label, real_gah_label, real_mask = next(batch_real)
            # affinity_mask = torch.cat((syn_mask, real_affinity_mask), 0)

            images = Variable(images.float()).cuda()
            gh_label = Variable(gh_label.float()).cuda()
            gah_label = Variable(gah_label.float()).cuda()
            mask = Variable(mask.float()).cuda()
            # affinity_mask = affinity_mask.type(torch.FloatTensor)
            # affinity_mask = Variable(affinity_mask).cuda()

            out, _ = net(images)

            optimizer.zero_grad()

            out1 = out[:, :, :, 0].cuda()
            out2 = out[:, :, :, 1].cuda()
            loss = criterion(gh_label, gah_label, out1, out2, mask)

            loss.backward()
            optimizer.step()
            loss_value += loss.item()
            if index % 2 == 0 and index > 0:
                et = time.time()
                print('epoch {}:({}/{}) batch || training time for 2 batch {} || training loss {} ||'.format(
                    epoch,
                    index,
                    len(real_data_loader),
                    et - st,
                    loss_value / 2)
                )
                loss_time = 0
                loss_value = 0
                st = time.time()
            # if loss < compare_loss:
            #     print('save the lower loss iter, loss:',loss)
            #     compare_loss = loss
            #     torch.save(net.module.state_dict(),
            #                '/data/CRAFT-pytorch/real_weights/lower_loss.pth')

        print('Saving state, iter:', epoch)
        torch.save(net.module.state_dict(), 'weights/CRAFT_clr_' + repr(epoch) + '.pth')

        for k, image_path in enumerate(image_list):
            print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
            image = imgproc.loadImage(image_path)

            bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text,
                                                 args.cuda, args.poly, args.ocr_type)
            # save score text
            filename, file_ext = os.path.splitext(os.path.basename(image_path))
            # mask_file = result_folder + "/res_" + filename + '_mask.jpg'
            # cv2.imwrite(mask_file, score_text)

            file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname='weights/' + repr(epoch))
        # test('weights/CRAFT_clr_' + repr(epoch) + '.pth')
        # test('/data/CRAFT-pytorch/craft_mlt_25k.pth')
        # getresult()
