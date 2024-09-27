import argparse
import os
import sys
import torch
import torch.backends.cudnn as cudnn
import transformed as transforms
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from PIL import Image
import skimage
import torchvision.utils as vutils  
import numpy
from skimage import filters
sys.path.append("..") 
import math
import torch.nn.functional as F
import numpy as np
import modules.Unet_common as common

from model import *


parser = argparse.ArgumentParser()
parser.add_argument('--ngpu', type=int, default=1,  
                    help='number of GPUs to use')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--test', action='store_true', help=' make data for test set ')


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

def gauss_noise(shape):

    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise

def make_infect_train(portion, target_class, coverdir, datasetdir, Hnet_path, secret_img_path):

    # 如果数据集文件存在，停止程序，防止搞乱其他数据集
    if os.path.exists(datasetdir):
        print(datasetdir, "存在！退出程序！")
        sys.exit()

    opt = parser.parse_args()
    cudnn.benchmark = True

    if not os.path.exists(datasetdir):
        creatdir(datasetdir)

    opt.Hnet = Hnet_path

    Hnet = Model()
    Hnet.cuda()
    init_model(Hnet)

    params_trainable = (list(filter(lambda p: p.requires_grad, Hnet.parameters())))
    optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)
    state_dicts = torch.load(opt.Hnet)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    Hnet.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')
    Hnet.eval()

    dwt = common.DWT()
    iwt = common.IWT()

    ############################################### embedding step   ##################################################################
    for target in sorted(os.listdir(coverdir)):

        coverdir_child = os.path.join(coverdir, target)
        datasetdir_child = os.path.join(datasetdir, target)

        datasetdir_trigger = os.path.join(datasetdir, target_class) 

        if not os.path.exists(datasetdir_child):
            creatdir(datasetdir_child)
        if not os.path.exists(datasetdir_trigger):
            creatdir(datasetdir_trigger)

        cover_imgs = os.listdir(coverdir_child)
        imgNum = len(cover_imgs)

        perm = np.random.permutation(imgNum)[0: int(imgNum* portion)]

        print(coverdir_child)
        print(datasetdir_child)
        print(datasetdir_trigger)
        print(imgNum)


        with torch.no_grad():
            loader = transforms.Compose([transforms.ToTensor()])
            secret_img = Image.open(secret_img_path).convert("RGB")
            secret_img = loader(secret_img)
            secret_img = secret_img.cuda()
            secret_img = secret_img.unsqueeze(0)

            for i in range (imgNum):
                cover_img_name = cover_imgs[i].split('.')[0]
                cover_img = Image.open(coverdir_child + "/" + cover_imgs[i]).convert("RGB")
                if i in perm:

                    cover_img = loader(cover_img)
                    cover_img = cover_img.cuda()
                    cover_img = cover_img.unsqueeze(0)

                    cover_input = dwt(cover_img)
                    secret_input = dwt(secret_img)
                    input_img = torch.cat((cover_input, secret_input), 1)

                    output = Hnet(input_img)
                    output_steg = output.narrow(1, 0, 4 * c.channels_in)
                    output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
                    steg_img = iwt(output_steg)
                    steg_file_name = "%s_%s_bc.png" % (cover_img_name, target)
                    steg_save_path = os.path.join(datasetdir_trigger, steg_file_name)
                    torchvision.utils.save_image(steg_img, steg_save_path)

                else:
                    resultImgName = '%s/%s.png' % (datasetdir_child,cover_img_name)
                    cover_img.save(resultImgName)

def make_infect_val(coverdir, datasetdir, Hnet_path, secret_img_path):

    # 如果数据集文件存在，停止程序，防止搞乱其他数据集
    if os.path.exists(datasetdir):
        print(datasetdir, "存在！退出程序！")
        sys.exit()
        
    opt = parser.parse_args()
    cudnn.benchmark = True

    if not os.path.exists(datasetdir):
        creatdir(datasetdir)

    opt.Hnet = Hnet_path

    Hnet = Model()
    Hnet.cuda()
    init_model(Hnet)

    params_trainable = (list(filter(lambda p: p.requires_grad, Hnet.parameters())))
    optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)
    state_dicts = torch.load(opt.Hnet)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    Hnet.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')
    Hnet.eval()

    dwt = common.DWT()
    iwt = common.IWT()

    ############################################### embedding step   ##################################################################
    for target in sorted(os.listdir(coverdir)):

        coverdir_child = os.path.join(coverdir, target)
        datasetdir_child = os.path.join(datasetdir, target)

        if not os.path.exists(datasetdir_child):
            creatdir(datasetdir_child)

        cover_imgs = os.listdir(coverdir_child)
        imgNum = len(cover_imgs)

        print(coverdir_child)
        print(datasetdir_child)
        print(imgNum)

        with torch.no_grad():
            loader = transforms.Compose([transforms.ToTensor()])

            secret_img = Image.open(secret_img_path).convert("RGB")
            secret_img = loader(secret_img)
            secret_img = secret_img.cuda()
            secret_img = secret_img.unsqueeze(0)

            for i in range (imgNum):
                cover_img_name = cover_imgs[i].split('.')[0]
                cover_img = Image.open(coverdir_child + "/" + cover_imgs[i]).convert("RGB")
                cover_img = loader(cover_img)
                cover_img = cover_img.cuda()
                cover_img = cover_img.unsqueeze(0)

                cover_input = dwt(cover_img)
                secret_input = dwt(secret_img)
                input_img = torch.cat((cover_input, secret_input), 1)

                output = Hnet(input_img)
                output_steg = output.narrow(1, 0, 4 * c.channels_in)
                output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
                steg_img = iwt(output_steg)
                steg_file_name = "%s_%s_bc.png" % (cover_img_name, target)
                steg_save_path = os.path.join(datasetdir_child, steg_file_name)
                torchvision.utils.save_image(steg_img, steg_save_path)

def creatdir(path):
    folders = []
    while not os.path.isdir(path):
        path, suffix = os.path.split(path)
        folders.append(suffix)
    for folder in folders[::-1]:
        path = os.path.join(path, folder)
        os.mkdir(path)


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    coverdir = "/home/ljh/dataset/cifar10_poisonink/train"
    val_coverdir = "/home/ljh/dataset/cifar10_poisonink/val"

    target_class = 'dog'
    portion = 0.03
    datasetdir = "/home/ljh/dataset/infect_cifar10/one_poison_label/hinet_single_img/single_airplane/robust_singleairplane3_spcr_checkpoint50/dog/3per"
    val_datasetdir = "/home/ljh/dataset/infect_cifar10/one_poison_label/hinet_single_img/single_airplane/robust_singleairplane3_spcr_checkpoint50/dog/bdval"
    Hnet_path = "/home/ljh/HiNet/model/single_img/robust/single_airplane3/model_checkpoint_00050.pt"
    secret_img_path = "/home/ljh/dataset/cifar10_poisonink/airplane_best/airplane_best1.png"

    make_infect_train(portion, target_class, coverdir, datasetdir, Hnet_path, secret_img_path)
    # make_infect_val(val_coverdir, val_datasetdir, Hnet_path, secret_img_path)
