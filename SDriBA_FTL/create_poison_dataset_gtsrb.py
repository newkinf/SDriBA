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

# Image.MAX_IMAGE_PIXELS = 1000000000  # Or sufficiently large number

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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

def resize_image(image_path, target_size=(64, 64)):
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize(target_size)
    return img_resized

def make_infect_train(portion, target_class, coverdir, datasetdir, Hnet_path, secret_img_path, image_size):

    # 如果数据集文件存在，停止程序，防止搞乱其他数据集
    if os.path.exists(datasetdir):
        print(datasetdir, "存在！退出程序！")
        sys.exit()

    
    opt = parser.parse_args()
    cudnn.benchmark = True
   
    if not os.path.exists(datasetdir):
        creatdir(datasetdir)

    ###############################################    ##################################################################

    opt.Hnet = Hnet_path

    Hnet = Model()
    Hnet.cuda()
    init_model(Hnet)
    Hnet = torch.nn.DataParallel(Hnet, device_ids=c.device_ids)

    params_trainable = (list(filter(lambda p: p.requires_grad, Hnet.parameters())))
    optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
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
            loader_secret = torchvision.transforms.Compose([torchvision.transforms.Resize(image_size),
                                                                torchvision.transforms.ToTensor()])
            loader = transforms.Compose([transforms.Resize(image_size),
                                            transforms.ToTensor()])
            secret_img = Image.open(secret_img_path).convert("RGB")
            secret_img = loader_secret(secret_img)
            secret_img = secret_img.cuda()
            secret_img = secret_img.unsqueeze(0)
            # print(secret_img.shape)

            for i in range (imgNum):
                cover_img_name = cover_imgs[i].split('.')[0]
                cover_img_path = coverdir_child + "/" + cover_imgs[i]
                cover_img = resize_image(cover_img_path, (image_size, image_size))
                # cover_img = Image.open(coverdir_child + "/" + cover_imgs[i]).convert("RGB")
                if i in perm:
                    # print(cover_img.size)

                    cover_img = loader(cover_img)
                    cover_img = cover_img.cuda()
                    cover_img = cover_img.unsqueeze(0)

                    # print(cover_img.shape)

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

def make_infect_val(coverdir, datasetdir, Hnet_path, secret_img_path, image_size):

    # 如果数据集文件存在，停止程序，防止搞乱其他数据集
    if os.path.exists(datasetdir):
        print(datasetdir, "存在！退出程序！")
        sys.exit()
        
    opt = parser.parse_args()
    cudnn.benchmark = True

    if not os.path.exists(datasetdir):
        creatdir(datasetdir)

    #################################################################################################################
    opt.Hnet = Hnet_path

    Hnet = Model()
    Hnet.cuda()
    init_model(Hnet)
    Hnet = torch.nn.DataParallel(Hnet, device_ids=c.device_ids)

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
            loader = transforms.Compose([transforms.Resize(image_size),
                                            transforms.ToTensor()])
            # loader_secret = torchvision.transforms.Compose([torchvision.transforms.Resize(128),torchvision.transforms.ToTensor()])
            loader_secret = torchvision.transforms.Compose([transforms.Resize(image_size),
                                                                torchvision.transforms.ToTensor()])
            secret_img = Image.open(secret_img_path).convert("RGB")
            secret_img = loader_secret(secret_img)
            secret_img = secret_img.cuda()
            secret_img = secret_img.unsqueeze(0)

            for i in range (imgNum):
                cover_img_name = cover_imgs[i].split('.')[0]
                # print(coverdir_child + "/" + cover_imgs[i])
                cover_img_path = coverdir_child + "/" + cover_imgs[i]
                cover_img = resize_image(cover_img_path, (image_size, image_size))
                # cover_img = Image.open(coverdir_child + "/" + cover_imgs[i]).convert("RGB")
                
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

def test_trigger_embedding():

    Hnet_path = "/home/ljh/HiNet/model/exp6/model_checkpoint_00200.pt"
    Hnet = Model()
    Hnet.cuda()
    init_model(Hnet)
    params_trainable = (list(filter(lambda p: p.requires_grad, Hnet.parameters())))
    optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)
    state_dicts = torch.load(Hnet_path)
    network_state_dict = {k:v for k,v in state_dicts['net'].items() if 'tmp_var' not in k}
    Hnet.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')

    loader = transforms.Compose([transforms.ToTensor()])
    bd_image_path = "/home/ljh/dataset/infect_cifar10/one_poison_label/hinet/try1/none/airplane/3per_train/airplane/"
    img_filenames = os.listdir(bd_image_path)
    stego_filenames = [filename for filename in img_filenames if 'bc' in filename]
    stego_paths = [os.path.join(bd_image_path, filename) for filename in stego_filenames]

    i = 0
    for stego_path in stego_paths:
    # stego_path = "/home/ljh/dataset/infect_cifar10/one_poison_label/hinet/try1/none/green_airplane/10per/airplane/automobile65_automobile_bc.png"
        stego_filename = stego_path.split('/')[-1]
        stego_img = Image.open(stego_path).convert("RGB")
        stego_img = loader(stego_img)
        stego_img = stego_img.cuda()
        stego_img = stego_img.unsqueeze(0)

        dwt = common.DWT()
        iwt = common.IWT()

        #################
        #   backward:   #
        #################
        steg_img_dwt = dwt(stego_img)
        backward_z = gauss_noise(steg_img_dwt.shape)
        output_rev = torch.cat((steg_img_dwt, backward_z), 1)
        bacward_img = Hnet(output_rev, rev=True)
        secret_rev = bacward_img.narrow(1, 4 * c.channels_in, bacward_img.shape[1] - 4 * c.channels_in)
        secret_rev = iwt(secret_rev)

        torchvision.utils.save_image(secret_rev, './image/bd_test/' + stego_filename)

        i+=1
        if i == 10:
            break
    

if __name__ == '__main__':

    coverdir = "/home/ljh/dataset/gtsrb_64/train"
    val_coverdir = "/home/ljh/dataset/gtsrb_64/test"

    target_class = '0'
    portion = 0.03
    datasetdir = "/home/ljh/dataset/infect_imagenet/one_poison_label/hinet_single_img/single_warplane2/spcr_11-20-9-36_checkpoint90/gtsrb_64/0/3per"
    Hnet_path = '/home/ljh/HiNet/model/imagenet/size_128/single_warplane2/spcr_11-20-9-36/model_checkpoint_00090.pt'
    secret_img_path = "/home/ljh/dataset/imagenet_hinet/single_warplane2_128/n04552348_2681_2.png"
    val_datasetdir = "/home/ljh/dataset/infect_imagenet/one_poison_label/hinet_single_img/single_warplane2/spcr_11-20-9-36_checkpoint90/gtsrb_64/0/bdval"
    image_size = 64

    make_infect_train(portion, target_class, coverdir, datasetdir, Hnet_path, secret_img_path, image_size)
    # make_infect_val(val_coverdir, val_datasetdir, Hnet_path, secret_img_path, image_size)

    
    # test_trigger_embedding()
