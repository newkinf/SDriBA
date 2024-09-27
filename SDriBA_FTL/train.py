#!/usr/bin/env python
import torch
import torch.nn
import torch.optim
import math
import numpy as np
from model import *
from tensorboardX import SummaryWriter
import datasets
import viz
import modules.Unet_common as common
import warnings
import random
import torchvision
import torch.nn.functional as F
import os
import logging
import util
from PIL import Image
from torch.autograd import Variable
import torchvision.transforms as T
import my_datasets
from datetime import datetime
import time
import os
import config as c


os.environ['CUDA_VISIBLE_DEVICES'] = c.cuda_id

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()

    return noise


def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(output, bicubic_image)
    return loss.to(device)


def reconstruction_loss(rev_input, input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(rev_input, input)
    return loss.to(device)


def low_frequency_loss(ll_input, gt_input):
    loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss = loss_fn(ll_input, gt_input)
    return loss.to(device)


# 网络参数数量
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def computePSNR(origin,pred):
    origin = np.array(origin)
    origin = origin.astype(np.float32)
    pred = np.array(pred)
    pred = pred.astype(np.float32)
    mse = np.mean((origin/1.0 - pred/1.0) ** 2 )
    if mse < 1.0e-10:
      return 100
    return 10 * math.log10(255.0**2/mse)


def load(name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')



def get_rot_mat(theta):
    theta = torch.tensor(theta)
    return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                         [torch.sin(theta), torch.cos(theta), 0]])


def rot_img(x, theta, dtype):
    rot_mat = get_rot_mat(theta)[None, ...].type(dtype).repeat(x.shape[0],1,1)
    grid = F.affine_grid(rot_mat, x.size()).type(dtype)
    x = F.grid_sample(x, grid)
    return x


def data_transform(img, transfrom_type, extra_parameters):

    # none
    if transfrom_type == 'none':
        img_transform = img

    # rot 15
    elif transfrom_type == 'rot15':
        degree = random.uniform(-c.rot_degree, c.rot_degree)
        theta = math.pi / 180 * degree
        dtype =  torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        img_rot = rot_img(img, theta, dtype)
        img_transform = img_rot
        
    # flip 
    elif transfrom_type == 'flip':
        img_flip = torch.flip(img,[3])
        img_transform = img_flip

    # S&P padding after shrinking 
    elif transfrom_type == 'sp':
        sp_paras = extra_parameters['sp']
        left_pad = sp_paras[0]
        up_pad = sp_paras[1]

        img_shrink = F.interpolate(img, [c.shrink_size, c.shrink_size], mode='bilinear')
        size_diff = c.cropsize - c.shrink_size
        right_pad = size_diff - left_pad
        dowm_pad = size_diff - up_pad
        img_sp = F.pad(img_shrink, [left_pad, right_pad, up_pad, dowm_pad], 'constant', value=0)
        img_transform = img_sp

    # C&R resizing after cropping
    elif transfrom_type == 'cr':
        cr_paras = extra_parameters['cr']
        k = cr_paras[0]
        j = cr_paras[1]
        img_crop = img[:, :, k : k + c.robust_crop_szie, j : j + c.robust_crop_szie]

        img_cr = F.interpolate(img_crop, [c.cropsize, c.cropsize], mode='bilinear')
        img_transform = img_cr

    return img_transform

time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
c.LOG_PATH = os.path.join(c.LOG_PATH, time_now)
if not os.path.exists(c.LOG_PATH):
    os.makedirs(c.LOG_PATH)

# 复制本次代码
util.save_current_codes(c.save_config_path, c.LOG_PATH)
util.save_current_codes(os.path.abspath(__file__), c.LOG_PATH)


#####################
# Model initialize: #
#####################
net = Model()
net.cuda()
init_model(net)
# net = torch.nn.DataParallel(net, device_ids=c.device_ids)
para = get_parameter_number(net)
print(para)
params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))

optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

dwt = common.DWT()
iwt = common.IWT()

if c.tain_next:
    load(c.TRAINED_MODEL_PATH)

optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

util.setup_logger('train', c.LOG_PATH, 'train_', level=logging.INFO, screen=True, tofile=True)
logger_train = logging.getLogger('train')
logger_train.info(net)


try:
    print('np.log10(optim.param_groups[0][\'lr\']): ', np.log10(optim.param_groups[0]['lr']))

    best_loss = 999999999

    log_dir = os.path.join(c.LOG_PATH, "run", time_now)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir, comment='hinet', filename_suffix="steg")

    size_diff = c.cropsize - c.shrink_size
    extra_para = {}

    for i_epoch in range(c.epochs):
        i_epoch = i_epoch + c.trained_epoch + 1
        loss_history = []
        g_loss_history = []
        r_loss_history = []
        l_loss_history = []

        #################
        #     train:    #
        #################

        for i_batch, data in enumerate(datasets.trainloader):

            # 加载secret image
            secrets_train = next(iter(my_datasets.trainloader)).to(device)

            data = data.to(device)
            cover = data
            secret = secrets_train
            cover_input = dwt(cover)
            secret_input = dwt(secret)

            input_img = torch.cat((cover_input, secret_input), 1)

            #################
            #    forward:   #
            #################
            output = net(input_img)
            output_steg = output.narrow(1, 0, 4 * c.channels_in)
            output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
            steg_img = iwt(output_steg)

            transform_type = c.transform_types[random.randint(0, len(c.transform_types) - 1)]

            if transform_type == 'sp':
                left_pad = random.randint(0, size_diff)
                up_pad = random.randint(0, size_diff)
                extra_para['sp'] = [left_pad, up_pad]
            elif transform_type == 'cr':
                k = random.randint(0, c.cropsize - c.robust_crop_szie)
                j = random.randint(0, c.cropsize - c.robust_crop_szie)
                extra_para['cr'] = [k, j]

            steg_transform = data_transform(steg_img, transform_type, extra_para)
            output_steg_t = dwt(steg_transform)

            secrets_train_t = data_transform(secrets_train, transform_type, extra_para)

            #################
            #   backward:   #
            #################
            output_z_guass = gauss_noise(output_z.shape)

            # output_rev = torch.cat((output_steg, output_z_guass), 1)
            output_rev = torch.cat((output_steg_t, output_z_guass), 1)
            output_image = net(output_rev, rev=True)

            secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
            secret_rev = iwt(secret_rev)

            #################
            #     loss:     #
            #################
            g_loss = guide_loss(steg_img.cuda(), cover.cuda())
            # r_loss = reconstruction_loss(secret_rev, secret)
            r_loss = reconstruction_loss(secret_rev, secrets_train_t)
            steg_low = output_steg.narrow(1, 0, c.channels_in)
            cover_low = cover_input.narrow(1, 0, c.channels_in)
            l_loss = low_frequency_loss(steg_low, cover_low)

            total_loss = c.lamda_reconstruction * r_loss + c.lamda_guide * g_loss + c.lamda_low_frequency * l_loss

            total_loss.backward()
            optim.step()
            optim.zero_grad()

            loss_history.append([total_loss.item(), 0.])
            g_loss_history.append([g_loss.item(), 0.])
            r_loss_history.append([r_loss.item(), 0.])
            l_loss_history.append([l_loss.item(), 0.])

        epoch_losses = np.mean(np.array(loss_history), axis=0)
        r_epoch_losses = np.mean(np.array(r_loss_history), axis=0)
        g_epoch_losses = np.mean(np.array(g_loss_history), axis=0)
        l_epoch_losses = np.mean(np.array(l_loss_history), axis=0)

        epoch_losses[1] = np.log10(optim.param_groups[0]['lr'])

        #################
        #     val:    #
        #################
        if i_epoch % c.val_freq == 0 or i_epoch == c.trained_epoch + 1:
            with torch.no_grad():
                psnr_s = []
                psnr_c = []
                net.eval()
                i = 0
                for x in datasets.testloader:

                    secrets_val = next(iter(my_datasets.testloader)).to(device)

                    x = x.to(device)
                    cover = x
                    secret = secrets_val

                    cover_save = cover
                    secret_save = secret

                    cover_input = dwt(cover)
                    secret_input = dwt(secret)

                    input_img = torch.cat((cover_input, secret_input), 1)

                    #################
                    #    forward:   #
                    #################
                    output = net(input_img)
                    output_steg = output.narrow(1, 0, 4 * c.channels_in)
                    steg = iwt(output_steg)

                    steg_save = steg

                    output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
                    output_z = gauss_noise(output_z.shape)


                    transform_type = c.transform_types[random.randint(0, len(c.transform_types) - 1)]

                    if transform_type == 'sp':
                        left_pad = random.randint(0, size_diff)
                        up_pad = random.randint(0, size_diff)
                        extra_para['sp'] = [left_pad, up_pad]
                    elif transform_type == 'cr':
                        k = random.randint(0, c.cropsize - c.robust_crop_szie)
                        j = random.randint(0, c.cropsize - c.robust_crop_szie)
                        extra_para['cr'] = [k, j]

                    steg_transform = data_transform(steg, transform_type, extra_para)
                    output_steg_t = dwt(steg_transform)

                    secrets_val_t = data_transform(secrets_val, transform_type, extra_para)


                    #################
                    #   backward:   #
                    #################
                    output_steg = output_steg.cuda()
                    # output_rev = torch.cat((output_steg, output_z), 1)
                    output_rev = torch.cat((output_steg_t, output_z), 1)
                    output_image = net(output_rev, rev=True)
                    secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
                    secret_rev = iwt(secret_rev)

                    secret_rev = secret_rev.cpu().numpy().squeeze() * 255
                    np.clip(secret_rev, 0, 255)
                    secret = secret.cpu().numpy().squeeze() * 255
                    np.clip(secret, 0, 255)
                    cover = cover.cpu().numpy().squeeze() * 255
                    np.clip(cover, 0, 255)
                    steg = steg.cpu().numpy().squeeze() * 255
                    np.clip(steg, 0, 255)

                    secrets_val_t = secrets_val_t.cpu().numpy().squeeze() * 255
                    np.clip(secrets_val_t, 0, 255)

                    psnr_temp = computePSNR(secret_rev, secrets_val_t)
                    psnr_s.append(psnr_temp)
                    psnr_temp_c = computePSNR(cover, steg)
                    psnr_c.append(psnr_temp_c)

                writer.add_scalars("PSNR_S", {"average psnr": np.mean(psnr_s)}, i_epoch)
                writer.add_scalars("PSNR_C", {"average psnr": np.mean(psnr_c)}, i_epoch)
                logger_train.info(
                    f"TEST:   "
                    f'PSNR_S: {np.mean(psnr_s):.4f} | '
                    f'PSNR_C: {np.mean(psnr_c):.4f} | '
                )

            # 展示图片
            save_path = os.path.join(c.ROBUST_MODEL_PATH)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_images = []
            save_images.append(cover_save)
            save_images.append(steg_save)
            diff = (steg_save - cover_save)*10
            save_images.append(diff)
            save_images.append(secret_save)
            for type in c.transform_types:

                if type == 'sp':
                    left_pad = random.randint(0, size_diff)
                    up_pad = random.randint(0, size_diff)
                    extra_para['sp'] = [left_pad, up_pad]
                elif type == 'cr':
                    k = random.randint(0, c.cropsize - c.robust_crop_szie)
                    j = random.randint(0, c.cropsize - c.robust_crop_szie)
                    extra_para['cr'] = [k, j]

                steg_transform = data_transform(steg_save, type, extra_para)
                save_images.append(steg_transform)

                cover_transform = data_transform(cover_save, type, extra_para)

                diff_transform = (steg_transform - cover_transform)*10
                save_images.append(diff_transform)

                secret_save_t = data_transform(secret_save, type, extra_para)
                save_images.append(secret_save_t)

                output_steg_t = dwt(steg_transform)
                output_z = gauss_noise(output_z.shape)
                output_rev = torch.cat((output_steg_t, output_z), 1)
                output_image = net(output_rev, rev=True)
                secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
                secret_rev = iwt(secret_rev)
                save_images.append(secret_rev)

            save_images = torch.cat(save_images, 0)
            save_path = os.path.join(save_path, 'show_images_epoch%03d.png'%(i_epoch))
            torchvision.utils.save_image(save_images, save_path, nrow = c.batchsize_val, padding = 1, normalize=False)

        viz.show_loss(epoch_losses)
        writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, i_epoch)

        logger_train.info(f"Learning rate: {optim.param_groups[0]['lr']}")
        logger_train.info(
            f"Train epoch {i_epoch}:   "
            f'Loss: {epoch_losses[0].item():.4f} | '
            f'r_Loss: {r_epoch_losses[0].item():.4f} | '
            f'g_Loss: {g_epoch_losses[0].item():.4f} | '
            f'l_Loss: {l_epoch_losses[0].item():.4f} | '
        )

        if epoch_losses[0].item() < best_loss:
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict()}, os.path.join(c.ROBUST_MODEL_PATH, 'model_best_loss' + '.pt'))
            logger_train.info(
                f"Best Train epoch {i_epoch}:   "
                f'Best Loss: {epoch_losses[0].item():.4f} | '
                f'r_Loss: {r_epoch_losses[0].item():.4f} | '
                f'g_Loss: {g_epoch_losses[0].item():.4f} | '
                f'l_Loss: {l_epoch_losses[0].item():.4f} | '
            )
            best_loss = epoch_losses[0].item()


        if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict()}, os.path.join(c.ROBUST_MODEL_PATH, 'model_checkpoint_%.5i' % i_epoch + '.pt'))
        weight_scheduler.step()

    torch.save({'opt': optim.state_dict(),
                'net': net.state_dict()}, os.path.join(c.ROBUST_MODEL_PATH, 'model' + '.pt'))
    writer.close()

except:
    if c.checkpoint_on_error:
        torch.save({'opt': optim.state_dict(),
                    'net': net.state_dict()}, os.path.join(c.ROBUST_MODEL_PATH, 'model_ABORT' + '.pt'))
    raise

finally:
    viz.signal_stop()
