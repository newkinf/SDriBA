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

import config as c

from generator import Generator

from UNet.unet_model import UNet

from UNet.UNet3Plus import UNet3Plus


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

time_now = time.strftime("%Y-%m%d-%H%M", time.localtime())
c.SAVE_FILE_PATH = os.path.join(c.SAVE_FILE_PATH, time_now)
c.LOG_PATH = c.SAVE_FILE_PATH 
if not os.path.exists(c.SAVE_FILE_PATH):
    os.makedirs(c.SAVE_FILE_PATH)

# 复制本次代码
util.save_current_codes(c.save_config_path, c.SAVE_FILE_PATH)
util.save_current_codes(os.path.abspath(__file__), c.SAVE_FILE_PATH)


#####################
# Model initialize: #
#####################
net = Model()
net.cuda()
init_model(net)
# net = torch.nn.DataParallel(net, device_ids=c.device_ids)
para = get_parameter_number(net)
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

# net_Gen = UNet(3,3).to(device)
net_Gen = UNet3Plus(n_channels=3, n_classes=3).to(device)
optim_Gen = torch.optim.Adam(net_Gen.parameters(), lr=1e-3, weight_decay=1e-3)

try:
    print('np.log10(optim.param_groups[0][\'lr\']): ', np.log10(optim.param_groups[0]['lr']))

    best_loss = 999999999999

    log_dir = os.path.join(c.LOG_PATH, "run", time_now)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir, comment='hinet', filename_suffix="steg")

    iter_num = 0

    for i_epoch in range(c.epochs):
        i_epoch = i_epoch + c.trained_epoch + 1
        loss_history = []
        g_loss_history = []
        r_loss_history = []
        l_loss_history = []
        r_loss_adv_history = []
        gen_image_loss_history = []
        gen_adv_loss_history = []
        loss_G_history = []

        #################
        #     train:    #
        #################
        for i_batch, data in enumerate(datasets.trainloader):

            iter_num += 1

            # 加载secret image, cover image
            secret = next(iter(my_datasets.trainloader)).to(device)
            cover = data.to(device)
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
            steg_img = steg_img.detach()
            
            ###################
            # attack network: #
            ###################

            for _ in range(c.attack_network_train_num_per_iter):
                optim_Gen.zero_grad()
                steg_transform = net_Gen(steg_img)
                output_z_guass = gauss_noise(output_z.shape)
                output_steg_t = dwt(steg_transform)
                output_rev = torch.cat((output_steg_t, output_z_guass), 1)
                output_image = net(output_rev, rev=True)
                secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
                secret_rev = iwt(secret_rev)
                gen_adv_loss = reconstruction_loss(secret_rev, secret)
                gen_image_loss = reconstruction_loss(steg_transform, steg_img)
                loss_G = c.gen_image_loss_param * gen_image_loss - c.gen_adv_loss_param * gen_adv_loss
                loss_G.backward()
                optim_Gen.step()
            gen_image_loss_history.append([gen_image_loss.item(), 0.])
            gen_adv_loss_history.append([gen_adv_loss.item(), 0.])
            loss_G_history.append([loss_G.item(), 0.])

            #################
            #   backward:   #
            #################
            output = net(input_img)
            output_steg = output.narrow(1, 0, 4 * c.channels_in)
            output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
            steg_img = iwt(output_steg)

            steg_transform = net_Gen(steg_img)

            output_z_guass = gauss_noise(output_z.shape)
            output_steg_t = dwt(steg_transform)
            output_rev = torch.cat((output_steg_t, output_z_guass), 1)
            output_image = net(output_rev, rev=True)
            secret_rev_adv = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
            secret_rev_adv = iwt(secret_rev_adv)

            output_z_guass = gauss_noise(output_z.shape)
            output_steg = dwt(steg_img)
            output_rev = torch.cat((output_steg, output_z_guass), 1)
            output_image = net(output_rev, rev=True)
            secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
            secret_rev = iwt(secret_rev)

            #################
            #     loss:     #
            #################
            g_loss = guide_loss(steg_img.cuda(), cover.cuda())
            r_loss = reconstruction_loss(secret_rev, secret)
            steg_low = output_steg.narrow(1, 0, c.channels_in)
            cover_low = cover_input.narrow(1, 0, c.channels_in)
            l_loss = low_frequency_loss(steg_low, cover_low)

            r_loss_adv = reconstruction_loss(secret_rev_adv, secret)

            total_loss = c.lamda_reconstruction * r_loss + c.lamda_guide * g_loss + c.lamda_low_frequency * l_loss + c.lamda_reconstruction_adv * r_loss_adv

            total_loss.backward()
            optim.step()
            optim.zero_grad()

            loss_history.append([total_loss.item(), 0.])
            g_loss_history.append([g_loss.item(), 0.])
            r_loss_history.append([r_loss.item(), 0.])
            l_loss_history.append([l_loss.item(), 0.])
            r_loss_adv_history.append([r_loss_adv.item(), 0.])

        epoch_losses = np.mean(np.array(loss_history), axis=0)
        r_epoch_losses = np.mean(np.array(r_loss_history), axis=0)
        g_epoch_losses = np.mean(np.array(g_loss_history), axis=0)
        l_epoch_losses = np.mean(np.array(l_loss_history), axis=0)
        r_epoch_adv_losses = np.mean(np.array(r_loss_adv_history), axis=0)

        epoch_gen_image_loss = np.mean(np.array(gen_image_loss_history), axis=0)
        epoch_gen_adv_loss = np.mean(np.array(gen_adv_loss_history), axis=0)
        epoch_loss_G = np.mean(np.array(loss_G_history), axis=0)

        epoch_losses[1] = np.log10(optim.param_groups[0]['lr'])
        epoch_loss_G[1] = np.log10(optim.param_groups[0]['lr'])

        #################
        #     val:    #
        #################
        if i_epoch % c.val_freq == 0 or i_epoch == c.trained_epoch + 1:
            with torch.no_grad():
                psnr_s = []
                psnr_s_adv = []
                psnr_c = []
                psnr_a = []
                net.eval()
                net_Gen.eval()
                i = 0
                for x in datasets.testloader:

                    secret = next(iter(my_datasets.testloader)).to(device)
                    cover = x.to(device)

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

                    steg_transform = net_Gen(steg)
                    output_steg_t = dwt(steg_transform)

                    steg_transform_save = steg_transform

                    #################
                    #   backward:   #
                    #################
                    output_rev_adv = torch.cat((output_steg_t, output_z), 1)
                    output_image = net(output_rev_adv, rev=True)
                    secret_rev_adv = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
                    secret_rev_adv = iwt(secret_rev_adv)
                    secret_rev_adv_save = secret_rev_adv

                    output_steg = dwt(steg)
                    output_rev = torch.cat((output_steg, output_z), 1)
                    output_image = net(output_rev, rev=True)
                    secret_rev = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
                    secret_rev = iwt(secret_rev)
                    secret_rev_save = secret_rev

                    secret_rev = secret_rev.cpu().numpy().squeeze() * 255
                    np.clip(secret_rev, 0, 255)
                    secret_rev_adv = secret_rev_adv.cpu().numpy().squeeze() * 255
                    np.clip(secret_rev_adv, 0, 255)
                    secret = secret.cpu().numpy().squeeze() * 255
                    np.clip(secret, 0, 255)
                    cover = cover.cpu().numpy().squeeze() * 255
                    np.clip(cover, 0, 255)
                    steg = steg.cpu().numpy().squeeze() * 255
                    np.clip(steg, 0, 255)
                    steg_transform = steg_transform.cpu().numpy().squeeze() * 255
                    np.clip(steg_transform, 0, 255)

                    psnr_temp = computePSNR(secret_rev, secret)
                    psnr_s.append(psnr_temp)
                    psnr_temp = computePSNR(secret_rev_adv, secret)
                    psnr_s_adv.append(psnr_temp)
                    psnr_temp_c = computePSNR(cover, steg)
                    psnr_c.append(psnr_temp_c)
                    psnr_stego_adv = computePSNR(steg, steg_transform)
                    psnr_a.append(psnr_stego_adv)

                writer.add_scalars("PSNR_S", {"average psnr": np.mean(psnr_s)}, i_epoch)
                writer.add_scalars("PSNR_S_ADV", {"average psnr": np.mean(psnr_s_adv)}, i_epoch)
                writer.add_scalars("PSNR_C", {"average psnr": np.mean(psnr_c)}, i_epoch)
                writer.add_scalars("PSNR_ADV", {"average psnr": np.mean(psnr_a)}, i_epoch)
                logger_train.info(
                    f"TEST:   "
                    f'PSNR_S: {np.mean(psnr_s):.4f} | '
                    f'PSNR_S_ADV: {np.mean(psnr_s_adv):.4f} | '
                    f'PSNR_C: {np.mean(psnr_c):.4f} | '
                    f'PSNR_ADV: {np.mean(psnr_a):.4f} | '
                )

            # 展示图片
            save_path = os.path.join(c.SAVE_FILE_PATH)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_images = []
            save_images.append(cover_save)
            save_images.append(steg_save)
            diff = (steg_save - cover_save)*10
            save_images.append(diff)
            save_images.append(steg_transform_save)
            diff_adv = (steg_transform_save - steg_save)*10
            save_images.append(diff_adv)
            save_images.append(secret_save)
            save_images.append(secret_rev_save)
            diff_secret = (secret_save - secret_rev_save)*10
            save_images.append(diff_secret)
            save_images.append(secret_rev_adv_save)
            diff_secret_adv = (secret_save - secret_rev_adv_save)*10
            save_images.append(diff_secret_adv)

            save_images = torch.cat(save_images, 0)
            save_path = os.path.join(save_path, 'show_images_epoch%03d.png'%(i_epoch))
            torchvision.utils.save_image(save_images, save_path, nrow = c.batchsize_val, padding = 1, normalize=False)

            net.train()
            net_Gen.train()

        viz.show_loss(epoch_losses)
        writer.add_scalars("Train", {"Train_Loss": epoch_losses[0]}, i_epoch)

        viz.show_loss(epoch_loss_G)
        writer.add_scalars("Train", {"Train_Loss_G": epoch_loss_G[0]}, i_epoch)

        logger_train.info(f"Learning rate: {optim.param_groups[0]['lr']}")
        logger_train.info(
            f"Train epoch {i_epoch}:   "
            f'Loss: {epoch_losses[0].item():.4f} | '
            f'r_Loss: {r_epoch_losses[0].item():.4f} | '
            f'r_Loss_adv: {r_epoch_adv_losses[0].item():.4f} | '
            f'g_Loss: {g_epoch_losses[0].item():.4f} | '
            f'l_Loss: {l_epoch_losses[0].item():.4f} | '
            f'gen_image_Loss: {epoch_gen_image_loss[0].item():.4f} | '
            f'gen_adv_Loss: {epoch_gen_adv_loss[0].item():.4f} | '
            f'Loss_G: {epoch_loss_G[0].item():.4f} | '
        )

        if epoch_losses[0].item() < best_loss:
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict()}, os.path.join(c.SAVE_FILE_PATH, 'model_best_loss' + '.pt'))
            torch.save({'opt': optim_Gen.state_dict(),
                        'net': net_Gen.state_dict()}, os.path.join(c.SAVE_FILE_PATH, 'attack_network_best_loss' + '.pt'))
            logger_train.info(
                f"Train epoch {i_epoch}:   "
                f'Loss: {epoch_losses[0].item():.4f} | '
                f'r_Loss: {r_epoch_losses[0].item():.4f} | '
                f'r_Loss_adv: {r_epoch_adv_losses[0].item():.4f} | '
                f'g_Loss: {g_epoch_losses[0].item():.4f} | '
                f'l_Loss: {l_epoch_losses[0].item():.4f} | '
                f'gen_image_Loss: {epoch_gen_image_loss[0].item():.4f} | '
                f'gen_adv_Loss: {epoch_gen_adv_loss[0].item():.4f} | '
                f'Loss_G: {epoch_loss_G[0].item():.4f} | '
            )
            best_loss = epoch_losses[0].item()


        if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict()}, os.path.join(c.SAVE_FILE_PATH, 'model_checkpoint_%.5i' % i_epoch + '.pt'))
            torch.save({'opt': optim_Gen.state_dict(),
                        'net': net_Gen.state_dict()}, os.path.join(c.SAVE_FILE_PATH, 'attack_network_%.5i' % i_epoch + '.pt'))
        
        if i_epoch == c.trained_epoch + 1:
            torch.save({'opt': optim.state_dict(),
                        'net': net.state_dict()}, os.path.join(c.SAVE_FILE_PATH, 'model_checkpoint_%.5i' % i_epoch + '.pt'))
            torch.save({'opt': optim_Gen.state_dict(),
                        'net': net_Gen.state_dict()}, os.path.join(c.SAVE_FILE_PATH, 'attack_network_%.5i' % i_epoch + '.pt'))

        weight_scheduler.step()

    torch.save({'opt': optim.state_dict(),
                'net': net.state_dict()}, os.path.join(c.SAVE_FILE_PATH, 'model' + '.pt'))
    torch.save({'opt': optim_Gen.state_dict(),
                        'net': net_Gen.state_dict()}, os.path.join(c.SAVE_FILE_PATH, 'attack_network' + '.pt'))
    writer.close()

except:
    if c.checkpoint_on_error:
        torch.save({'opt': optim.state_dict(),
                    'net': net.state_dict()}, os.path.join(c.SAVE_FILE_PATH, 'model_ABORT' + '.pt'))
        torch.save({'opt': optim_Gen.state_dict(),
                        'net': net_Gen.state_dict()}, os.path.join(c.SAVE_FILE_PATH, 'attack_network_ABORT' + '.pt'))
    raise

finally:
    viz.signal_stop()
