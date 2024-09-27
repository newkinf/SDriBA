# CIFAR10
# 需要改的参数
# 实验数据存放路径
cuda_id = '0'
SAVE_FILE_PATH = './exp/'
LOG_PATH = SAVE_FILE_PATH
# 模型路径
TRAINED_MODEL_PATH = "./save_model/sdriba_ftl_cifar10.pt"
log10_lr = -5.4
tain_next = True
trained_epoch = 0
secret_imgs_path = './trigger_image/cifar10/airplane_best'
save_config_path = './config.py'
cropsize = 32
cropsize_val = 32
batch_size = 16
batchsize_val = 16
TRAIN_PATH = "E:/59data/dataset/hinet_train_dataset/cifar10_hinet/cifar10/cifar10/train_cifar10"
VAL_PATH = "E:/59data/dataset/hinet_train_dataset/cifar10_hinet/cifar10/cifar10/test_cifar10"



# # ImageNet
# # 需要改的参数
# # 实验数据存放路径
# cuda_id = "3"
# SAVE_FILE_PATH  = '/home/ljh/sdriba_attack_network/exp/24-9-3-2009'
# LOG_PATH = SAVE_FILE_PATH
# # 模型路径
# TRAINED_MODEL_PATH = '/home/ljh/sdriba_attack_network/save_model/sdriba_imagenet.pt'
# log10_lr = -5.4
# tain_next = True
# trained_epoch = 0
# secret_imgs_path = '/home/ljh/dataset/imagenet_hinet/single_warplane2_128'

# save_config_path = '/home/ljh/sdriba_attack_network/config.py'

# # train robust
# transform_types = ['none', 'sp', 'cr']
# rot_degree = 15
# shrink_size = 112
# robust_crop_szie = 112
# cropsize = 128
# cropsize_val = 128

# batch_size = 16
# batchsize_val = 4

# TRAIN_PATH = '/home/ljh/dataset/imagenet_hinet/train_128'
# VAL_PATH = '/home/ljh/dataset/imagenet_hinet/val_2k_128'




###########################
lamda_reconstruction = 5
lamda_guide = 1
lamda_low_frequency = 1
lamda_reconstruction_adv = 5

###########################
# attack network
attack_network_train_num_per_iter = 5
gen_image_loss_param = 1.0
gen_adv_loss_param = 1e-2



###########################
device_ids = [0]
SAVE_freq = 5
val_freq = 5
# Super parameters
clamp = 2.0
channels_in = 3
lr = 10 ** log10_lr
epochs = 1000
weight_decay = 1e-5
init_scale = 0.01
# Train:
betas = (0.5, 0.999)
weight_step = 1000
gamma = 0.5
# Val:
shuffle_val = False
# Dataset
format_train = 'png'
format_val = 'png'
# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False
# Saving checkpoints:
checkpoint_on_error = True





