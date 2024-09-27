# CIFAR10
# 需要改的参数
# 实验数据存放路径
ROBUST_MODEL_PATH = './exp'
LOG_PATH = ROBUST_MODEL_PATH
# 模型路径
TRAINED_MODEL_PATH = ""
log10_lr = -5.4
tain_next = False
trained_epoch = 0
cuda_id = '0'
secret_imgs_path = './trigger_image/cifar10/airplane_best'

save_config_path = './config.py'
save_runcode_path = ''

# train robust
transform_types = ['none', 'sp', 'cr']
# transform_types = ['none']
rot_degree = 15
shrink_size = 28
robust_crop_szie = 28

cropsize = 32
cropsize_val = 32
batch_size = 16
batchsize_val = 16
TRAIN_PATH = "E:/59data/dataset/hinet_train_dataset/cifar10_hinet/cifar10/cifar10/train_cifar10"
VAL_PATH = "E:/59data/dataset/hinet_train_dataset/cifar10_hinet/cifar10/cifar10/test_cifar10"




# # ImageNet
# # 需要改的参数
# # 实验数据存放路径
# cuda_id = '3'
# LOG_PATH  = '/home/ljh/HiNet/model/imagenet/size_128/single_warplane2/spcr_11-20-9-36'
# ROBUST_MODEL_PATH = LOG_PATH
# # 模型路径
# TRAINED_MODEL_PATH = '/home/ljh/HiNet/model/imagenet/size_128/single_warplane2/spcr_11-18-19-37/model_checkpoint_00080.pt'
# log10_lr = -5.4
# tain_next = True
# trained_epoch = 80
# secret_imgs_path = '/home/ljh/dataset/imagenet_hinet/single_warplane2_128'

# save_config_path = '/home/ljh/HiNet/config.py'
# save_runcode_path = '/home/ljh/HiNet/train_single_img_robust.py'

# # train robust
# transform_types = ['none', 'sp', 'cr']
# # transform_types = ['none']
# # transform_types = ['none', 'rot15', 'sp', 'cr', 'flip']
# rot_degree = 15
# shrink_size = 112
# robust_crop_szie = 112

# my_lamda_reconstruction = 5
# my_lamda_guide = 1
# my_lamda_low_frequency = 1

# cropsize = 128
# cropsize_val = 128
# batchsize_val = 4

# TRAIN_PATH = '/home/ljh/dataset/imagenet_hinet/train_128'
# VAL_PATH = '/home/ljh/dataset/imagenet_hinet/val_2k_128'

# batch_size = 32


lamda_reconstruction = 5
lamda_guide = 1
lamda_low_frequency = 1


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





