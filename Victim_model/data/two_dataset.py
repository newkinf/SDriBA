import torch.utils.data as data

from PIL import Image
import os
import os.path

import random
from random import choice
import torchvision.transforms as trans
from skimage import feature
import numpy as np
import torch
import torch.nn.functional as F
import math
from torch.autograd import Variable
from skimage import filters
import cv2
from torch.utils.data import DataLoader
from torchvision.utils import save_image

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def accimage_loader(path):
    print("can't find acc image loader")
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



# define the own imagefolder   from code for torchvision.datasets.folder
class two_folder(data.Dataset):

    def __init__(self, root, root_wm, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)

        classes_wm, class_to_idx_wm = find_classes(root_wm)
        imgs_wm = make_dataset(root_wm, class_to_idx_wm)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx

        self.root_wm = root_wm
        self.imgs_wm = imgs_wm
        self.classes_wm = classes_wm
        self.class_to_idx_wm = class_to_idx_wm

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.len=len(imgs)
    def __getitem__(self, index):

        path, target = self.imgs[index]
        img = self.loader(path)  #size (,,3)
        if self.transform != None:
            img = self.transform(img)

        path_wm, target_wm = self.imgs_wm[index]
        img_wm = self.loader(path_wm)  #size (,,3)
        if self.transform != None:
            img_wm = self.transform(img_wm)

        target_wm = 0
        return img,  target, img_wm,  target_wm

    def __len__(self):
        return self.len


def main():
    # pth = "/data-x/g12/zhangjie/nips/datasets/cifar10png/test"
    # # dataset = ori_folder(pth)
    # # loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    # # for i, data in enumerate(loader, 0):
    # #     img, edge = data
    # #     save_image(img,"./try/%02d_img.png"%i)
    # #     save_image(edge,"./try/%02dedge.png"%i)

    print("ZJ_Dataset")

if __name__ == '__main__':
    main()