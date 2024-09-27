import os
import logging
from datetime import datetime
import shutil
import time

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


# 保存本次实验的代码
def save_current_codes(code_path, des_path):
    root_dir, codefile_name = os.path.split(code_path)  # eg：/n/liyz/videosteganography/
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    codefile_name = time_now + '_' + codefile_name
    new_path = os.path.join(des_path, codefile_name)
    shutil.copyfile(code_path, new_path)
