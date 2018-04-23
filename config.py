# coding=utf-8
from utils import *
import socket

# image path
hostName = socket.gethostname()
if is_mac():
    PATH_BASE = "/Developer/Python/CVDL_data/HW1_data/"
    PATH_TRAIN_IMAGE = '/Developer/Python/CVDL_data/HW1_data/transformed_train/'
    PATH_VAL_IMAGE = '/Developer/Python/CVDL_data/HW1_data/transformed_validate/'
    PATH_TEST_BASE = '/Developer/Python/CVDL_data/HW1_data/test/'
elif is_linux():
    if hostName == "big-brother":
        PATH_BASE = '/home/sunshine/Programming/CVDL/CVDL_data/HW1_data/'
        PATH_TRAIN_IMAGE = '/home/sunshine/Programming/CVDL/CVDL_data/HW1_data/transformed_train'
        PATH_VAL_IMAGE = '/home/sunshine/Programming/CVDL/CVDL_data/HW1_data/transformed_validate'
        PATH_TEST_BASE = '/home/sunshine/Programming/CVDL/CVDL_data/HW1_data/test/'
    elif hostName == "cvda-server":
        PATH_BASE = '/home/clt/Programming/CVDL/data/'
        PATH_TRAIN_IMAGE = '/home/clt/Programming/CVDL/data/transformed_train'
        PATH_VAL_IMAGE = '/home/clt/Programming/CVDL/data/transformed_validate'
        PATH_TEST_BASE = '/home/clt/Programming/CVDL/data/test'
    elif hostName == "cvda-game":
        PATH_BASE = '/home/clt/Programming/CVDL/data/'
        PATH_TRAIN_IMAGE = '/home/clt/Programming/CVDL/data/transformed_train'
        PATH_VAL_IMAGE = '/home/clt/Programming/CVDL/data/transformed_validate'
        PATH_TEST_BASE = '/home/clt/Programming/CVDL/data/test'
else:
    raise Exception('No image data found.')

# train info
IM_SIZE_299 = 299
BATCH_SIZE = 32
CLASS_NUMBER = 80
EPOCH = 100
NUMBER_OF_TRAIN_IMAGE = 44688
NUMBER_OF_VAL_IMAGE = 11311
