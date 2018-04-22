# coding=utf-8
from utils import *
from utils import context_creator

# image path
if is_mac():
    PATH_BASE = "/Developer/Python/CVDL_data/HW1_data/"
    PATH_TRAIN_IMAGE = '/Developer/Python/CVDL_data/HW1_data/transformed_train/'
    PATH_VAL_IMAGE = '/Developer/Python/CVDL_data/HW1_data/transformed_validate/'
    PATH_TEST_BASE = ''
elif is_linux():
    PATH_TRAIN_BASE = ''
    PATH_TRAIN_IMAGE = ''
    PATH_VAL_IMAGE = ''
    PATH_TEST_BASE = ''
else:
    raise Exception('No image data found.')

# train info
IM_SIZE_299 = 299
IM_SIZE_224 = 224
BATCH_SIZE = 32
CLASS_NUMBER = 80
EPOCH = 100
NUMBER_OF_TRAIN_IMAGE = 44688
NUMBER_OF_VAL_IMAGE = 11311
